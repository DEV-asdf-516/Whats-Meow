from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler
from sklearn.svm import SVC
import torch
import pickle
from collections import defaultdict

from tqdm import tqdm

from labeling.cdbn.cdbn import (
    CDBN,
    apply_fdap,
    extract_cdbn_features,
    extract_mel_spectrogram,
    load_pretrained_cdbn_model,
)

logger = logging.getLogger(__name__)


def get_cdbn_classifiers(categories):
    cdbn_classifiers = {
        "knn": KNeighborsClassifier(n_neighbors=min(5, len(categories))),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
        "lda": LinearDiscriminantAnalysis(),
        "svm": SVC(kernel="rbf", probability=True, random_state=42),
    }
    return cdbn_classifiers


class CDBNProgressiveLearner:
    def __init__(
        self,
        categories: List[str],
        dataset_path: str,
        raw_dataset_path: str,
        output_path: str,
        save_model_path: str,
        pretrained_model_path: str,
        batch_size: int = 30,
        ensemble_weights: Dict[str, float] = {},
        retrain_cdbn_count: int = 0,  # 0이면 재학습 안함
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.categories = categories
        self.dataset_path: Path = Path(dataset_path)
        self.unlabeled_path: Path = Path(raw_dataset_path)
        self.output_path: Path = Path(output_path)
        self.save_model_path: Path = Path(save_model_path)
        self.pretrained_model_path: Path = Path(pretrained_model_path)
        self.batch_size = batch_size
        self.retrain_cdbn_count = retrain_cdbn_count
        self.classifiers = get_cdbn_classifiers(self.categories)

        self.current_batch_idx = 0
        self.total_processed = 0
        self.processed_files = set()

        self.state_file = self.output_path / "processing_state.json"
        self.processed_file_logs = self.output_path / "processed_files.txt"
        self.ensemble_weights = ensemble_weights

        if not self.pretrained_model_path.exists():
            raise FileNotFoundError(f"Pretrained model not found at {self.pretrained_model_path}") # fmt: skip
        self.cdbn = load_pretrained_cdbn_model(self.pretrained_model_path)

    def initialize_model(self):
        if self.state_file.exists():
            self.load_processing_state()
            self.load_model_state()
        else:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.build_reference_fetaures()
            self._train_initial_classifiers()
            self.save_processing_state()
            self.save_model_state()

    def retrain(self):
        if self.state_file.exists():
            self.load_processing_state()
            self.load_model_state()
            self.build_reference_fetaures()
            self._train_initial_classifiers()
            self.save_model_state()

    def _train_initial_classifiers(self):
        # 초기 분류기 학습
        X_cdbn, y = [], []

        for category in self.categories:
            for features in self.reference_features[category]:
                X_cdbn.append(features)
                y.append(category)

        if len(X_cdbn) <= 0:
            logger.warning("No CDBN features found for training classifiers.")
            return

        X_cdbn = np.array(X_cdbn)
        self.scaler = StandardScaler()
        X_cdbn_scaled = self.scaler.fit_transform(X_cdbn)

        for _, clf in self.classifiers.items():
            logger.info(f"Training classifier: {clf.__class__.__name__}")
            clf.fit(X_cdbn_scaled, y)

    def build_reference_fetaures(self):
        # 기준 데이터셋에서 CDBN 특징 추출
        logger.info("Building reference features for CDBN classifiers...")
        self.reference_features = {category: [] for category in self.categories}

        for category in self.categories:
            category_path = self.dataset_path / category

            if not category_path.exists():
                logger.warning(f"Category path {category_path} does not exist. Skipping.") # fmt: skip
                continue

            catrgory_audio_files = [
                p
                for p in Path(category_path).rglob("*")
                if p.suffix.lower() in {".wav", ".mp3", ".m4a"}
            ]

            for audio_file in catrgory_audio_files:
                features = extract_cdbn_features(
                    device=self.device,
                    model=self.cdbn,
                    audio_path=audio_file,
                )
                if len(features) <= 0:
                    logger.warning(f"No features extracted for {audio_file}. Skipping.")
                    continue
                self.reference_features[category].append(features)

    def save_model_state(self):
        model_state_path = self.save_model_path / f"batch_{self.current_batch_idx:03d}" # fmt: skip

        logger.info(f"Saving model state to {model_state_path}")
        model_state_path.mkdir(parents=True, exist_ok=True)

        try:
            with open(model_state_path / "classifiers.pkl", "wb") as f:
                pickle.dump(self.classifiers, f)

            with open(model_state_path / "scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)

            with open(model_state_path / "reference_features.pkl", "wb") as f:
                pickle.dump(self.reference_features, f)

        except Exception as e:
            logger.error(f"Failed to save classifiers: {e}")
            return

    def load_model_state(self):
        model_state_path = self.save_model_path / f"batch_{self.current_batch_idx:03d}"
        logger.info(f"Loading model state from {model_state_path}")
        if not model_state_path.exists():
            logger.error(f"Model state path does not exist: {model_state_path}")
            return

        try:
            with open(model_state_path / "classifiers.pkl", "rb") as f:
                self.classifiers = pickle.load(f)

            with open(model_state_path / "scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)

            with open(model_state_path / "reference_features.pkl", "rb") as f:
                self.reference_features = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load model state: {e}")
            return

    def save_processing_state(self):
        state = {
            "current_batch_idx": self.current_batch_idx,
            "total_processed": self.total_processed,
            "last_updated": datetime.now().isoformat(),
            "categories": self.categories,
            "batch_size": self.batch_size,
        }

        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def load_processing_state(self):
        with open(self.state_file, "r", encoding="utf-8") as f:
            state: Dict = json.load(f)
            self.current_batch_idx = state.get("current_batch_idx", 0)
            self.total_processed = state.get("total_processed", 0)
            logger.info(f"Loaded processing state: {state}")

        if self.processed_file_logs.exists():
            with open(self.processed_file_logs, "r", encoding="utf-8") as f:
                self.processed_files = set(line.strip() for line in f)

    def process_next_batch(self) -> Dict:
        results = []

        logger.info(f"배치 {self.current_batch_idx + 1} 자동 분류 시작")

        unlabeld_audio_files = [
            p
            for p in Path(self.unlabeled_path).rglob("*")
            if p.suffix.lower() in {".wav", ".mp3", ".m4a"}
        ]

        unlabeld_audio_files = sorted(unlabeld_audio_files)

        unprocessed_files = [
            f for f in unlabeld_audio_files if str(f) not in self.processed_files
        ]

        batch_files = unprocessed_files[: self.batch_size]

        if not batch_files:
            logger.info("Nothing to process in this batch.")
            return

        for audio_file in tqdm(batch_files, leave=True):
            features = extract_cdbn_features(
                device=self.device,
                model=self.cdbn,
                audio_path=audio_file,
            )

            features_scaled = self.scaler.transform(features.reshape(1, -1))

            predictions = {}
            confidences = {}

            for name, clf in self.classifiers.items():
                pred = clf.predict(features_scaled)[0]

                if hasattr(clf, "predict_proba"):
                    prob = clf.predict_proba(features_scaled)[0]
                    confidence = np.max(prob)
                else:
                    confidence = 1.0

                weighted_confidence = confidence * self.ensemble_weights.get(name, 0.2)

                predictions[name] = pred
                confidences[name] = weighted_confidence

            ensemble_predictions, ensemble_confidence = self.ensemble_predict(predictions, confidences) # fmt: skip

            results.append(
                {
                    "file_path": str(audio_file),
                    "status": "success",
                    "predictions": predictions,
                    "confidences": confidences,
                    "ensemble_prediction": ensemble_predictions,
                    "ensemble_confidence": ensemble_confidence,
                }
            )

        self.current_batch_idx += 1
        self.total_processed += len(batch_files)
        self.save_processing_state()
        self.save_model_state()

        with open(self.processed_file_logs, "a", encoding="utf-8") as f:
            for file_path in batch_files:
                f.write(f"{file_path}\n")
                self.processed_files.add(str(file_path))

        with open(
            Path(f"{self.output_path}/batch_{self.current_batch_idx:03d}_result.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                results,
                f,
                indent=2,
                ensure_ascii=False,
                default=lambda o: (
                    float(o) if isinstance(o, (np.float32, np.float64)) else str(o)
                ),
            )

        return {
            "results": results,
            "batch_idx": self.current_batch_idx,
            "files_processed": len(batch_files),
        }

    def ensemble_predict(
        self, predictions: Dict, confidences: Dict
    ) -> Tuple[str, float]:
        votes = defaultdict(float)
        total_weight = 0.0

        for clf, pred in predictions.items():
            weight = self.ensemble_weights.get(clf, 0.2)
            confidence_weight = confidences[clf]
            final_weight = weight * confidence_weight
            votes[pred] += final_weight
            total_weight += final_weight

        ensemble_pred = max(votes.items(), key=lambda x: x[1])[0]
        ensemble_conf = votes[ensemble_pred] / total_weight if total_weight > 0 else 0

        return ensemble_pred, ensemble_conf
