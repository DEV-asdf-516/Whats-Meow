import logging
from pathlib import Path
from typing import List
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import librosa
import soundfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


argumentor = Compose(
    [
        TimeStretch(min_rate=0.9, max_rate=1.0, p=0.8),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.8),
        AddGaussianNoise(min_amplitude=0.1, max_amplitude=0.5, p=0.6),
        Shift(min_shift=-0.2, max_shift=0.2, p=0.6),
    ],
    p=1.0,
)


def expand_data(data_path: str, argumentor: Compose, multiplier: int = 2):

    expanded_count = 0

    audio_files = [
        p
        for p in Path(data_path).rglob("*")
        if p.suffix.lower() in {".wav", ".mp3", ".m4a"}
    ]

    for audio_file in audio_files:
        try:
            # 오디오 증강 샘플 생성
            audio, sr = librosa.load(audio_file, sr=16000)

            for i in range(multiplier):
                augmented = argumentor(samples=audio, sample_rate=sr)
                aug_file = f"{audio_file.stem}_aug{i+1}{audio_file.suffix}"
                soundfile.write(data_path / aug_file, augmented, sr)
                expanded_count += 1

        except Exception as e:
            logger.error(f"Error augmenting {audio_file}: {e}")


# 기준 데이터셋 확장
def expand_seed(
    data_path: str,
    categories: List[str],
    argumentor: Compose,
    multiplier: int = 2,
):
    # 3x_Aug
    expanded_count = 0

    for category in categories:
        category_path: Path = Path(data_path) / category

        if not category_path.exists():
            logger.warning(f"Category path {category_path} does not exist.")
            continue

        audio_files = [
            f
            for f in category_path.iterdir()
            if f.is_file() and f.suffix.lower() in {".wav", ".mp3", ".m4a"}
        ]

        for audio_file in audio_files:
            try:
                # 오디오 증강 샘플 생성
                audio, sr = librosa.load(audio_file, sr=16000)

                for i in range(multiplier):
                    augmented = argumentor(samples=audio, sample_rate=sr)
                    aug_file = f"{audio_file.stem}_aug{i+1}{audio_file.suffix}"
                    aug_path = category_path / aug_file
                    soundfile.write(aug_path, augmented, sr)
                    expanded_count += 1

            except Exception as e:
                logger.error(f"Error augmenting {audio_file}: {e}")
