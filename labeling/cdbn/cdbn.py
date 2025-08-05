import logging
from pathlib import Path
from typing import Dict
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# CPU 사용
class CDBN(nn.Module):
    # 5층 CDBN 구조
    # [3×7], [3×3], [3×3], [3×3], [3×3]
    def __init__(self):
        super().__init__()
        self.layer1 = CRBM(1, 50, kernel_size=(3, 7))
        self.layer2 = CRBM(50, 50, kernel_size=(3, 3))
        self.layer3 = CRBM(50, 50, kernel_size=(3, 3))
        self.layer4 = CRBM(50, 50, kernel_size=(3, 3))
        self.layer5 = CRBM(50, 50, kernel_size=(3, 3))

        self.pool = nn.MaxPool2d((2, 2))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        features = []

        # Layer-wise feature extraction
        h1 = self.layer1(x)
        assert not torch.isnan(h1).any(), "NaN in h1"
        h1_pooled = self.pool(h1)
        features.append(h1_pooled)

        h2 = self.layer2(h1_pooled)
        assert not torch.isnan(h2).any(), "NaN in h2"
        h2_pooled = self.pool(h2)
        features.append(h2_pooled)

        h3 = self.layer3(h2_pooled)
        assert not torch.isnan(h3).any(), "NaN in h3"
        h3_pooled = self.pool(h3)
        features.append(h3_pooled)

        h4 = self.layer4(h3_pooled)
        assert not torch.isnan(h4).any(), "NaN in h4"
        h4_pooled = self.pool(h4)
        features.append(h4_pooled)

        h5 = self.layer5(h4_pooled)
        assert not torch.isnan(h5).any(), "NaN in h5"
        h5_pooled = self.pool(h5)
        features.append(h5_pooled)

        return features

    def pretrain(
        self,
        spectrograms,
        epochs: int,
        batch_size: int,
        lr: float,
        cd_k: int = 3,
    ):
        total_samples = len(spectrograms)
        try:
            for i in range(5):
                layer: CRBM = getattr(self, f"layer{i + 1}")
                optimizer = torch.optim.Adam(layer.parameters(), lr=lr)
                layer.train()
                logger.info(f"Pretraining Layer {i + 1}/5...")

                for e in tqdm(range(epochs), desc=f"Epoch CDBN model..."):
                    total_loss = 0
                    num_batches = 0
                    for batch_start in range(0, total_samples, batch_size):

                        batch_end = min(batch_start + batch_size, total_samples)
                        batch_specs = spectrograms[batch_start:batch_end]
                        batch_data = torch.stack(
                            [
                                torch.FloatTensor(spec).unsqueeze(0)
                                for spec in batch_specs
                            ]
                        )
                        batch_data = batch_data.to(self.device)
                        current_input = batch_data
                        with torch.no_grad():
                            for prev_layer_idx in range(i):
                                prev_layer = getattr(self, f"layer{prev_layer_idx + 1}")
                                current_input = prev_layer.forward(current_input)
                                if prev_layer_idx < 4:
                                    current_input = self.pool(current_input)
                        # CD 학습
                        h_pos, h_neg, v_neg, v_pos = layer.contrastive_divergence(
                            current_input,
                            k=cd_k,
                        )
                        # Loss 계산
                        positive_phase = torch.mean(torch.sum(v_pos * h_pos, dim=[2, 3])) # fmt: skip
                        negative_phase = torch.mean(torch.sum(v_neg * h_neg, dim=[2, 3])) # fmt: skip
                        loss = -(positive_phase - negative_phase)
                        # 역전파
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(layer.parameters(), max_norm=1.0)
                        optimizer.step()

                        total_loss += loss.item()
                        num_batches += 1
                        # 메모리 정리
                        del (
                            batch_data,
                            current_input,
                            h_pos,
                            h_neg,
                            v_neg,
                            v_pos,
                        )
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                layer.eval()
        except Exception as e:
            logger.error(f"Error during pretraining: {e}")
            return False
        return True


class CRBM(nn.Module):
    """
    Convolutional Restricted Boltzmann Machine 구현
    """

    def __init__(self, visible_channels, hidden_channels, kernel_size):
        super().__init__()
        self.visible_channels = visible_channels
        self.hidden_channels = hidden_channels
        self.kernel_height, self.kernel_width = kernel_size

        # 가중치 초기화
        self.w = nn.Parameter(
            torch.randn(
                hidden_channels,
                visible_channels,
                self.kernel_height,
                self.kernel_width,
            )
            * 0.01
        )
        self.visible_bias = nn.Parameter(torch.zeros(visible_channels))
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_channels))
        self.padding_h = self.kernel_height // 2
        self.padding_w = self.kernel_width // 2

    def forward(self, visible):
        return torch.sigmoid(
            F.conv2d(
                visible,
                self.w,
                bias=self.hidden_bias,
                padding=(self.padding_h, self.padding_w),
            )
        )

    def sample_hidden(self, visible):
        hidden_prob = self.forward(visible)
        hidden_sample = torch.bernoulli(hidden_prob)
        return hidden_prob, hidden_sample

    def sample_visible(self, hidden):
        visible_prob = torch.sigmoid(
            F.conv_transpose2d(
                hidden,
                self.w,
                bias=self.visible_bias,
                padding=(self.padding_h, self.padding_w),
            )
        )
        visible_sample = torch.bernoulli(visible_prob)
        return visible_prob, visible_sample

    def contrastive_divergence(self, visible, k=3):
        v_data = visible.clone()
        h_pos, _ = self.sample_hidden(v_data)
        v = v_data.clone()
        for _ in range(k):
            _, h_sample = self.sample_hidden(v)
            _, v_sample = self.sample_visible(h_sample)
            v = v_sample  # 다음 단계로 전파

        h_neg = self.forward(v)
        return h_pos, h_neg, v, v_data


def extract_mel_spectrogram(audio_path: Path):
    # Mel-spectrogram 추출
    target_length = 155  # 시간 축의 고정 길이
    audio, sr = librosa.load(audio_path, sr=16000)

    if len(audio) < 1024:
        audio = np.pad(audio, (0, 1024 - len(audio)), mode="constant")

    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=96,  # 최종 출력 스펙트로그램의 세로 축 해상도
        n_fft=2048,  # 한 번에 처리할 샘플 수
        hop_length=512,  # 시간 해상도 조절
    )

    mel_spectrogram = np.maximum(mel_spectrogram, 1e-10)

    # 로그 스케일
    mel_spec_log_scaled = librosa.power_to_db(
        mel_spectrogram, ref=np.max(mel_spectrogram)
    )

    # 길이 조정 (Zero padding both sides)
    if mel_spec_log_scaled.shape[1] < target_length:
        padding = target_length - mel_spec_log_scaled.shape[1]
        pad_left = padding // 2
        pad_right = padding - pad_left
        mel_spec_log_scaled = np.pad(
            mel_spec_log_scaled,
            ((0, 0), (pad_left, pad_right)),
            mode="constant",
            constant_values=-80.0,
        )
    else:
        start_idx = (mel_spec_log_scaled.shape[1] - target_length) // 2
        mel_spec_log_scaled = mel_spec_log_scaled[:, start_idx : start_idx + target_length] # fmt: skip

    if mel_spec_log_scaled is None:
        return {"success": False}

    mean = np.mean(mel_spec_log_scaled)
    std = np.std(mel_spec_log_scaled)

    mel_spec_whitened = (
        mel_spec_log_scaled - mean
        if std <= 1e-8
        else (mel_spec_log_scaled - mean) / std
    )
    mel_spec_whitened = np.nan_to_num(mel_spec_whitened, nan=0.0)

    return {
        "success": True,
        "spectrogram": mel_spec_whitened,
    }


def train_cdbn_model(
    dataset_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    model: CDBN = CDBN(),
):
    model.to(model.device)
    audio_files = [
        p
        for p in Path(dataset_path).rglob("*")
        if p.suffix.lower() in {".wav", ".mp3", ".m4a"}
    ]
    spectrograms = []
    for audio_file in tqdm(audio_files, desc=f"Train CDBN model..."):
        spectrogram = extract_mel_spectrogram(audio_file)
        if not spectrogram:
            logger.warning(f"Failed to extract spectrogram from {audio_file}")
            continue
        spectrograms.append(spectrogram["spectrogram"])

    if not spectrograms:
        logger.error("No valid spectrograms found for training.")
        return

    success = model.pretrain(
        spectrograms=spectrograms,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )
    if not success:
        logger.error("CDBN pretraining failed.")
    else:
        logger.info("CDBN model pretraining completed successfully.")
        save_path = Path("./model/trained_cdbn_model.pth")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)


def load_pretrained_cdbn_model(model_path: Path):
    model = CDBN()
    if not model_path.exists():
        logger.error(f"Model file {model_path} does not exist.")
        return None
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        logger.info(f"Pretrained CDBN model loaded from {model_path}.")
        return model.to(model.device)
    except Exception as e:
        logger.error(f"Failed to load pretrained CDBN model: {e}")
        return None


def apply_fdap(feature_map: np.ndarray, num_bands: int = 4) -> np.ndarray:
    # 주파수 차원을 여러 대역으로 나누어 각각에 GAP 적용
    target_size = 50
    if len(feature_map.shape) == 3:  # (frequency, time, channel)
        freq_dim, time_dim, channel_dim = feature_map.shape
    elif len(feature_map.shape) == 2:  # (frequency, time)
        freq_dim, time_dim = feature_map.shape
        channel_dim = 1
        feature_map = feature_map.reshape(freq_dim, time_dim, channel_dim)

    if np.isnan(feature_map).any():
        feature_map = np.nan_to_num(feature_map, nan=0.0)

    # 주파수 대역 분할
    band_size = max(1, freq_dim // num_bands)
    fdap_features = []
    for i in range(num_bands):
        start_idx = i * band_size
        if i == num_bands - 1:  # 마지막 대역은 남은 모든 주파수 포함
            end_idx = freq_dim
        else:
            end_idx = (i + 1) * band_size

        # 각 대역에서 GAP 적용
        band_slice = feature_map[start_idx:end_idx, :, :]
        gap_features = (
            np.mean(np.nan_to_num(band_slice), axis=(0, 1))
            if not np.isnan(band_slice).all()
            else np.zeros(channel_dim)
        )

        if len(gap_features) < target_size:
            gap_features = np.pad(
                gap_features,
                (0, target_size - len(gap_features)),
                mode="constant",
                constant_values=0.0,
            )
        elif len(gap_features) > target_size:
            gap_features = gap_features[:target_size]

        fdap_features.append(gap_features)

    result = np.concatenate(fdap_features)

    expected_size = num_bands * target_size
    if len(result) != expected_size:
        logger.warning(
            f"FDAP feature size mismatch: got {len(result)}, expected {expected_size}"
        )
        if len(result) < expected_size:
            padding = expected_size - len(result)
            result = np.pad(result, (0, padding), mode="constant", constant_values=0.0)
        else:
            result = result[:expected_size]

    return result


def extract_cdbn_features(device: torch.device, model: CDBN, audio_path: Path):
    try:
        mel_spec_whitened = extract_mel_spectrogram(audio_path)  # fmt: skip

        mel_tensor = (
            torch.FloatTensor(mel_spec_whitened["spectrogram"])
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)  # fmt: skip
        )

        with torch.no_grad():
            layer_features = model(mel_tensor)

        # FDAP 적용
        fdap_features = []
        expected_dims = []

        for i, feature_map in enumerate(layer_features):
            feature_np = feature_map.squeeze(0).permute(1, 2, 0).cpu().numpy()
            feature_np = np.nan_to_num(feature_np, nan=0.0)
            fdap_feature: np.ndarray = None
            if i < 3:  # Layer 1-3: 4개 균등 대역
                fdap_feature = apply_fdap(feature_np, num_bands=4)
                expected_dims.append(200)  # 4 × 50
            elif i == 3:  # Layer 4: 3개 대역
                fdap_feature = apply_fdap(feature_np, num_bands=3)
                expected_dims.append(150)  # 3 × 50
            else:  # Layer 5: 2개 대역
                fdap_feature = apply_fdap(feature_np, num_bands=2)
                expected_dims.append(100)  # 2 × 50

            if len(fdap_feature) != expected_dims[i]:
                logger.warning(
                    f"Layer {i+1} FDAP dimension: got {len(fdap_feature)}, expected {expected_dims[i]}"
                )
            fdap_features.append(fdap_feature)

        combined_features = np.concatenate(fdap_features)

        expected_dim = 850
        if len(combined_features) != expected_dim:
            logger.warning( f"Total feature dimension: got {len(combined_features)}, expected {expected_dim}") # fmt: skip
            if len(combined_features) < expected_dim:
                padding = expected_dim - len(combined_features)
                combined_features = np.pad(
                    combined_features,
                    (0, padding),
                    mode="constant",
                    constant_values=0.0,
                )
            else:
                combined_features = combined_features[:expected_dim]

        combined_features = np.nan_to_num(combined_features, nan=0.0)

        return combined_features

    except Exception as e:
        logger.error(f"Error extracting CDBN features from {audio_path}: {e}") # fmt: skip
