# Whats Meow?

## 고양이 울음소리 감정 분석 접근

### 1. CDBN

> [Domestic Cat Sound Classification Using Learned Features from Deep Neural Nets](https://www.mdpi.com/2076-3417/8/10/1949)
> 에서 언급된 CDBN 구조 기반 구현

- 고양이 소리 3,525개를 활용한 layer-wise 사전학습(pretraining)
- 각 소리를 Mel-spectrogram으로 변환 후, 5단계 CRBM 계층을 통해 특징 추출
- 추출된 특징은 감정 분류기(classifier)의 입력으로 활용

### 데이터 보강

감정 분류를 위한 라벨링 데이터가 부족하여, 기준 데이터셋을 기반으로 카테고리별 3배(AUG×3) 증강 수행

### 분류기 학습 전략

- 30개 단위로 수동 라벨링하여 신뢰도 높은 샘플을 확보
- 점진적 학습 방식으로 분류기 성능을 반복적으로 개선

### 2. Wav2Vec2 모델

- HuggingFace의 사전학습된 Wav2Vec2 음성 인식 모델 기반
- 고양이 소리 3,525개를 활용해 도메인 특화 `self-supervised` 전이학습 수행
