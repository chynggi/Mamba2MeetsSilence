# BSMamba2 음성 분리 모델 구현 가이드

## 프로젝트 개요
BSMamba2는 음악에서 보컬을 분리하는 State Space Model 기반의 딥러닝 모델입니다. Mamba2 아키텍처와 Band-splitting 전략을 결합하여 intermittent vocal 처리에 우수한 성능을 보입니다.

## 아키텍처 구성
- **Band-Split Module**: 주파수 축을 62개 서브밴드로 분할
- **Dual-Path Module**: Mamba2 블록을 이용한 시간-주파수 의존성 모델링  
- **Mask Estimation Module**: 시간-주파수 마스크 생성

## 핵심 구현 요구사항

### 1. 프로젝트 구조
```
bsmamba2/
├── models/
│   ├── __init__.py
│   ├── bsmamba2.py          # 메인 모델
│   ├── mamba2.py            # Mamba2 블록 구현
│   └── components.py        # Band-split, Mask estimation 모듈
├── data/
│   ├── __init__.py
│   ├── dataset.py           # MUSDB18HQ 데이터 로더
│   └── transforms.py        # 오디오 전처리
├── training/
│   ├── __init__.py
│   ├── train.py            # 학습 스크립트
│   ├── loss.py             # 손실 함수
│   └── metrics.py          # 평가 메트릭
├── inference/
│   ├── __init__.py
│   └── separate.py         # 추론 스크립트
├── utils/
│   ├── __init__.py
│   ├── audio_utils.py      # STFT/ISTFT 유틸리티
│   └── config.py           # 설정 관리
├── configs/
│   └── bsmamba2.yaml       # 하이퍼파라미터 설정
├── requirements.txt
├── README.md
└── setup.py
```

### 2. 모델 구현 세부사항

#### BSMamba2 메인 모델 (`models/bsmamba2.py`)
- Input: Complex spectrogram (T × F)
- STFT parameters: window_size=2048, hop_size=441, sr=44100
- Hidden dimension: 256
- Sub-bands: 62개
- Dual-path modules: 6층

#### Mamba2 블록 (`models/mamba2.py`)
- State space model with selective updates
- Bidirectional processing (forward + backward)
- Scalar state transition matrix A = aI
- Input-dependent parameters Δ

#### 컴포넌트 구현 (`models/components.py`)
```python
class BandSplitModule:
    # 주파수 축을 K개 서브밴드로 분할
    # 각 서브밴드를 MLP로 feature extraction
    
class DualPathModule:
    # 시간축과 밴드축에서 bidirectional Mamba2 적용
    # L번 반복 처리
    
class MaskEstimationModule:
    # RMSNorm + Linear layers + Tanh + GLU
    # 서브밴드별 마스크 생성 후 concatenation
```

### 3. 데이터 처리

#### 데이터셋 (`data/dataset.py`)
- MUSDB18HQ 데이터셋 지원
- Train: 86 tracks, Validation: 14 tracks, Test: 50 tracks
- 8초 세그먼트로 분할
- 4개 소스 (vocals, drums, bass, other) 랜덤 믹싱

#### 전처리 (`data/transforms.py`)
- 44.1kHz stereo 오디오 처리
- STFT 변환 (window=2048, hop=441)
- 데이터 증강 기법들

### 4. 학습 구성

#### 손실 함수 (`training/loss.py`)
```python
def bsmamba2_loss(pred, target, lambda_time=10):
    # L1 time domain loss
    time_loss = torch.mean(torch.abs(pred - target))
    
    # Multi-resolution STFT loss
    stft_loss = 0
    for win_size in [4096, 2048, 1024, 512, 256]:
        # hop_size = 147
        stft_loss += stft_l1_loss(pred, target, win_size, 147)
    
    return lambda_time * time_loss + stft_loss
```

#### 메트릭 (`training/metrics.py`)
- cSDR (chunk-level SDR): 1초 청크별 median SDR
- uSDR (utterance-level SDR): 전체 트랙의 mean SDR

#### 학습 설정 (`training/train.py`)
- Optimizer: AdamW
- Learning rate: 5e-4
- Precision: bfloat16
- Batch size per GPU: 5
- Gradient accumulation: 6 steps
- 총 파라미터: 약 48.1M

### 5. 추론 및 검증

#### 추론 스크립트 (`inference/separate.py`)
- 8초 비중복 세그먼트로 처리
- Sequential concatenation으로 전체 오디오 재구성
- ISTFT로 시간 도메인 신호 복원

#### 검증 (`training/validate.py`)
- 다양한 입력 길이 (1-16초) 테스트
- Vocal onset duration별 성능 분석

### 6. 설정 관리

#### 하이퍼파라미터 (`configs/bsmamba2.yaml`)
```yaml
model:
  hidden_dim: 256
  num_layers: 6
  num_subbands: 62
  
audio:
  sample_rate: 44100
  n_fft: 2048
  hop_length: 441
  segment_length: 8  # seconds

training:
  batch_size: 5
  gradient_accumulation_steps: 6
  learning_rate: 5e-4
  precision: "bf16"
  lambda_time: 10
  
loss:
  stft_windows: [4096, 2048, 1024, 512, 256]
  stft_hop: 147
```

### 7. 의존성 관리 (`requirements.txt`)
```
torch>=2.0.0
torchaudio>=2.0.0
librosa>=0.10.0
numpy>=1.21.0
scipy>=1.7.0
soundfile>=0.10.0
pyyaml>=6.0
tensorboard>=2.8.0
musdb>=0.4.0
museval>=0.4.0
einops>=0.6.0
```

### 8. 주요 구현 포인트

1. **Mamba2 구현**: causal_conv1d와 mamba_ssm 라이브러리 활용
2. **Band-splitting**: 주파수 범위를 균등 또는 Mel-scale로 분할
3. **Dual-path**: 시간축 → 밴드축 순서로 처리
4. **메모리 효율성**: gradient checkpointing 적용
5. **혼합 정밀도**: bfloat16으로 학습 안정성 확보

### 9. 성능 목표
- cSDR: 11.03 dB (SOTA)
- uSDR: 10.70 dB
- 다양한 입력 길이에서 안정적인 성능
- Intermittent vocal에서 BS-RoFormer 대비 우수한 성능

### 10. 개발 우선순위
1. Mamba2 핵심 블록 구현
2. Band-split 모듈과 Dual-path 구조
3. 손실 함수와 메트릭
4. MUSDB18HQ 데이터 로더
5. 학습 파이프라인
6. 추론 및 평가 스크립트

## 코딩 스타일
- Type hints 사용
- Docstring (Google 스타일)
- 모듈화된 설계
- 설정 파일 기반 하이퍼파라미터 관리
- 로깅과 체크포인트 지원

이 가이드를 따라 BSMamba2 모델의 전체 구현을 완성하세요.