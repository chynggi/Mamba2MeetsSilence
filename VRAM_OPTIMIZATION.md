# VRAM 최적화 가이드

## 적용된 최적화 사항

### 1. 배치 크기 및 Gradient Accumulation
- **배치 크기**: 5 → **2** (60% 감소)
- **Gradient Accumulation**: 6 → **15** (효과적인 배치 크기는 동일하게 유지: 30)
- **메모리 절감**: ~60% VRAM 감소

### 2. 모델 크기 축소
- **Hidden Dimension**: 256 → **192** (25% 감소)
- **레이어 수**: 6 → **4** (33% 감소)
- **서브밴드 수**: 62 → **48** (23% 감소)
- **파라미터 수**: ~48M → **~27M** (약 44% 감소)

### 3. 세그먼트 길이 감소
- **오디오 세그먼트**: 8초 → **4초** (50% 감소)
- **시간축 프레임 수**: ~808 → **~404** (메모리 50% 감소)

### 4. Gradient Checkpointing
- **DualPathModule**에 gradient checkpointing 적용
- 학습 시 메모리 사용량 30-50% 감소
- 계산 시간 약간 증가 (10-20%)

### 5. DataLoader 최적화
- **num_workers**: 4 → **2** (CPU 메모리 절약)
- **prefetch_factor**: 기본값 → **2** (메모리 prefetch 감소)

### 6. Mixed Precision Training
- **BFloat16** 사용으로 메모리 50% 절약
- PyTorch 2.x의 새로운 AMP API 사용

## 예상 VRAM 사용량

### 이전 설정 (32GB 초과)
- 모델: ~4.8GB
- Optimizer states: ~9.6GB
- Activations (batch=5, 8초): ~18GB
- **총합**: ~32GB+ ❌

### 현재 설정 (32GB 이내)
- 모델: ~2.7GB (44% 감소)
- Optimizer states: ~5.4GB (44% 감소)
- Activations (batch=2, 4초): ~4.5GB (75% 감소)
- Gradient checkpointing: ~3GB (저장)
- **총합**: ~22-24GB ✅

## 추가 최적화 옵션

메모리가 여전히 부족한 경우:

### 1. 배치 크기를 1로 감소
```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 30
```

### 2. 세그먼트를 2초로 감소
```yaml
audio:
  segment_length: 2
```

### 3. Hidden dimension을 더 축소
```yaml
model:
  hidden_dim: 128
  num_subbands: 32
```

### 4. Flash Attention 사용
Mamba2에 Flash Attention 적용 (이미 구현되어 있음)

## 성능 영향

### 훈련 속도
- Gradient accumulation으로 인한 영향: 미미
- Gradient checkpointing: 10-20% 느림
- 작은 배치: 5-10% 느림
- **전체**: 15-30% 느림 (메모리 절약 대비 acceptable)

### 모델 성능
- 파라미터 감소로 최종 성능 약간 하락 예상
- cSDR: 11.03 → ~10.5-10.8 dB (예상)
- 더 많은 epoch로 보완 가능

## 모니터링

학습 중 VRAM 사용량 확인:
```bash
watch -n 1 nvidia-smi
```

또는 Python에서:
```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
```

## 적용 방법

변경사항은 이미 적용되었습니다. 바로 학습을 시작하세요:

```bash
python examples/train_example.py
```

학습 중 메모리 사용량을 모니터링하고 필요시 추가 최적화를 적용하세요.
