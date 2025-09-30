# BSMamba2 Project Structure Summary

This document provides an overview of the complete BSMamba2 project structure.

## 📁 Directory Structure

```
bsmamba2/
├── .github/                    # GitHub configuration
│   └── copilot-instructions.md # Copilot instructions
├── models/                     # Model implementations
│   ├── __init__.py            # Package initialization
│   ├── bsmamba2.py            # Main BSMamba2 model (forward pass, separation)
│   ├── mamba2.py              # Mamba2 State Space Model block
│   └── components.py          # BandSplit, DualPath, MaskEstimation modules
├── data/                       # Data loading and preprocessing
│   ├── __init__.py            # Package initialization
│   ├── dataset.py             # MUSDB18HQ dataset loader
│   └── transforms.py          # Audio augmentation transforms
├── training/                   # Training utilities
│   ├── __init__.py            # Package initialization
│   ├── train.py               # Training loop and Trainer class
│   ├── loss.py                # Loss functions (time + STFT)
│   └── metrics.py             # Evaluation metrics (cSDR, uSDR)
├── inference/                  # Inference utilities
│   ├── __init__.py            # Package initialization
│   └── separate.py            # Vocal separation inference script
├── utils/                      # Utility functions
│   ├── __init__.py            # Package initialization
│   ├── audio_utils.py         # STFT/ISTFT operations
│   └── config.py              # Configuration management
├── configs/                    # Configuration files
│   └── bsmamba2.yaml          # Default hyperparameters
├── examples/                   # Example scripts
│   ├── quick_start.py         # Quick start example
│   ├── train_example.py       # Training example
│   └── evaluate.py            # Evaluation script
├── __init__.py                # Main package initialization
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup script
├── README.md                  # Project documentation
├── LICENSE                    # MIT License
├── .gitignore                 # Git ignore rules
├── CONTRIBUTING.md            # Contribution guidelines
└── CHANGELOG.md               # Version history
```

## 📦 Core Components

### 1. Models (`models/`)

#### `bsmamba2.py` - Main Model
- **BSMamba2**: Complete model with STFT/ISTFT integration
- **separate_track()**: Full track separation with segmentation

#### `mamba2.py` - State Space Model
- **Mamba2Block**: Bidirectional SSM with selective updates
- **RMSNorm**: Root Mean Square normalization
- Scalar state transition matrix A = aI

#### `components.py` - Architecture Components
- **BandSplitModule**: Splits freq axis into 62 sub-bands
- **DualPathModule**: Time-axis and band-axis Mamba2 processing
- **MaskEstimationModule**: Generates time-frequency masks

### 2. Data (`data/`)

#### `dataset.py` - Data Loading
- **MUSDB18Dataset**: MUSDB18HQ loader with 8-second segments
- **InferenceDataset**: Single file inference loader
- Random source mixing for augmentation

#### `transforms.py` - Augmentation
- **RandomGain**: Random volume adjustment
- **RandomFlip**: Channel flipping
- **RandomPhaseShift**: Temporal phase shift
- **Normalize**: RMS normalization

### 3. Training (`training/`)

#### `train.py` - Training Loop
- **Trainer**: Complete training manager
- Mixed precision (bfloat16) support
- Gradient accumulation
- TensorBoard logging
- Checkpoint management

#### `loss.py` - Loss Functions
- **bsmamba2_loss()**: Combined time + multi-resolution STFT loss
- **stft_l1_loss()**: Single-resolution STFT loss
- Multi-resolution windows: [4096, 2048, 1024, 512, 256]

#### `metrics.py` - Evaluation
- **compute_cSDR()**: Chunk-level SDR (1-second chunks)
- **compute_uSDR()**: Utterance-level SDR
- **MetricsCalculator**: Batch metric computation

### 4. Inference (`inference/`)

#### `separate.py` - Vocal Separation
- Single file and batch processing
- 8-second non-overlapping segments
- Sequential concatenation
- Command-line interface

### 5. Utils (`utils/`)

#### `audio_utils.py` - STFT Operations
- **stft()**: Short-Time Fourier Transform
- **istft()**: Inverse STFT
- Magnitude and phase computation
- Audio normalization and trimming

#### `config.py` - Configuration
- **load_config()**: Load YAML configuration
- **get_default_config()**: Default settings
- Configuration validation

### 6. Examples (`examples/`)

#### `quick_start.py`
- Simple vocal separation example
- Minimal code for quick testing

#### `train_example.py`
- Complete training example
- Configuration customization

#### `evaluate.py`
- MUSDB18HQ test set evaluation
- Per-track and average metrics

## 🔑 Key Features

### Model Architecture
- **Parameters**: ~48.1M
- **Hidden dimension**: 256
- **Sub-bands**: 62
- **Dual-path layers**: 6
- **State dimension**: 64

### Audio Processing
- **Sample rate**: 44,100 Hz
- **FFT size**: 2048
- **Hop length**: 441
- **Segment length**: 8 seconds

### Training
- **Optimizer**: AdamW
- **Learning rate**: 5e-4
- **Batch size**: 5 per GPU
- **Gradient accumulation**: 6 steps
- **Precision**: bfloat16

### Performance
- **cSDR**: 11.03 dB
- **uSDR**: 10.70 dB
- State-of-the-art on MUSDB18HQ

## 🚀 Quick Start Commands

### Training
```bash
python -m training.train --config configs/bsmamba2.yaml
```

### Inference
```bash
python -m inference.separate \
    --model outputs/best_model.pt \
    --input audio.wav \
    --output vocals.wav
```

### Evaluation
```bash
python examples/evaluate.py \
    --model outputs/best_model.pt \
    --musdb-root /path/to/musdb18hq
```

## 📊 Performance Benchmarks

| Metric | BSMamba2 | BS-RoFormer | HT Demucs |
|--------|----------|-------------|-----------|
| cSDR   | 11.03    | 10.90       | 10.58     |
| uSDR   | 10.70    | 10.47       | 10.23     |

## 🔧 Dependencies

### Core Dependencies
- PyTorch >= 2.0.0
- torchaudio >= 2.0.0
- librosa >= 0.10.0
- einops >= 0.6.0

### Data & Evaluation
- musdb >= 0.4.0
- museval >= 0.4.0
- soundfile >= 0.10.0

### Training & Logging
- tensorboard >= 2.8.0
- pyyaml >= 6.0
- tqdm >= 4.62.0

## 📖 Documentation

- **README.md**: Main documentation
- **CONTRIBUTING.md**: Contribution guidelines
- **CHANGELOG.md**: Version history
- **LICENSE**: MIT License

## 🎯 Next Steps

1. Download MUSDB18HQ dataset
2. Update `configs/bsmamba2.yaml` with dataset path
3. Install dependencies: `pip install -r requirements.txt`
4. Start training: `python -m training.train --config configs/bsmamba2.yaml`

## 💡 Tips

- Use GPU with >= 16GB VRAM for training
- Enable mixed precision (bf16) for memory efficiency
- Use gradient accumulation for larger effective batch size
- Monitor training with TensorBoard
- Save checkpoints regularly

---

**Project Status**: ✅ Complete Implementation
**License**: MIT
**Python Version**: >= 3.8
