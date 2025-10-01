# BSMamba2: Music Source Separation with Mamba2

BSMamba2 is a state-of-the-art music source separation model that leverages the Mamba2 State Space Model architecture combined with Band-splitting strategy to achieve superior performance in vocal separation, especially for intermittent vocals.

## 🎵 Key Features

- **State-of-the-art Performance**: Achieves 11.03 dB cSDR and 10.70 dB uSDR on MUSDB18HQ
- **Band-Splitting Architecture**: Splits frequency axis into 62 sub-bands for efficient processing
- **Mamba2 Blocks**: Bidirectional State Space Models for time-frequency modeling
- **Efficient Training**: ~48.1M parameters with mixed precision (bfloat16) support
- **Robust to Input Length**: Consistent performance across varying audio lengths (1-16 seconds)
- **🚀 Optimized Performance**: 5x faster training with optimized Mamba2 scan and STFT operations

## 📁 Project Structure

```
bsmamba2/
├── models/              # Model implementations
│   ├── bsmamba2.py     # Main BSMamba2 model
│   ├── mamba2.py       # Mamba2 block
│   └── components.py   # Band-split, Dual-path, Mask estimation
├── data/               # Data loading and preprocessing
│   ├── dataset.py      # MUSDB18HQ dataset loader
│   └── transforms.py   # Audio augmentation
├── training/           # Training utilities
│   ├── train.py        # Training script
│   ├── loss.py         # Loss functions
│   └── metrics.py      # Evaluation metrics
├── inference/          # Inference utilities
│   └── separate.py     # Vocal separation script
├── utils/              # Utility functions
│   ├── audio_utils.py  # STFT/ISTFT operations
│   └── config.py       # Configuration management
├── configs/            # Configuration files
│   └── bsmamba2.yaml   # Default configuration
├── requirements.txt    # Python dependencies
├── setup.py           # Package setup
└── README.md          # This file
```

## 🚀 Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 2.1.0 with CUDA support
- CUDA Toolkit 11.8+ (for GPU acceleration and mamba-ssm optimization)
- Visual Studio Build Tools (Windows) or GCC (Linux)

### Install from source

```bash
git clone https://github.com/chynggi/Mamba2MeetsSilence.git
cd Mamba2MeetsSilence

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### ⚡ Optional: Install mamba-ssm for 5-10x speedup

For **significantly faster training** (5-10x speedup), install the optimized mamba-ssm package:

```bash
# Install dependencies for CUDA kernels
pip install causal-conv1d>=1.1.0

# Install mamba-ssm (requires CUDA)
pip install mamba-ssm>=2.0.0
```

**Note**: mamba-ssm requires CUDA and will compile CUDA kernels during installation. See [MAMBA_SSM_INTEGRATION.md](MAMBA_SSM_INTEGRATION.md) for detailed installation instructions and troubleshooting.

**Performance comparison**:
- Without mamba-ssm: ~60 seconds/step (native PyTorch)
- With mamba-ssm: ~12 seconds/step (CUDA optimized) ⚡

> **⚠️ Important Note for New Clones:**
> 
> If you encounter `ModuleNotFoundError: No module named 'data'` after cloning, ensure you've pulled the latest changes where the `data/` source code folder is properly tracked in Git. See [BUGFIX_DATA_GITIGNORE.md](BUGFIX_DATA_GITIGNORE.md) for details.
>
> Quick fix:
> ```bash
> git pull origin main  # Get latest changes
> python test_imports.py  # Verify all imports work
> ```

## 📊 Dataset Preparation

Download the MUSDB18HQ dataset:

```bash
# Download from https://zenodo.org/record/3338373
# Extract to your preferred location
# Update the path in configs/bsmamba2.yaml
```

Dataset structure:
```
musdb18hq/
├── train/
│   ├── track1/
│   ├── track2/
│   └── ...
├── test/
│   └── ...
```

## 🏋️ Training

### Quick Start

Train with default configuration:

```bash
python -m training.train --config configs/bsmamba2.yaml
```

### Custom Training

Edit `configs/bsmamba2.yaml` to customize hyperparameters:

```yaml
model:
  hidden_dim: 256
  num_layers: 6
  num_subbands: 62

training:
  batch_size: 5
  learning_rate: 5e-4
  num_epochs: 100
  precision: "bf16"
```

### Training Options

```bash
python -m training.train \
    --config configs/bsmamba2.yaml \
    --output_dir outputs/experiment1 \
    --resume_from outputs/checkpoint.pt
```

## 🎤 Inference

### Separate vocals from a single file

```bash
python -m inference.separate \
    --model outputs/best_model.pt \
    --config configs/bsmamba2.yaml \
    --input audio/mixture.wav \
    --output audio/vocals.wav
```

### Batch processing

```bash
python -m inference.separate \
    --model outputs/best_model.pt \
    --config configs/bsmamba2.yaml \
    --input audio/input_dir/ \
    --output audio/output_dir/
```

### Python API

```python
import torch
from models.bsmamba2 import BSMamba2
from utils.config import load_config

# Load model
config = load_config('configs/bsmamba2.yaml')
model = BSMamba2(
    n_fft=config['audio']['n_fft'],
    hop_length=config['audio']['hop_length'],
    num_subbands=config['model']['num_subbands'],
    hidden_dim=config['model']['hidden_dim'],
    num_layers=config['model']['num_layers'],
)

# Load checkpoint
checkpoint = torch.load('outputs/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Separate vocals
# mixture: (batch, channels, samples)
with torch.no_grad():
    separated = model.separate_track(mixture, sr=44100, segment_length=8)
```

## 🏆 Performance

### MUSDB18HQ Test Set Results

| Metric | BSMamba2 | BS-RoFormer | HT Demucs |
|--------|----------|-------------|-----------|
| cSDR   | **11.03** | 10.90       | 10.58     |
| uSDR   | **10.70** | 10.47       | 10.23     |

### Performance by Input Length

BSMamba2 shows consistent performance across different input lengths (1-16 seconds), unlike some competing models that degrade with shorter inputs.

## 🔧 Architecture Details

### Model Configuration

- **Parameters**: ~48.1M
- **Hidden dimension**: 256
- **Number of sub-bands**: 62
- **Dual-path layers**: 6
- **State dimension**: 64

### Training Configuration

- **Optimizer**: AdamW
- **Learning rate**: 5e-4
- **Batch size**: 5 per GPU
- **Gradient accumulation**: 6 steps
- **Precision**: bfloat16
- **Loss**: L1 time-domain + Multi-resolution STFT

## 📈 Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir outputs/logs
```

## ⚡ Performance Optimization

BSMamba2 includes several optimizations for faster training:

### 🚀 Mamba-SSM Integration (Recommended)

For **5-10x training speedup**, use the optimized mamba-ssm CUDA kernels:

```bash
# Benchmark current performance
python benchmark_mamba_ssm.py

# Install mamba-ssm (if not already installed)
pip install mamba-ssm>=2.0.0 causal-conv1d>=1.1.0
```

**Performance comparison**:

| Component | Before | After | Speedup |
|-----------|--------|-------|---------|
| Mamba2 Selective Scan | ~40s | ~5s | **8x** ⭐ |
| Causal Conv1d | ~3s | ~1s | **3x** |
| Total per step | ~60s | ~12s | **5x** |

See [MAMBA_SSM_INTEGRATION.md](MAMBA_SSM_INTEGRATION.md) for detailed installation and usage guide.

### Benchmarking

Run a quick benchmark to check training speed:

```bash
# Overall training speed
python benchmark_speed.py

# Mamba-SSM specific benchmark
python benchmark_mamba_ssm.py
```

### Profiling

Profile training to identify bottlenecks:

```bash
python profile_training.py
```

The profiling results will be saved to `outputs/profiling/trace.json`. View it at `chrome://tracing`.

### Optimization Details

See [PERFORMANCE_OPTIMIZATION.md](PERFORMANCE_OPTIMIZATION.md) for detailed information about:
- Mamba2 scan optimization (5-10x speedup with mamba-ssm)
- STFT loss optimization (2-3x speedup)
- Data processing improvements
- Memory usage optimization

**Expected performance**: 
- Native PyTorch: ~60 seconds/step
- With optimizations: ~20 seconds/step
- With mamba-ssm: ~12 seconds/step ⚡

## 🛠️ Development

### Running tests

```bash
pytest tests/
```

### Code formatting

```bash
black models/ data/ training/ inference/ utils/
```

### Type checking

```bash
mypy models/ data/ training/ inference/ utils/
```

## 📝 Citation

If you use BSMamba2 in your research, please cite:

```bibtex
@article{bsmamba2,
    author = {Kim, Euiyeon and Choi, Yong-Hoon},
    year = {2025},
    month = {08},
    pages = {},
    title = {Mamba2 Meets Silence: Robust Vocal Source Separation for Sparse Regions},
    doi = {10.48550/arXiv.2508.14556}
}
```

## 📄 License

This project is licensed under the CC BY 4.0.

## 🙏 Acknowledgments

- MUSDB18HQ dataset creators
- Mamba2 architecture authors
- PyTorch team

## 🐛 Known Issues

- Requires significant GPU memory for training (recommend 16GB+ VRAM)
- MUSDB18HQ dataset requires ~40GB disk space

## 📮 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: chynggi69@gmail.com

## 🗺️ Roadmap

- [ ] Add support for multi-stem separation (drums, bass, other)
- [ ] Implement real-time inference
- [ ] Add pre-trained model weights
- [ ] Support for more datasets (DSD100, etc.)
- [ ] Web demo with Gradio
- [ ] Mobile deployment with ONNX

---

**Note**: This is a research implementation. For production use, additional optimization and testing may be required.
