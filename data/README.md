# Data Package

This directory contains the data loading and preprocessing modules for BSMamba2.

## Modules

- `dataset.py` - MUSDB18HQ dataset loader
- `transforms.py` - Audio augmentation and preprocessing transforms

## Important Note

⚠️ **This is source code, not a dataset directory!**

This `data/` folder contains Python modules for data loading. 
It does NOT contain actual audio dataset files.

## Dataset Location

Place your MUSDB18HQ dataset in a separate directory (e.g., `/path/to/musdb18hq/` or `~/datasets/musdb18hq/`) and configure the path in your training script or config file.

Example:
```python
config['data']['root'] = '/path/to/musdb18hq'
```

Do not put large dataset files in this `data/` source code directory.
