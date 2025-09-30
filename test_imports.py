#!/usr/bin/env python
"""Test script to verify import paths are working correctly."""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Also add as absolute path to avoid any relative path issues
if str(project_root.absolute()) not in sys.path:
    sys.path.insert(0, str(project_root.absolute()))

# Set PYTHONPATH environment variable as well
os.environ['PYTHONPATH'] = str(project_root) + os.pathsep + os.environ.get('PYTHONPATH', '')

print("=" * 60)
print("Testing BSMamba2 Import Paths")
print("=" * 60)
print(f"\nProject root: {project_root}")
print(f"\nPython executable: {sys.executable}")
print(f"\nFirst 5 entries in sys.path:")
for i, path in enumerate(sys.path[:5]):
    print(f"  {i}: {path}")

print("\n" + "=" * 60)
print("Testing imports...")
print("=" * 60)

try:
    print("\n1. Testing utils.config...")
    from utils.config import load_config
    print("   ✓ utils.config imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import utils.config: {e}")

try:
    print("\n2. Testing models.mamba2...")
    from models.mamba2 import Mamba2Block
    print("   ✓ models.mamba2 imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import models.mamba2: {e}")

try:
    print("\n3. Testing models.components...")
    from models.components import BandSplitModule
    print("   ✓ models.components imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import models.components: {e}")

try:
    print("\n4. Testing models.bsmamba2...")
    from models.bsmamba2 import BSMamba2
    print("   ✓ models.bsmamba2 imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import models.bsmamba2: {e}")

try:
    print("\n5. Testing data.dataset...")
    from data.dataset import MUSDB18Dataset
    print("   ✓ data.dataset imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import data.dataset: {e}")

try:
    print("\n6. Testing training.loss...")
    from training.loss import bsmamba2_loss
    print("   ✓ training.loss imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import training.loss: {e}")

try:
    print("\n7. Testing training.metrics...")
    from training.metrics import compute_cSDR
    print("   ✓ training.metrics imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import training.metrics: {e}")

try:
    print("\n8. Testing training.train...")
    from training.train import train_model
    print("   ✓ training.train imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import training.train: {e}")

print("\n" + "=" * 60)
print("Import test completed!")
print("=" * 60)
