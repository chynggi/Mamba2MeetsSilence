#!/usr/bin/env python
"""Debug script to investigate the 'data' module import issue."""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("Debugging 'data' module import issue")
print("=" * 60)

print(f"\nProject root: {project_root}")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Check if data directory exists
data_dir = project_root / 'data'
print(f"\nData directory: {data_dir}")
print(f"Data directory exists: {data_dir.exists()}")
print(f"Data directory is a directory: {data_dir.is_dir()}")

if data_dir.exists():
    print("\nContents of data directory:")
    for item in sorted(data_dir.iterdir()):
        print(f"  - {item.name}")

# Check __init__.py
init_file = data_dir / '__init__.py'
print(f"\n__init__.py exists: {init_file.exists()}")
if init_file.exists():
    print(f"__init__.py size: {init_file.stat().st_size} bytes")
    print(f"__init__.py readable: {os.access(init_file, os.R_OK)}")

# Try to import step by step
print("\n" + "=" * 60)
print("Step-by-step import test:")
print("=" * 60)

# Test 1: Try to import the package
print("\n1. Attempting: import data")
try:
    import data
    print(f"   ✓ Success! data module: {data}")
    print(f"   data.__file__: {data.__file__}")
    print(f"   data.__path__: {data.__path__}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    print(f"   Exception type: {type(e).__name__}")
    
    # Try to get more details
    import importlib.util
    spec = importlib.util.find_spec('data')
    print(f"   Module spec: {spec}")

# Test 2: Try to import dataset module
print("\n2. Attempting: from data import dataset")
try:
    from data import dataset
    print(f"   ✓ Success! dataset module: {dataset}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 3: Try to import class directly
print("\n3. Attempting: from data.dataset import MUSDB18Dataset")
try:
    from data.dataset import MUSDB18Dataset
    print(f"   ✓ Success! MUSDB18Dataset class: {MUSDB18Dataset}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 4: Try manual import
print("\n4. Attempting manual module loading")
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "data.dataset",
        str(data_dir / "dataset.py")
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules["data.dataset"] = module
        spec.loader.exec_module(module)
        print(f"   ✓ Manual loading succeeded!")
        print(f"   Module: {module}")
        print(f"   MUSDB18Dataset: {module.MUSDB18Dataset}")
    else:
        print(f"   ✗ Could not create module spec")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Check sys.path
print("\n" + "=" * 60)
print("sys.path inspection:")
print("=" * 60)
for i, path in enumerate(sys.path[:10]):
    print(f"{i}: {path}")
    if Path(path) == project_root:
        # Check if data directory is accessible from this path
        test_data = Path(path) / 'data'
        print(f"   -> {test_data} exists: {test_data.exists()}")

# Check for naming conflicts
print("\n" + "=" * 60)
print("Checking for naming conflicts:")
print("=" * 60)

# Check if 'data' is a built-in module or in site-packages
import importlib.util
for path in sys.path:
    potential_conflict = Path(path) / 'data'
    if potential_conflict.exists() and potential_conflict != data_dir:
        print(f"⚠ Warning: Found another 'data' at: {potential_conflict}")

print("\n" + "=" * 60)
print("Debug completed!")
print("=" * 60)
