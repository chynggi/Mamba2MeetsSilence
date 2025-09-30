#!/usr/bin/env python3
"""
Wrapper script to run examples with proper PYTHONPATH setup.
Usage: python run.py examples/train_example.py
"""

import sys
import os
from pathlib import Path

# Get the directory of this script (project root)
project_root = Path(__file__).resolve().parent

# Add project root to Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Also set PYTHONPATH environment variable
os.environ['PYTHONPATH'] = str(project_root) + os.pathsep + os.environ.get('PYTHONPATH', '')

print(f"Project root: {project_root}")
print(f"PYTHONPATH set to: {os.environ['PYTHONPATH']}")
print()

# Get the script to run from command line arguments
if len(sys.argv) < 2:
    print("Usage: python run.py <script_to_run>")
    print("Example: python run.py examples/train_example.py")
    sys.exit(1)

script_path = sys.argv[1]
script_args = sys.argv[2:]

# Check if script exists
if not Path(script_path).exists():
    print(f"Error: Script not found: {script_path}")
    sys.exit(1)

# Read and execute the script
print(f"Running: {script_path}")
print("=" * 60)
print()

with open(script_path, 'r') as f:
    script_content = f.read()

# Set up the execution environment
exec_globals = {
    '__name__': '__main__',
    '__file__': str(Path(script_path).resolve()),
}

# Replace sys.argv for the script
original_argv = sys.argv
sys.argv = [script_path] + script_args

try:
    exec(script_content, exec_globals)
finally:
    # Restore original sys.argv
    sys.argv = original_argv
