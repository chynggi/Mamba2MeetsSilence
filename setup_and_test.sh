#!/bin/bash
# Quick setup and run script for Linux/Mac

set -e  # Exit on error

echo "BSMamba2 Quick Setup Script"
echo "============================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Project directory: $SCRIPT_DIR"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
echo ""

# Set PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
echo "PYTHONPATH set to: $SCRIPT_DIR"
echo ""

# Test imports
echo "Testing imports..."
python3 test_imports.py

echo ""
echo "============================"
echo "Setup complete!"
echo ""
echo "You can now run examples:"
echo "  python3 examples/train_example.py"
echo "  python3 examples/quick_start.py"
echo "  python3 examples/evaluate.py"
echo ""
echo "Or use the run.py wrapper:"
echo "  python3 run.py examples/train_example.py"
echo ""
