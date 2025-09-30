"""
Import helper for BSMamba2 project.

This module ensures that all project modules can be imported correctly
by setting up the Python path and handling potential naming conflicts.

Usage:
    import setup_imports  # Put this at the top of your script
    
Or:
    from setup_imports import ensure_imports
    ensure_imports()
"""

import sys
import os
from pathlib import Path


def ensure_imports():
    """Ensure that BSMamba2 modules can be imported.
    
    This function:
    1. Adds the project root to sys.path
    2. Sets PYTHONPATH environment variable
    3. Verifies that critical directories exist
    """
    # Get project root (directory containing this file)
    project_root = Path(__file__).resolve().parent
    
    # Add project root to sys.path if not already there
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    # Also try absolute path
    project_root_abs = str(project_root.absolute())
    if project_root_abs not in sys.path and project_root_abs != project_root_str:
        sys.path.insert(0, project_root_abs)
    
    # Set PYTHONPATH environment variable
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if project_root_str not in current_pythonpath:
        os.environ['PYTHONPATH'] = project_root_str + os.pathsep + current_pythonpath
    
    # Verify critical directories exist
    critical_dirs = ['data', 'models', 'training', 'utils', 'inference']
    for dir_name in critical_dirs:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            print(f"Warning: {dir_name} directory not found at {dir_path}", file=sys.stderr)
    
    return project_root


# Automatically run when imported
_project_root = ensure_imports()


# Export commonly used imports
__all__ = ['ensure_imports']


if __name__ == '__main__':
    print("=" * 60)
    print("BSMamba2 Import Setup")
    print("=" * 60)
    print(f"\nProject root: {_project_root}")
    print(f"\nPython executable: {sys.executable}")
    print(f"\nPYTHONPATH: {os.environ.get('PYTHONPATH', '(not set)')}")
    print(f"\nFirst 5 entries in sys.path:")
    for i, path in enumerate(sys.path[:5]):
        print(f"  {i}: {path}")
    print("\n" + "=" * 60)
    print("Testing imports...")
    print("=" * 60)
    
    modules_to_test = [
        ('utils.config', 'load_config'),
        ('models.mamba2', 'Mamba2Block'),
        ('models.components', 'BandSplitModule'),
        ('models.bsmamba2', 'BSMamba2'),
        ('data.dataset', 'MUSDB18Dataset'),
        ('training.loss', 'bsmamba2_loss'),
        ('training.metrics', 'compute_cSDR'),
    ]
    
    success_count = 0
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✓ {module_name}.{class_name}")
            success_count += 1
        except Exception as e:
            print(f"✗ {module_name}.{class_name}: {e}")
    
    print("\n" + "=" * 60)
    print(f"Results: {success_count}/{len(modules_to_test)} imports successful")
    print("=" * 60)
