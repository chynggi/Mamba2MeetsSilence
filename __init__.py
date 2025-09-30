"""BSMamba2: Music Source Separation with Mamba2.

This is the main package for BSMamba2, a state-of-the-art music source separation
model based on the Mamba2 State Space Model architecture.
"""

__version__ = '0.1.0'
__author__ = 'BSMamba2 Contributors'

from models.bsmamba2 import BSMamba2
from utils.config import load_config, get_default_config

__all__ = [
    'BSMamba2',
    'load_config',
    'get_default_config',
]
