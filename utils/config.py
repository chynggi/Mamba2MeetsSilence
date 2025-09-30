"""Configuration management utilities.

This module provides utilities for loading and managing configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for BSMamba2.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'model': {
            'hidden_dim': 256,
            'num_layers': 6,
            'num_subbands': 62,
            'd_state': 64,
            'd_conv': 4,
        },
        'audio': {
            'sample_rate': 44100,
            'n_fft': 2048,
            'hop_length': 441,
            'segment_length': 8,
        },
        'training': {
            'batch_size': 5,
            'gradient_accumulation_steps': 6,
            'learning_rate': 5e-4,
            'num_epochs': 100,
            'precision': 'bf16',
            'lambda_time': 10,
            'dropout': 0.0,
            'output_dir': 'outputs',
            'resume_from': None,
        },
        'loss': {
            'stft_windows': [4096, 2048, 1024, 512, 256],
            'stft_hop': 147,
        },
        'data': {
            'root': '/path/to/musdb18hq',
        },
    }


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f'Configuration file not found: {config_path}')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Merge with defaults
    default_config = get_default_config()
    config = merge_configs(default_config, config)
    
    return config


def merge_configs(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries.
    
    Args:
        default: Default configuration
        override: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = default.copy()
    
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration dictionary.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_keys = ['model', 'audio', 'training', 'loss', 'data']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f'Missing required configuration key: {key}')
    
    # Validate model config
    model_keys = ['hidden_dim', 'num_layers', 'num_subbands']
    for key in model_keys:
        if key not in config['model']:
            raise ValueError(f'Missing required model configuration key: {key}')
    
    # Validate audio config
    audio_keys = ['sample_rate', 'n_fft', 'hop_length', 'segment_length']
    for key in audio_keys:
        if key not in config['audio']:
            raise ValueError(f'Missing required audio configuration key: {key}')
    
    # Validate training config
    training_keys = ['batch_size', 'learning_rate', 'num_epochs']
    for key in training_keys:
        if key not in config['training']:
            raise ValueError(f'Missing required training configuration key: {key}')
    
    return True
