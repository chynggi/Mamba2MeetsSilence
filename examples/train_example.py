"""Training example script for BSMamba2.

This script shows how to train BSMamba2 from scratch or resume training.
"""

import torch
from training.train import train_model
from utils.config import load_config


def main():
    """Main training example."""
    # Load configuration
    config = load_config('configs/bsmamba2.yaml')
    
    # Override some settings for this example
    config['training']['num_epochs'] = 100
    config['training']['batch_size'] = 5
    config['training']['output_dir'] = 'outputs/example_run'
    
    # Update dataset path (IMPORTANT: Change this to your dataset location)
    config['data']['root'] = '/path/to/your/musdb18hq'
    
    # Print configuration
    print('Training Configuration:')
    print(f"  Model: {config['model']['num_layers']} layers, {config['model']['hidden_dim']} hidden dim")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Epochs: {config['training']['num_epochs']}")
    print(f"  Precision: {config['training']['precision']}")
    print(f"  Output dir: {config['training']['output_dir']}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nUsing device: {device}')
    
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    
    # Start training
    print('\nStarting training...')
    train_model(config, device)


if __name__ == '__main__':
    main()
