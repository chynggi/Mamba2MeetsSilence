"""Quick start example for BSMamba2.

This script demonstrates basic usage of BSMamba2 for vocal separation.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import soundfile as sf
from models.bsmamba2 import BSMamba2
from utils.config import load_config


def quick_separate(
    model_path: str,
    audio_path: str,
    output_path: str,
    config_path: str = 'configs/bsmamba2.yaml',
):
    """Quick vocal separation example.
    
    Args:
        model_path: Path to trained model checkpoint
        audio_path: Path to input audio file
        output_path: Path to save separated vocals
        config_path: Path to configuration file
    """
    # Load config
    config = load_config(config_path)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model
    model = BSMamba2(
        n_fft=config['audio']['n_fft'],
        hop_length=config['audio']['hop_length'],
        num_subbands=config['model']['num_subbands'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f'Model loaded with {model.get_num_parameters():,} parameters')
    
    # Load audio
    audio, sr = sf.read(audio_path, always_2d=True)
    print(f'Loaded audio: {audio.shape}, sr={sr}')
    
    # Resample if needed
    if sr != config['audio']['sample_rate']:
        from librosa import resample
        audio = resample(audio.T, orig_sr=sr, target_sr=config['audio']['sample_rate']).T
        sr = config['audio']['sample_rate']
    
    # Convert to tensor
    audio = torch.from_numpy(audio.T).float().unsqueeze(0).to(device)  # (1, channels, samples)
    
    # Separate vocals
    print('Separating vocals...')
    with torch.no_grad():
        separated = model.separate_track(
            audio,
            sr=sr,
            segment_length=config['audio']['segment_length'],
        )
    
    # Convert back to numpy
    separated = separated.squeeze(0).cpu().numpy().T  # (samples, channels)
    
    # Save output
    sf.write(output_path, separated, sr)
    print(f'Saved separated vocals to {output_path}')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick vocal separation example')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input audio file')
    parser.add_argument('--output', type=str, required=True, help='Output audio file')
    parser.add_argument('--config', type=str, default='configs/bsmamba2.yaml', help='Config file')
    
    args = parser.parse_args()
    
    quick_separate(
        model_path=args.model,
        audio_path=args.input,
        output_path=args.output,
        config_path=args.config,
    )
