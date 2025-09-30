"""Inference script for BSMamba2 vocal separation.

This module provides utilities for separating vocals from music using BSMamba2.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm

from ..models.bsmamba2 import BSMamba2
from ..utils.config import load_config


logger = logging.getLogger(__name__)


def separate_vocals(
    model: BSMamba2,
    input_path: str,
    output_path: str,
    device: torch.device,
    segment_length: int = 8,
    sample_rate: int = 44100,
) -> None:
    """Separate vocals from an audio file.
    
    Args:
        model: BSMamba2 model
        input_path: Path to input audio file
        output_path: Path to save separated vocals
        device: Device to run inference on
        segment_length: Segment length in seconds
        sample_rate: Sample rate
    """
    logger.info(f'Separating vocals from {input_path}')
    
    # Load audio
    audio, sr = sf.read(input_path, always_2d=True)
    
    # Resample if needed
    if sr != sample_rate:
        from librosa import resample
        audio = resample(audio.T, orig_sr=sr, target_sr=sample_rate).T
    
    # Convert to (channels, samples)
    audio = audio.T
    
    # Ensure stereo
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    
    # Convert to tensor
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # (1, channels, samples)
    audio_tensor = audio_tensor.to(device)
    
    # Separate vocals
    model.eval()
    with torch.no_grad():
        separated = model.separate_track(
            audio_tensor,
            sr=sample_rate,
            segment_length=segment_length,
        )
    
    # Convert back to numpy
    separated = separated.squeeze(0).cpu().numpy()  # (channels, samples)
    separated = separated.T  # (samples, channels)
    
    # Save output
    sf.write(output_path, separated, sample_rate)
    logger.info(f'Saved separated vocals to {output_path}')


def batch_separate(
    model_path: str,
    input_dir: str,
    output_dir: str,
    config: dict,
    device: Optional[torch.device] = None,
) -> None:
    """Separate vocals from all audio files in a directory.
    
    Args:
        model_path: Path to trained model checkpoint
        input_dir: Directory containing input audio files
        output_dir: Directory to save separated vocals
        config: Configuration dictionary
        device: Device to run inference on
    """
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load model
    logger.info(f'Loading model from {model_path}')
    model = BSMamba2(
        n_fft=config['audio']['n_fft'],
        hop_length=config['audio']['hop_length'],
        num_subbands=config['model']['num_subbands'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files
    input_dir = Path(input_dir)
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_dir.glob(f'*{ext}'))
    
    logger.info(f'Found {len(audio_files)} audio files')
    
    # Process each file
    for audio_file in tqdm(audio_files, desc='Separating vocals'):
        output_file = output_dir / f'{audio_file.stem}_vocals{audio_file.suffix}'
        
        try:
            separate_vocals(
                model=model,
                input_path=str(audio_file),
                output_path=str(output_file),
                device=device,
                segment_length=config['audio']['segment_length'],
                sample_rate=config['audio']['sample_rate'],
            )
        except Exception as e:
            logger.error(f'Error processing {audio_file}: {e}')


def main():
    """Main entry point for inference script."""
    parser = argparse.ArgumentParser(description='Separate vocals using BSMamba2')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/bsmamba2.yaml', help='Path to config file')
    parser.add_argument('--input', type=str, required=True, help='Input audio file or directory')
    parser.add_argument('--output', type=str, required=True, help='Output file or directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    # Load config
    config = load_config(args.config)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Check if input is file or directory
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file inference
        logger.info('Single file mode')
        
        # Load model
        model = BSMamba2(
            n_fft=config['audio']['n_fft'],
            hop_length=config['audio']['hop_length'],
            num_subbands=config['model']['num_subbands'],
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
        )
        
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Separate
        separate_vocals(
            model=model,
            input_path=args.input,
            output_path=args.output,
            device=device,
            segment_length=config['audio']['segment_length'],
            sample_rate=config['audio']['sample_rate'],
        )
    
    elif input_path.is_dir():
        # Batch inference
        logger.info('Batch mode')
        batch_separate(
            model_path=args.model,
            input_dir=args.input,
            output_dir=args.output,
            config=config,
            device=device,
        )
    
    else:
        logger.error(f'Input path {input_path} does not exist')
        return
    
    logger.info('Inference completed!')


if __name__ == '__main__':
    main()
