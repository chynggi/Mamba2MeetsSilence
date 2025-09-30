"""Evaluation script for BSMamba2 on MUSDB18HQ test set.

This script evaluates a trained BSMamba2 model on the MUSDB18HQ test set
and computes cSDR and uSDR metrics.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import logging
import torch
import musdb
import numpy as np
from tqdm import tqdm

from models.bsmamba2 import BSMamba2
from utils.config import load_config
from training.metrics import compute_cSDR, compute_uSDR


logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str,
    config_path: str,
    musdb_root: str,
    output_dir: str = 'evaluation_results',
):
    """Evaluate BSMamba2 on MUSDB18HQ test set.
    
    Args:
        model_path: Path to model checkpoint
        config_path: Path to configuration file
        musdb_root: Root directory of MUSDB18HQ dataset
        output_dir: Directory to save evaluation results
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    # Load config
    config = load_config(config_path)
    
    # Setup device
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
    
    logger.info(f'Model loaded with {model.get_num_parameters():,} parameters')
    
    # Load MUSDB18 test set
    logger.info(f'Loading MUSDB18HQ test set from {musdb_root}')
    mus = musdb.DB(root=musdb_root, subsets='test', is_wav=True)
    
    # Evaluate each track
    results = {
        'track_name': [],
        'cSDR': [],
        'uSDR': [],
    }
    
    for track in tqdm(mus.tracks, desc='Evaluating tracks'):
        logger.info(f'Processing track: {track.name}')
        
        # Load mixture
        mixture = track.audio.T  # (channels, samples)
        mixture_tensor = torch.from_numpy(mixture).float().unsqueeze(0).to(device)
        
        # Load target vocals
        target = track.sources['vocals'].audio.T  # (channels, samples)
        
        # Separate vocals
        with torch.no_grad():
            separated = model.separate_track(
                mixture_tensor,
                sr=config['audio']['sample_rate'],
                segment_length=config['audio']['segment_length'],
            )
        
        separated = separated.squeeze(0).cpu()  # (channels, samples)
        target_tensor = torch.from_numpy(target).float()
        
        # Compute metrics
        csdr = compute_cSDR(separated, target_tensor, config['audio']['sample_rate'])
        usdr = compute_uSDR(separated, target_tensor)
        
        logger.info(f'  cSDR: {csdr:.2f} dB, uSDR: {usdr:.2f} dB')
        
        # Store results
        results['track_name'].append(track.name)
        results['cSDR'].append(csdr)
        results['uSDR'].append(usdr)
    
    # Compute average metrics
    avg_csdr = np.mean(results['cSDR'])
    avg_usdr = np.mean(results['uSDR'])
    
    logger.info(f'\nAverage Results:')
    logger.info(f'  cSDR: {avg_csdr:.2f} dB')
    logger.info(f'  uSDR: {avg_usdr:.2f} dB')
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / 'evaluation_results.txt'
    with open(results_file, 'w') as f:
        f.write('BSMamba2 Evaluation Results\n')
        f.write('=' * 50 + '\n\n')
        f.write(f'Average cSDR: {avg_csdr:.2f} dB\n')
        f.write(f'Average uSDR: {avg_usdr:.2f} dB\n\n')
        f.write('Per-track results:\n')
        f.write('-' * 50 + '\n')
        
        for i, track_name in enumerate(results['track_name']):
            f.write(f'{track_name}: cSDR={results["cSDR"][i]:.2f} dB, uSDR={results["uSDR"][i]:.2f} dB\n')
    
    logger.info(f'Results saved to {results_file}')


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate BSMamba2 on MUSDB18HQ')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/bsmamba2.yaml', help='Path to config file')
    parser.add_argument('--musdb-root', type=str, required=True, help='MUSDB18HQ root directory')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Output directory')
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        config_path=args.config,
        musdb_root=args.musdb_root,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
