"""Evaluation metrics for BSMamba2.

This module implements cSDR and uSDR metrics for evaluating source separation.
"""

from typing import Optional
import torch
import numpy as np
import museval


def sdr(
    pred: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute Signal-to-Distortion Ratio (SDR).
    
    Args:
        pred: Predicted audio of shape (batch, channels, samples)
        target: Target audio of same shape
        epsilon: Small constant for numerical stability
        
    Returns:
        SDR value in dB
    """
    # Flatten batch and channels
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    
    # Compute SDR
    target_power = torch.sum(target ** 2)
    noise_power = torch.sum((pred - target) ** 2)
    
    sdr_value = 10 * torch.log10(target_power / (noise_power + epsilon) + epsilon)
    
    return sdr_value


def compute_cSDR(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int = 44100,
    chunk_length: float = 1.0,
) -> float:
    """Compute chunk-level SDR (cSDR).
    
    Computes median SDR over 1-second chunks.
    
    Args:
        pred: Predicted audio of shape (channels, samples)
        target: Target audio of same shape
        sample_rate: Sample rate (default: 44100)
        chunk_length: Chunk length in seconds (default: 1.0)
        
    Returns:
        cSDR value in dB
    """
    chunk_samples = int(chunk_length * sample_rate)
    num_chunks = pred.shape[-1] // chunk_samples
    
    sdrs = []
    for i in range(num_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        
        pred_chunk = pred[..., start:end]
        target_chunk = target[..., start:end]
        
        # Compute SDR for this chunk
        chunk_sdr = sdr(pred_chunk, target_chunk)
        sdrs.append(chunk_sdr.item())
    
    # Return median SDR
    if len(sdrs) == 0:
        return 0.0
    
    return float(np.median(sdrs))


def compute_uSDR(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Compute utterance-level SDR (uSDR).
    
    Computes SDR over the entire track.
    
    Args:
        pred: Predicted audio of shape (channels, samples)
        target: Target audio of same shape
        
    Returns:
        uSDR value in dB
    """
    return sdr(pred, target).item()


def compute_museval_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    sample_rate: int = 44100,
) -> dict:
    """Compute metrics using museval library.
    
    Args:
        pred: Predicted audio of shape (samples, channels)
        target: Target audio of same shape
        sample_rate: Sample rate (default: 44100)
        
    Returns:
        Dictionary of metric values
    """
    # Ensure correct shape (samples, channels)
    if pred.ndim == 1:
        pred = pred[:, np.newaxis]
    if target.ndim == 1:
        target = target[:, np.newaxis]
    
    # Compute metrics
    sdr, isr, sir, sar = museval.evaluate(
        target,
        pred,
        win=sample_rate,
        hop=sample_rate,
    )
    
    return {
        'SDR': float(np.nanmedian(sdr)),
        'ISR': float(np.nanmedian(isr)),
        'SIR': float(np.nanmedian(sir)),
        'SAR': float(np.nanmedian(sar)),
    }


class MetricsCalculator:
    """Calculator for evaluation metrics.
    
    Args:
        sample_rate: Sample rate (default: 44100)
        use_museval: Whether to use museval library (default: False)
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        use_museval: bool = False,
    ):
        self.sample_rate = sample_rate
        self.use_museval = use_museval
        
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.csdrs = []
        self.usdrs = []
        self.museval_metrics = []
    
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ):
        """Update metrics with new batch.
        
        Args:
            pred: Predicted audio
            target: Target audio
        """
        # Move to CPU
        pred = pred.detach().cpu()
        target = target.detach().cpu()
        
        batch_size = pred.shape[0]
        
        for i in range(batch_size):
            pred_i = pred[i]  # (channels, samples)
            target_i = target[i]
            
            # Compute cSDR
            csdr = compute_cSDR(pred_i, target_i, self.sample_rate)
            self.csdrs.append(csdr)
            
            # Compute uSDR
            usdr = compute_uSDR(pred_i, target_i)
            self.usdrs.append(usdr)
            
            # Compute museval metrics if requested
            if self.use_museval:
                pred_np = pred_i.numpy().T  # (samples, channels)
                target_np = target_i.numpy().T
                
                metrics = compute_museval_metrics(pred_np, target_np, self.sample_rate)
                self.museval_metrics.append(metrics)
    
    def compute(self) -> dict:
        """Compute average metrics.
        
        Returns:
            Dictionary of metric values
        """
        results = {
            'cSDR': float(np.mean(self.csdrs)) if self.csdrs else 0.0,
            'uSDR': float(np.mean(self.usdrs)) if self.usdrs else 0.0,
        }
        
        if self.use_museval and self.museval_metrics:
            # Average museval metrics
            for key in ['SDR', 'ISR', 'SIR', 'SAR']:
                values = [m[key] for m in self.museval_metrics]
                results[key] = float(np.mean(values))
        
        return results
