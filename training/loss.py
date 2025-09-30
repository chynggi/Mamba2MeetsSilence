"""Loss functions for BSMamba2 training.

This module implements the combined time-domain and multi-resolution STFT loss.
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


# Global window cache to avoid recreating windows every time
_window_cache = {}

def _get_cached_window(n_fft: int, device: torch.device) -> torch.Tensor:
    """Get or create cached Hann window."""
    key = (n_fft, device)
    if key not in _window_cache:
        _window_cache[key] = torch.hann_window(n_fft, device=device)
    return _window_cache[key]

def stft_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_fft: int,
    hop_length: int,
    window: str = 'hann',
) -> torch.Tensor:
    """Compute L1 loss in STFT domain (optimized).
    
    Args:
        pred: Predicted audio of shape (batch, channels, samples)
        target: Target audio of same shape
        n_fft: FFT size
        hop_length: Hop length
        window: Window type (default: 'hann')
        
    Returns:
        STFT L1 loss value
    """
    # Flatten batch and channels for efficient STFT
    batch, channels, samples = pred.shape
    pred_flat = pred.reshape(batch * channels, samples)
    target_flat = target.reshape(batch * channels, samples)
    
    # Get cached window
    window_tensor = _get_cached_window(n_fft, pred.device)
    
    # Compute STFT (use center=False to reduce computation)
    pred_stft = torch.stft(
        pred_flat,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window_tensor,
        center=False,  # Faster without padding
        return_complex=True,
    )
    target_stft = torch.stft(
        target_flat,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window_tensor,
        center=False,
        return_complex=True,
    )
    
    # Only use magnitude loss (skip phase loss for speed)
    pred_mag = torch.abs(pred_stft)
    target_mag = torch.abs(target_stft)
    mag_loss = F.l1_loss(pred_mag, target_mag)
    
    return mag_loss


def multi_resolution_stft_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    fft_sizes: List[int] = [2048, 1024, 512],  # Reduced from 5 to 3 resolutions
    hop_length: int = 147,
) -> torch.Tensor:
    """Compute multi-resolution STFT loss (optimized).
    
    Args:
        pred: Predicted audio of shape (batch, channels, samples)
        target: Target audio of same shape
        fft_sizes: List of FFT sizes for multi-resolution
        hop_length: Hop length (default: 147)
        
    Returns:
        Multi-resolution STFT loss value
    """
    loss = 0.0
    
    # Use only 3 resolutions instead of 5 for faster computation
    for n_fft in fft_sizes:
        loss += stft_l1_loss(pred, target, n_fft, hop_length)
    
    # Average over resolutions
    loss = loss / len(fft_sizes)
    
    return loss


def bsmamba2_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_time: float = 10.0,
    fft_sizes: List[int] = [4096, 2048, 1024, 512, 256],
    stft_hop: int = 147,
) -> torch.Tensor:
    """BSMamba2 combined loss function.
    
    Combines L1 time-domain loss and multi-resolution STFT loss:
    Loss = λ_time * L1_time + L_STFT
    
    Args:
        pred: Predicted audio of shape (batch, channels, samples)
        target: Target audio of same shape
        lambda_time: Weight for time-domain loss (default: 10.0)
        fft_sizes: List of FFT sizes (default: [4096, 2048, 1024, 512, 256])
        stft_hop: Hop length for STFT (default: 147)
        
    Returns:
        Combined loss value
    """
    # L1 time-domain loss
    time_loss = F.l1_loss(pred, target)
    
    # Multi-resolution STFT loss
    stft_loss = multi_resolution_stft_loss(pred, target, fft_sizes, stft_hop)
    
    # Combined loss
    total_loss = lambda_time * time_loss + stft_loss
    
    return total_loss


class BSMamba2Loss(nn.Module):
    """BSMamba2 loss module (optimized).
    
    Args:
        lambda_time: Weight for time-domain loss (default: 10.0)
        fft_sizes: List of FFT sizes (default: [2048, 1024, 512])
        stft_hop: Hop length for STFT (default: 147)
    """
    
    def __init__(
        self,
        lambda_time: float = 10.0,
        fft_sizes: List[int] = [2048, 1024, 512],  # Reduced for speed
        stft_hop: int = 147,
    ):
        super().__init__()
        self.lambda_time = lambda_time
        self.fft_sizes = fft_sizes
        self.stft_hop = stft_hop
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss.
        
        Args:
            pred: Predicted audio
            target: Target audio
            
        Returns:
            Loss value
        """
        return bsmamba2_loss(
            pred,
            target,
            lambda_time=self.lambda_time,
            fft_sizes=self.fft_sizes,
            stft_hop=self.stft_hop,
        )
