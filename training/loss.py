"""Loss functions for BSMamba2 training.

This module implements the combined time-domain and multi-resolution STFT loss.
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


def stft_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_fft: int,
    hop_length: int,
    window: str = 'hann',
) -> torch.Tensor:
    """Compute L1 loss in STFT domain.
    
    Args:
        pred: Predicted audio of shape (batch, channels, samples)
        target: Target audio of same shape
        n_fft: FFT size
        hop_length: Hop length
        window: Window type (default: 'hann')
        
    Returns:
        STFT L1 loss value
    """
    # Flatten batch and channels
    batch, channels, samples = pred.shape
    pred_flat = pred.reshape(batch * channels, samples)
    target_flat = target.reshape(batch * channels, samples)
    
    # Compute STFT
    pred_stft = torch.stft(
        pred_flat,
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft).to(pred.device),
        return_complex=True,
    )
    target_stft = torch.stft(
        target_flat,
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft).to(target.device),
        return_complex=True,
    )
    
    # Magnitude loss
    pred_mag = torch.abs(pred_stft)
    target_mag = torch.abs(target_stft)
    mag_loss = F.l1_loss(pred_mag, target_mag)
    
    # Phase loss (using real and imaginary parts)
    real_loss = F.l1_loss(pred_stft.real, target_stft.real)
    imag_loss = F.l1_loss(pred_stft.imag, target_stft.imag)
    
    return mag_loss + (real_loss + imag_loss) * 0.5


def multi_resolution_stft_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    fft_sizes: List[int] = [4096, 2048, 1024, 512, 256],
    hop_length: int = 147,
) -> torch.Tensor:
    """Compute multi-resolution STFT loss.
    
    Args:
        pred: Predicted audio of shape (batch, channels, samples)
        target: Target audio of same shape
        fft_sizes: List of FFT sizes for multi-resolution
        hop_length: Hop length (default: 147)
        
    Returns:
        Multi-resolution STFT loss value
    """
    loss = 0.0
    
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
    """BSMamba2 loss module.
    
    Args:
        lambda_time: Weight for time-domain loss (default: 10.0)
        fft_sizes: List of FFT sizes (default: [4096, 2048, 1024, 512, 256])
        stft_hop: Hop length for STFT (default: 147)
    """
    
    def __init__(
        self,
        lambda_time: float = 10.0,
        fft_sizes: List[int] = [4096, 2048, 1024, 512, 256],
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
