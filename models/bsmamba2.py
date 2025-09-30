"""BSMamba2 main model.

This module implements the complete BSMamba2 architecture for music source
separation, combining Band-Split, Dual-Path, and Mask Estimation modules.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from .components import BandSplitModule, DualPathModule, MaskEstimationModule


class BSMamba2(nn.Module):
    """BSMamba2 model for music source separation.
    
    Architecture:
    1. Band-Split Module: Splits frequency axis into K sub-bands
    2. Dual-Path Module: Mamba2 blocks for time-frequency modeling
    3. Mask Estimation Module: Generates time-frequency masks
    
    Args:
        n_fft: FFT size (default: 2048)
        hop_length: Hop length for STFT (default: 441)
        num_subbands: Number of sub-bands K (default: 62)
        hidden_dim: Hidden dimension (default: 256)
        num_layers: Number of dual-path layers L (default: 6)
        d_state: State dimension for Mamba2 (default: 64)
        d_conv: Convolution size for Mamba2 (default: 4)
        dropout: Dropout rate (default: 0.0)
    """
    
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 441,
        num_subbands: int = 62,
        hidden_dim: int = 256,
        num_layers: int = 6,
        d_state: int = 64,
        d_conv: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_subbands = num_subbands
        self.hidden_dim = hidden_dim
        
        # Band-Split Module
        self.band_split = BandSplitModule(
            n_fft=n_fft,
            num_subbands=num_subbands,
            hidden_dim=hidden_dim,
        )
        
        # Dual-Path Module
        self.dual_path = DualPathModule(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            dropout=dropout,
        )
        
        # Mask Estimation Module
        self.mask_estimation = MaskEstimationModule(
            n_fft=n_fft,
            num_subbands=num_subbands,
            hidden_dim=hidden_dim,
        )
        
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Forward pass of BSMamba2.
        
        Args:
            spec: Complex spectrogram of shape (batch, time, freq, 2)
                  Last dimension is [real, imaginary] parts
            
        Returns:
            Separated spectrogram of same shape
        """
        # 1. Band-Split: frequency decomposition
        band_features = self.band_split(spec)  # (batch, time, K, hidden_dim)
        
        # 2. Dual-Path: time-frequency modeling
        band_features = self.dual_path(band_features)  # (batch, time, K, hidden_dim)
        
        # 3. Mask Estimation: generate and apply mask
        separated_spec = self.mask_estimation(band_features, spec)  # (batch, time, freq, 2)
        
        return separated_spec
    
    def separate_track(
        self,
        mixture: torch.Tensor,
        sr: int = 44100,
        segment_length: int = 8,
    ) -> torch.Tensor:
        """Separate vocals from a full audio track.
        
        Args:
            mixture: Input audio waveform of shape (batch, channels, samples)
            sr: Sample rate (default: 44100)
            segment_length: Segment length in seconds (default: 8)
            
        Returns:
            Separated vocal waveform of same shape
        """
        from ..utils.audio_utils import stft, istft
        
        batch, channels, samples = mixture.shape
        segment_samples = segment_length * sr
        
        # Process in segments
        num_segments = (samples + segment_samples - 1) // segment_samples
        separated_segments = []
        
        for i in range(num_segments):
            start = i * segment_samples
            end = min(start + segment_samples, samples)
            
            segment = mixture[:, :, start:end]
            
            # Apply STFT to each channel
            specs = []
            for ch in range(channels):
                spec = stft(
                    segment[:, ch, :],
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                )
                specs.append(spec)
            
            # Stack channels: (batch, time, freq, 2, channels)
            spec = torch.stack(specs, dim=-1)
            
            # Process each channel
            separated_specs = []
            for ch in range(channels):
                sep_spec = self.forward(spec[..., ch])  # (batch, time, freq, 2)
                separated_specs.append(sep_spec)
            
            # Apply ISTFT to each channel
            separated_channels = []
            for ch in range(channels):
                audio = istft(
                    separated_specs[ch],
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    length=segment.shape[-1],
                )
                separated_channels.append(audio)
            
            # Stack channels: (batch, channels, samples)
            separated_segment = torch.stack(separated_channels, dim=1)
            separated_segments.append(separated_segment)
        
        # Concatenate segments
        separated = torch.cat(separated_segments, dim=-1)
        
        # Trim to original length
        separated = separated[:, :, :samples]
        
        return separated
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters.
        
        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
