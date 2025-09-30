"""BSMamba2 model components.

This module implements the Band-Split, Dual-Path, and Mask Estimation modules
for the BSMamba2 voice separation model.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mamba2 import Mamba2Block, RMSNorm


class BandSplitModule(nn.Module):
    """Band-Split module for frequency decomposition.
    
    Splits the frequency axis into K sub-bands and applies MLP for
    feature extraction on each sub-band.
    
    Args:
        n_fft: FFT size
        num_subbands: Number of sub-bands (K)
        hidden_dim: Hidden dimension for features
        freq_dim: Original frequency dimension (n_fft // 2 + 1)
    """
    
    def __init__(
        self,
        n_fft: int = 2048,
        num_subbands: int = 62,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_subbands = num_subbands
        self.freq_dim = n_fft // 2 + 1  # 1025 for n_fft=2048
        self.hidden_dim = hidden_dim
        
        # Calculate sub-band boundaries
        self.band_boundaries = self._calculate_band_boundaries()
        
        # MLP for each sub-band
        # Each band has (freq_bins * 2) dimensions (real + imag)
        self.band_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear((self.band_boundaries[i+1] - self.band_boundaries[i]) * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for i in range(num_subbands)
        ])
        
    def _calculate_band_boundaries(self) -> List[int]:
        """Calculate sub-band boundaries.
        
        Returns:
            List of boundary indices for K+1 positions
        """
        # Equal split of frequency axis
        boundaries = []
        freq_per_band = self.freq_dim / self.num_subbands
        
        for i in range(self.num_subbands + 1):
            boundaries.append(int(i * freq_per_band))
        
        return boundaries
    
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Split spectrum into sub-bands and extract features.
        
        Args:
            spec: Complex spectrogram of shape (batch, time, freq, 2)
                  where last dim is [real, imag]
            
        Returns:
            Band features of shape (batch, time, num_subbands, hidden_dim)
        """
        batch, time, freq, _ = spec.shape
        
        # Flatten real and imaginary parts
        spec_flat = spec.view(batch, time, freq * 2)  # (batch, time, freq*2)
        
        # Process each sub-band
        band_features = []
        for i in range(self.num_subbands):
            start = self.band_boundaries[i]
            end = self.band_boundaries[i+1]
            
            # Extract band (both real and imag)
            band_real = spec[:, :, start:end, 0]  # (batch, time, band_width)
            band_imag = spec[:, :, start:end, 1]  # (batch, time, band_width)
            band = torch.cat([band_real, band_imag], dim=-1)  # (batch, time, band_width*2)
            
            band_feat = self.band_mlps[i](band)  # (batch, time, hidden_dim)
            band_features.append(band_feat)
        
        # Stack bands
        band_features = torch.stack(band_features, dim=2)  # (batch, time, K, hidden_dim)
        
        return band_features


class DualPathModule(nn.Module):
    """Dual-Path module with Mamba2 blocks.
    
    Applies bidirectional Mamba2 blocks along time and band axes.
    
    Args:
        hidden_dim: Hidden dimension
        num_layers: Number of dual-path layers (L)
        d_state: State dimension for Mamba2
        d_conv: Convolution size for Mamba2
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        d_state: int = 64,
        d_conv: int = 4,
        dropout: float = 0.0,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Time-axis Mamba2 blocks
        self.time_blocks = nn.ModuleList([
            nn.ModuleDict({
                'mamba': Mamba2Block(hidden_dim, d_state, d_conv, expand=2, dropout=dropout),
                'norm': RMSNorm(hidden_dim),
            })
            for _ in range(num_layers)
        ])
        
        # Band-axis Mamba2 blocks
        self.band_blocks = nn.ModuleList([
            nn.ModuleDict({
                'mamba': Mamba2Block(hidden_dim, d_state, d_conv, expand=2, dropout=dropout),
                'norm': RMSNorm(hidden_dim),
            })
            for _ in range(num_layers)
        ])
        
    def _process_time_axis(self, x, time_block):
        """Process time axis with Mamba2 block."""
        x_norm = time_block['norm'](x)
        x_out, _ = time_block['mamba'](x_norm)
        return x + x_out
    
    def _process_band_axis(self, x, band_block):
        """Process band axis with Mamba2 block."""
        x_norm = band_block['norm'](x)
        x_out, _ = band_block['mamba'](x_norm)
        return x + x_out
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dual-path processing.
        
        Args:
            x: Band features of shape (batch, time, num_bands, hidden_dim)
            
        Returns:
            Processed features of same shape
        """
        batch, time, num_bands, hidden_dim = x.shape
        
        for layer_idx in range(self.num_layers):
            # Time-axis processing
            # Reshape to process each band independently
            x_time = x.view(batch * num_bands, time, hidden_dim)
            
            # Apply Mamba2 block with residual connection (with optional gradient checkpointing)
            if self.use_gradient_checkpointing and self.training:
                x_time = torch.utils.checkpoint.checkpoint(
                    self._process_time_axis,
                    x_time,
                    self.time_blocks[layer_idx],
                    use_reentrant=False
                )
            else:
                x_time = self._process_time_axis(x_time, self.time_blocks[layer_idx])
            
            # Reshape back
            x = x_time.view(batch, num_bands, time, hidden_dim)
            x = x.transpose(1, 2)  # (batch, time, num_bands, hidden_dim)
            
            # Band-axis processing
            # Reshape to process each time step independently
            x_band = x.view(batch * time, num_bands, hidden_dim)
            
            # Apply Mamba2 block with residual connection (with optional gradient checkpointing)
            if self.use_gradient_checkpointing and self.training:
                x_band = torch.utils.checkpoint.checkpoint(
                    self._process_band_axis,
                    x_band,
                    self.band_blocks[layer_idx],
                    use_reentrant=False
                )
            else:
                x_band = self._process_band_axis(x_band, self.band_blocks[layer_idx])
            
            # Reshape back
            x = x_band.view(batch, time, num_bands, hidden_dim)
        
        return x


class MaskEstimationModule(nn.Module):
    """Mask Estimation module.
    
    Generates time-frequency masks for source separation using RMSNorm,
    Linear layers, Tanh activation, and GLU.
    
    Args:
        n_fft: FFT size
        num_subbands: Number of sub-bands
        hidden_dim: Hidden dimension
    """
    
    def __init__(
        self,
        n_fft: int = 2048,
        num_subbands: int = 62,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_subbands = num_subbands
        self.freq_dim = n_fft // 2 + 1
        self.hidden_dim = hidden_dim
        
        # Calculate sub-band boundaries (same as BandSplitModule)
        self.band_boundaries = self._calculate_band_boundaries()
        
        # Mask generation for each sub-band
        self.mask_mlps = nn.ModuleList([
            nn.Sequential(
                RMSNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GLU(dim=-1),
                nn.Linear(hidden_dim, (self.band_boundaries[i+1] - self.band_boundaries[i]) * 2),
                nn.Tanh(),
            )
            for i in range(num_subbands)
        ])
        
    def _calculate_band_boundaries(self) -> List[int]:
        """Calculate sub-band boundaries."""
        boundaries = []
        freq_per_band = self.freq_dim / self.num_subbands
        
        for i in range(self.num_subbands + 1):
            boundaries.append(int(i * freq_per_band))
        
        return boundaries
    
    def forward(self, band_features: torch.Tensor, original_spec: torch.Tensor) -> torch.Tensor:
        """Generate and apply time-frequency mask.
        
        Args:
            band_features: Features of shape (batch, time, num_bands, hidden_dim)
            original_spec: Original spectrogram of shape (batch, time, freq, 2)
            
        Returns:
            Masked spectrogram of same shape as original_spec
        """
        batch, time, num_bands, _ = band_features.shape
        
        # Generate masks for each sub-band
        masks = []
        for i in range(num_bands):
            band_feat = band_features[:, :, i, :]  # (batch, time, hidden_dim)
            mask = self.mask_mlps[i](band_feat)  # (batch, time, band_width*2)
            masks.append(mask)
        
        # Concatenate masks along frequency axis
        mask = torch.cat(masks, dim=-1)  # (batch, time, freq*2)
        
        # Reshape to match spectrogram shape
        mask = mask.view(batch, time, self.freq_dim, 2)
        
        # Apply mask to original spectrogram
        masked_spec = original_spec * mask
        
        return masked_spec
