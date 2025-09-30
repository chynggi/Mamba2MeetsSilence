"""Mamba2 block implementation for BSMamba2.

This module implements the Mamba2 state space model with selective updates
and bidirectional processing capabilities.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Mamba2Block(nn.Module):
    """Mamba2 block with bidirectional processing.
    
    Implements a state space model with selective updates, using scalar state
    transition matrix A = aI and input-dependent parameters Δ.
    
    Args:
        d_model: Model dimension
        d_state: State dimension (default: 64)
        d_conv: Convolution kernel size (default: 4)
        expand: Expansion factor (default: 2)
        dropout: Dropout rate (default: 0.0)
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolutional layer for local dependencies
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # State space parameters
        self.x_proj = nn.Linear(self.d_inner, d_state + d_state + self.d_inner, bias=False)
        
        # Scalar state transition parameter (A = aI)
        self.A_log = nn.Parameter(torch.randn(self.d_inner))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # Normalization
        self.norm = nn.LayerNorm(self.d_inner)
        
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with bidirectional processing.
        
        Args:
            x: Input tensor of shape (batch, length, d_model)
            state: Optional initial state of shape (batch, d_inner, d_state)
            
        Returns:
            Tuple of (output tensor, final state)
        """
        batch, seqlen, _ = x.shape
        
        # Forward direction
        out_fwd, state_fwd = self._forward_direction(x, state)
        
        # Backward direction
        x_rev = torch.flip(x, dims=[1])
        out_bwd, state_bwd = self._forward_direction(x_rev, state)
        out_bwd = torch.flip(out_bwd, dims=[1])
        
        # Combine bidirectional outputs
        out = (out_fwd + out_bwd) * 0.5
        
        return out, (state_fwd + state_bwd) * 0.5
    
    def _forward_direction(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single direction forward pass.
        
        Args:
            x: Input tensor of shape (batch, length, d_model)
            state: Optional initial state
            
        Returns:
            Tuple of (output tensor, final state)
        """
        batch, seqlen, _ = x.shape
        
        # Input projection with gating
        xz = self.in_proj(x)  # (batch, seqlen, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each: (batch, seqlen, d_inner)
        
        # Apply causal convolution
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :seqlen]  # Truncate to original length
        x = rearrange(x, 'b d l -> b l d')
        
        # Apply activation
        x = F.silu(x)
        
        # State space computation
        x_proj = self.x_proj(x)  # (batch, seqlen, d_state + d_state + d_inner)
        delta, B, C = torch.split(
            x_proj,
            [self.d_state, self.d_state, self.d_inner],
            dim=-1
        )
        
        # Discretize state space parameters
        A = -torch.exp(self.A_log.float())  # (d_inner,)
        delta = F.softplus(delta)  # (batch, seqlen, d_state)
        
        # Apply state space model
        y = self._selective_scan(x, delta, A, B, C, self.D, state)
        
        # Gating
        y = y * F.silu(z)
        
        # Output projection
        out = self.out_proj(y)
        out = self.dropout(out)
        
        # Return output and final state (simplified)
        return out, None
    
    def _selective_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Selective scan implementation.
        
        Implements the core state space computation:
        x(t+1) = A * x(t) + B * u(t)
        y(t) = C * x(t) + D * u(t)
        
        Args:
            u: Input of shape (batch, seqlen, d_inner)
            delta: Time steps of shape (batch, seqlen, d_state)
            A: State transition of shape (d_inner,)
            B: Input matrix of shape (batch, seqlen, d_state)
            C: Output matrix of shape (batch, seqlen, d_inner)
            D: Direct feedthrough of shape (d_inner,)
            state: Optional initial state
            
        Returns:
            Output tensor of shape (batch, seqlen, d_inner)
        """
        batch, seqlen, d_inner = u.shape
        d_state = delta.shape[-1]
        
        # Initialize state
        if state is None:
            state = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        
        # Discretize A and B
        deltaA = torch.exp(delta.unsqueeze(2) * A.view(1, 1, -1, 1))  # (batch, seqlen, d_inner, 1)
        deltaB = delta.unsqueeze(2) * B.unsqueeze(2)  # (batch, seqlen, d_inner, d_state)
        
        # Sequential scan
        ys = []
        x = state
        for i in range(seqlen):
            x = deltaA[:, i] * x + deltaB[:, i] * u[:, i].unsqueeze(-1)
            y = torch.sum(C[:, i].unsqueeze(1) * x, dim=-1) + D * u[:, i]
            ys.append(y)
        
        y = torch.stack(ys, dim=1)  # (batch, seqlen, d_inner)
        
        return y


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    Args:
        d_model: Model dimension
        eps: Epsilon for numerical stability
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.
        
        Args:
            x: Input tensor of shape (batch, ..., d_model)
            
        Returns:
            Normalized tensor
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.weight * x_norm
