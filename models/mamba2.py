"""Mamba2 block implementation for BSMamba2.

This module implements the Mamba2 state space model with selective updates
and bidirectional processing capabilities.

Performance Optimization:
- Uses mamba-ssm library for CUDA-optimized selective scan (5-10x faster)
- Falls back to native PyTorch implementation if mamba-ssm is unavailable
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Try to import optimized mamba-ssm selective_scan
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_SSM_AVAILABLE = True
except ImportError:
    MAMBA_SSM_AVAILABLE = False
    print("Warning: mamba-ssm not available. Using slower native PyTorch implementation.")
    print("Install with: pip install mamba-ssm causal-conv1d")

# Try to import causal_conv1d for optimized convolution
try:
    from causal_conv1d import causal_conv1d_fn
    CAUSAL_CONV1D_AVAILABLE = True
except ImportError:
    CAUSAL_CONV1D_AVAILABLE = False
    print("Warning: causal-conv1d not available. Using standard Conv1d.")


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
        # Project to delta, B, C where delta has d_inner dimensions (one per feature)
        self.x_proj = nn.Linear(self.d_inner, self.d_inner + d_state + d_state, bias=False)
        
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
        
        # Combine states if they exist
        if state_fwd is not None and state_bwd is not None:
            final_state = (state_fwd + state_bwd) * 0.5
        else:
            final_state = None
        
        return out, final_state
    
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
        
        # Apply causal convolution (use optimized version if available)
        if CAUSAL_CONV1D_AVAILABLE and x.is_cuda:
            # Optimized CUDA causal conv1d (2-3x faster)
            x = rearrange(x, 'b l d -> b d l')
            # causal_conv1d_fn requires contiguous input
            x = x.contiguous()
            # Get conv1d weight and bias
            weight = self.conv1d.weight  # (d_inner, 1, d_conv)
            bias = self.conv1d.bias if self.conv1d.bias is not None else None
            try:
                x = causal_conv1d_fn(
                    x,
                    weight.squeeze(1),  # Remove groups dimension
                    bias,
                    activation="silu"
                )
            except Exception as e:
                # Fallback to standard conv1d
                print(f"Warning: causal_conv1d_fn failed ({e}), using standard conv1d")
                x = self.conv1d(x)[:, :, :seqlen]
            x = rearrange(x, 'b d l -> b l d')
        else:
            # Standard PyTorch conv1d
            x = rearrange(x, 'b l d -> b d l')
            x = self.conv1d(x)[:, :, :seqlen]  # Truncate to original length
            x = rearrange(x, 'b d l -> b l d')
        
        # Apply activation
        x = F.silu(x)
        
        # State space computation
        x_proj = self.x_proj(x)  # (batch, seqlen, d_inner + d_state + d_state)
        delta, B, C = torch.split(
            x_proj,
            [self.d_inner, self.d_state, self.d_state],
            dim=-1
        )
        
        # Discretize state space parameters
        A = -torch.exp(self.A_log.float())  # (d_inner,)
        delta = F.softplus(delta)  # (batch, seqlen, d_inner)
        
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
        """Selective scan implementation with CUDA optimization.
        
        Implements the core state space computation:
        x(t+1) = A * x(t) + B * u(t)
        y(t) = C * x(t) + D * u(t)
        
        Uses mamba-ssm's optimized CUDA kernels when available (5-10x faster),
        otherwise falls back to PyTorch implementation.
        
        Args:
            u: Input of shape (batch, seqlen, d_inner)
            delta: Time steps of shape (batch, seqlen, d_inner)
            A: State transition of shape (d_inner,)
            B: Input matrix of shape (batch, seqlen, d_state)
            C: Output matrix of shape (batch, seqlen, d_state)
            D: Direct feedthrough of shape (d_inner,)
            state: Optional initial state
            
        Returns:
            Output tensor of shape (batch, seqlen, d_inner)
        """
        if MAMBA_SSM_AVAILABLE and u.is_cuda:
            return self._selective_scan_cuda(u, delta, A, B, C, D, state)
        else:
            return self._selective_scan_pytorch(u, delta, A, B, C, D, state)
    
    def _selective_scan_cuda(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """CUDA-optimized selective scan using mamba-ssm.
        
        This version uses the highly optimized CUDA kernels from mamba-ssm,
        providing 5-10x speedup over the native PyTorch implementation.
        """
        batch, seqlen, d_inner = u.shape
        _, _, d_state = B.shape
        
        # Reshape for mamba-ssm interface
        # mamba-ssm expects:
        # - u: (batch, d_inner, seqlen)
        # - delta: (batch, d_inner, seqlen)  <- FIXED: was incorrectly (batch, d_state, seqlen)
        # - A: (d_inner, d_state)
        # - B: (batch, d_state, seqlen)
        # - C: (batch, d_state, seqlen)
        u_t = rearrange(u, 'b l d -> b d l')
        delta_t = rearrange(delta, 'b l d -> b d l')  # Now delta has d_inner dimension
        B_t = rearrange(B, 'b l n -> b n l')
        C_t = rearrange(C, 'b l n -> b n l')
        
        # Expand A to match (d_inner, d_state) dimensions
        # A is (d_inner,) representing scalar matrix aI
        A_expanded = A.unsqueeze(-1).expand(d_inner, d_state)  # (d_inner, d_state)
        
        # Call optimized selective_scan_fn
        # Note: selective_scan_fn has signature:
        # selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, 
        #                   delta_softplus=True, return_last_state=False)
        try:
            y = selective_scan_fn(
                u_t,           # (batch, d_inner, seqlen)
                delta_t,       # (batch, d_inner, seqlen) <- FIXED
                A_expanded,    # (d_inner, d_state)
                B_t,           # (batch, d_state, seqlen)
                C_t,           # (batch, d_state, seqlen)
                D=D,           # (d_inner,)
                z=None,
                delta_bias=None,
                delta_softplus=False,  # We already applied softplus
                return_last_state=False
            )
            # y is (batch, d_inner, seqlen)
            y = rearrange(y, 'b d l -> b l d')
            return y
        except Exception as e:
            print(f"Warning: CUDA selective scan failed ({e}), falling back to PyTorch")
            return self._selective_scan_pytorch(u, delta, A, B, C, D, state)
    
    def _selective_scan_pytorch(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Native PyTorch selective scan (fallback implementation).
        
        This is the original sequential implementation, slower but compatible
        with all devices and debugging scenarios.
        """
        batch, seqlen, d_inner = u.shape
        _, _, d_state = B.shape
        
        # Initialize state
        if state is None:
            state = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        
        # Discretize A and B
        # delta: (batch, seqlen, d_inner) -> (batch, seqlen, d_inner, 1)
        # A: (d_inner,) -> (1, 1, d_inner, 1)
        deltaA = torch.exp(delta.unsqueeze(-1) * A.view(1, 1, -1, 1))  # (batch, seqlen, d_inner, 1)
        
        # delta: (batch, seqlen, d_inner) -> (batch, seqlen, d_inner, 1)
        # B: (batch, seqlen, d_state) -> (batch, seqlen, 1, d_state)
        # u: (batch, seqlen, d_inner) -> (batch, seqlen, d_inner, 1)
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)  # (batch, seqlen, d_inner, d_state)
        
        # Sequential scan
        ys = []
        x = state
        for i in range(seqlen):
            # Update state: x(t+1) = A * x(t) + B * u(t)
            x = deltaA[:, i] * x + deltaB_u[:, i]  # (batch, d_inner, d_state)
            
            # Compute output: y(t) = C * x(t) + D * u(t)
            # C[:, i]: (batch, d_state)
            # x: (batch, d_inner, d_state)
            # Result: (batch, d_inner)
            y = torch.einsum('bn,bdn->bd', C[:, i], x) + D * u[:, i]  # (batch, d_inner)
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
