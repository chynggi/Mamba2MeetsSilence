"""Audio utilities for STFT/ISTFT operations.

This module provides utilities for Short-Time Fourier Transform operations.
"""

from typing import Optional
import torch
import torch.nn.functional as F


# Global window cache to avoid recreating windows
_window_cache = {}


def get_window(window_type: str, window_length: int, device: torch.device) -> torch.Tensor:
    """Get window function with caching.
    
    Args:
        window_type: Window type ('hann', 'hamming', 'blackman')
        window_length: Window length
        device: Device to create window on
        
    Returns:
        Window tensor
    """
    key = (window_type, window_length, device)
    if key not in _window_cache:
        if window_type == 'hann':
            _window_cache[key] = torch.hann_window(window_length, device=device)
        elif window_type == 'hamming':
            _window_cache[key] = torch.hamming_window(window_length, device=device)
        elif window_type == 'blackman':
            _window_cache[key] = torch.blackman_window(window_length, device=device)
        else:
            raise ValueError(f'Unknown window type: {window_type}')
    return _window_cache[key]


def stft(
    audio: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 441,
    window: str = 'hann',
    center: bool = True,
    normalized: bool = False,
    return_complex: bool = False,
) -> torch.Tensor:
    """Compute Short-Time Fourier Transform.
    
    Args:
        audio: Input audio of shape (batch, samples)
        n_fft: FFT size
        hop_length: Hop length
        window: Window type
        center: Whether to center the window
        normalized: Whether to normalize
        return_complex: Whether to return complex tensor
        
    Returns:
        Complex spectrogram of shape (batch, time, freq, 2) if not return_complex,
        else (batch, freq, time) as complex tensor
    """
    window_tensor = get_window(window, n_fft, audio.device)
    
    # Compute STFT
    spec = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window_tensor,
        center=center,
        normalized=normalized,
        return_complex=True,
    )
    
    if return_complex:
        return spec  # (batch, freq, time)
    else:
        # Convert to (batch, time, freq, 2) format
        spec = torch.view_as_real(spec)  # (batch, freq, time, 2)
        spec = spec.permute(0, 2, 1, 3)  # (batch, time, freq, 2)
        return spec


def istft(
    spec: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 441,
    window: str = 'hann',
    center: bool = True,
    normalized: bool = False,
    length: Optional[int] = None,
) -> torch.Tensor:
    """Compute Inverse Short-Time Fourier Transform.
    
    Args:
        spec: Complex spectrogram of shape (batch, time, freq, 2)
        n_fft: FFT size
        hop_length: Hop length
        window: Window type
        center: Whether window was centered
        normalized: Whether to normalize
        length: Target audio length
        
    Returns:
        Audio of shape (batch, samples)
    """
    window_tensor = get_window(window, n_fft, spec.device)
    
    # Convert from (batch, time, freq, 2) to complex tensor
    spec = spec.permute(0, 2, 1, 3)  # (batch, freq, time, 2)
    spec = torch.view_as_complex(spec.contiguous())  # (batch, freq, time)
    
    # Compute ISTFT
    audio = torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window_tensor,
        center=center,
        normalized=normalized,
        length=length,
    )
    
    return audio


def compute_magnitude(spec: torch.Tensor) -> torch.Tensor:
    """Compute magnitude from complex spectrogram.
    
    Args:
        spec: Complex spectrogram of shape (batch, time, freq, 2)
        
    Returns:
        Magnitude of shape (batch, time, freq)
    """
    real = spec[..., 0]
    imag = spec[..., 1]
    magnitude = torch.sqrt(real ** 2 + imag ** 2)
    return magnitude


def compute_phase(spec: torch.Tensor) -> torch.Tensor:
    """Compute phase from complex spectrogram.
    
    Args:
        spec: Complex spectrogram of shape (batch, time, freq, 2)
        
    Returns:
        Phase of shape (batch, time, freq)
    """
    real = spec[..., 0]
    imag = spec[..., 1]
    phase = torch.atan2(imag, real)
    return phase


def apply_mask(
    spec: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply mask to spectrogram.
    
    Args:
        spec: Complex spectrogram of shape (batch, time, freq, 2)
        mask: Mask of shape (batch, time, freq) or (batch, time, freq, 1)
        
    Returns:
        Masked spectrogram of same shape as input
    """
    if mask.ndim == 3:
        mask = mask.unsqueeze(-1)  # (batch, time, freq, 1)
    
    return spec * mask


def normalize_audio(
    audio: torch.Tensor,
    target_level: float = -20.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize audio to target dB level.
    
    Args:
        audio: Input audio of shape (batch, samples) or (batch, channels, samples)
        target_level: Target level in dB
        eps: Small constant for numerical stability
        
    Returns:
        Normalized audio of same shape
    """
    # Compute current level
    rms = torch.sqrt(torch.mean(audio ** 2, dim=-1, keepdim=True) + eps)
    current_level = 20 * torch.log10(rms + eps)
    
    # Compute scale
    scale = 10 ** ((target_level - current_level) / 20)
    
    # Apply scale
    normalized = audio * scale
    
    return normalized


def trim_silence(
    audio: torch.Tensor,
    threshold_db: float = -40.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    """Trim silence from audio.
    
    Args:
        audio: Input audio of shape (batch, samples) or (batch, channels, samples)
        threshold_db: Threshold in dB below which to trim
        frame_length: Frame length for energy computation
        hop_length: Hop length for energy computation
        
    Returns:
        Trimmed audio
    """
    # Compute frame energies
    audio_squared = audio ** 2
    
    # Unfold into frames
    if audio.ndim == 2:
        frames = audio_squared.unfold(-1, frame_length, hop_length)
    else:
        frames = audio_squared.unfold(-1, frame_length, hop_length)
    
    # Compute energy in dB
    energy = torch.mean(frames, dim=-1)
    energy_db = 10 * torch.log10(energy + 1e-8)
    
    # Find non-silent frames
    non_silent = energy_db > threshold_db
    
    # Find start and end indices
    if non_silent.any():
        start_frame = non_silent.to(torch.int).argmax(dim=-1)
        end_frame = non_silent.shape[-1] - non_silent.flip(-1).to(torch.int).argmax(dim=-1)
        
        start_sample = start_frame * hop_length
        end_sample = end_frame * hop_length + frame_length
        
        # Trim
        if audio.ndim == 2:
            audio = audio[:, start_sample:end_sample]
        else:
            audio = audio[:, :, start_sample:end_sample]
    
    return audio
