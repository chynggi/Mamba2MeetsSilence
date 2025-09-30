"""Audio transforms for data augmentation and preprocessing.

This module provides various audio transformations for training BSMamba2.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torchaudio
import random


class AudioTransform:
    """Base class for audio transformations.
    
    Args:
        sample_rate: Sample rate of audio (default: 44100)
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def __call__(
        self,
        mixture: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply transformation.
        
        Args:
            mixture: Mixture audio of shape (channels, samples)
            target: Target audio of shape (channels, samples)
            
        Returns:
            Transformed (mixture, target) tuple
        """
        raise NotImplementedError


class RandomGain(AudioTransform):
    """Apply random gain to audio.
    
    Args:
        min_gain: Minimum gain in dB (default: -5)
        max_gain: Maximum gain in dB (default: 5)
        sample_rate: Sample rate (default: 44100)
    """
    
    def __init__(
        self,
        min_gain: float = -5.0,
        max_gain: float = 5.0,
        sample_rate: int = 44100,
    ):
        super().__init__(sample_rate)
        self.min_gain = min_gain
        self.max_gain = max_gain
    
    def __call__(
        self,
        mixture: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random gain."""
        gain_db = random.uniform(self.min_gain, self.max_gain)
        gain_linear = 10 ** (gain_db / 20)
        
        mixture = mixture * gain_linear
        target = target * gain_linear
        
        return mixture, target


class RandomFlip(AudioTransform):
    """Randomly flip audio channels.
    
    Args:
        p: Probability of flipping (default: 0.5)
        sample_rate: Sample rate (default: 44100)
    """
    
    def __init__(self, p: float = 0.5, sample_rate: int = 44100):
        super().__init__(sample_rate)
        self.p = p
    
    def __call__(
        self,
        mixture: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly flip channels."""
        if random.random() < self.p and mixture.shape[0] == 2:
            mixture = mixture.flip(0)
            target = target.flip(0)
        
        return mixture, target


class RandomPhaseShift(AudioTransform):
    """Apply random phase shift to audio.
    
    Args:
        max_shift: Maximum phase shift in samples (default: 100)
        sample_rate: Sample rate (default: 44100)
    """
    
    def __init__(self, max_shift: int = 100, sample_rate: int = 44100):
        super().__init__(sample_rate)
        self.max_shift = max_shift
    
    def __call__(
        self,
        mixture: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random phase shift."""
        shift = random.randint(-self.max_shift, self.max_shift)
        
        if shift > 0:
            mixture = torch.nn.functional.pad(mixture, (shift, 0))[:, :-shift]
            target = torch.nn.functional.pad(target, (shift, 0))[:, :-shift]
        elif shift < 0:
            mixture = torch.nn.functional.pad(mixture, (0, -shift))[:, -shift:]
            target = torch.nn.functional.pad(target, (0, -shift))[:, -shift:]
        
        return mixture, target


class Normalize(AudioTransform):
    """Normalize audio to target RMS level.
    
    Args:
        target_rms: Target RMS level (default: 0.1)
        sample_rate: Sample rate (default: 44100)
    """
    
    def __init__(self, target_rms: float = 0.1, sample_rate: int = 44100):
        super().__init__(sample_rate)
        self.target_rms = target_rms
    
    def __call__(
        self,
        mixture: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize audio."""
        # Calculate current RMS
        current_rms = torch.sqrt(torch.mean(mixture ** 2))
        
        # Avoid division by zero
        if current_rms > 1e-8:
            scale = self.target_rms / current_rms
            mixture = mixture * scale
            target = target * scale
        
        return mixture, target


class Compose:
    """Compose multiple transforms.
    
    Args:
        transforms: List of transform functions
    """
    
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(
        self,
        mixture: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply all transforms sequentially."""
        for transform in self.transforms:
            mixture, target = transform(mixture, target)
        return mixture, target


def get_transforms(
    subset: str = 'train',
    sample_rate: int = 44100,
) -> Optional[Compose]:
    """Get transforms for dataset subset.
    
    Args:
        subset: 'train', 'valid', or 'test'
        sample_rate: Sample rate (default: 44100)
        
    Returns:
        Composed transforms or None
    """
    if subset == 'train':
        return Compose([
            RandomGain(min_gain=-5.0, max_gain=5.0, sample_rate=sample_rate),
            RandomFlip(p=0.5, sample_rate=sample_rate),
            RandomPhaseShift(max_shift=100, sample_rate=sample_rate),
            Normalize(target_rms=0.1, sample_rate=sample_rate),
        ])
    elif subset in ['valid', 'test']:
        return Compose([
            Normalize(target_rms=0.1, sample_rate=sample_rate),
        ])
    else:
        return None
