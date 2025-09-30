"""MUSDB18HQ dataset loader for BSMamba2.

This module provides dataset classes for loading and processing the MUSDB18HQ
dataset for music source separation.
"""

from typing import Optional, Tuple, List
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import musdb
import soundfile as sf


class MUSDB18Dataset(Dataset):
    """MUSDB18HQ dataset for music source separation.
    
    Loads the MUSDB18HQ dataset and provides 8-second segments of mixed audio
    with corresponding source labels.
    
    Args:
        root: Root directory of MUSDB18HQ dataset
        subset: 'train', 'valid', or 'test'
        segment_length: Segment length in seconds (default: 8)
        sample_rate: Target sample rate (default: 44100)
        sources: List of source names (default: ['vocals'])
        random_mix: Whether to randomly mix sources (default: True)
        transform: Optional audio transform function
    """
    
    def __init__(
        self,
        root: str,
        subset: str = 'train',
        segment_length: int = 8,
        sample_rate: int = 44100,
        sources: List[str] = ['vocals'],
        random_mix: bool = True,
        transform: Optional[callable] = None,
    ):
        super().__init__()
        self.root = root
        self.subset = subset
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.sources = sources
        self.random_mix = random_mix
        self.transform = transform
        
        # Load MUSDB18 dataset
        self.mus = musdb.DB(root=root, subsets=subset, is_wav=True)
        
        # Calculate number of segments per track
        self.segment_samples = segment_length * sample_rate
        self.segments_per_track = self._calculate_segments()
        
    def _calculate_segments(self) -> List[int]:
        """Calculate number of segments for each track.
        
        Returns:
            List of segment counts per track
        """
        segments = []
        for track in self.mus.tracks:
            duration = track.duration
            num_segments = int(duration // self.segment_length)
            segments.append(num_segments)
        return segments
    
    def __len__(self) -> int:
        """Get total number of segments."""
        return sum(self.segments_per_track)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single segment.
        
        Args:
            idx: Segment index
            
        Returns:
            Tuple of (mixture, target) tensors of shape (channels, samples)
        """
        # Find track and segment indices
        track_idx = 0
        segment_idx = idx
        
        for i, num_segments in enumerate(self.segments_per_track):
            if segment_idx < num_segments:
                track_idx = i
                break
            segment_idx -= num_segments
        
        # Load track
        track = self.mus.tracks[track_idx]
        
        # Get segment start time
        start_sample = segment_idx * self.segment_samples
        end_sample = start_sample + self.segment_samples
        
        # Load audio sources
        if self.random_mix and self.subset == 'train':
            # Random mixing of sources
            mixture, target = self._random_mix_sources(track, start_sample, end_sample)
        else:
            # Standard loading
            mixture = track.audio[start_sample:end_sample].T  # (channels, samples)
            target_audio = np.zeros_like(mixture)
            
            for source_name in self.sources:
                source = track.sources[source_name].audio[start_sample:end_sample].T
                target_audio += source
            
            target = target_audio
        
        # Convert to tensors
        mixture = torch.from_numpy(mixture).float()
        target = torch.from_numpy(target).float()
        
        # Apply transforms
        if self.transform is not None:
            mixture, target = self.transform(mixture, target)
        
        return mixture, target
    
    def _random_mix_sources(
        self,
        track: musdb.MultiTrack,
        start_sample: int,
        end_sample: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create random mixture from sources.
        
        Args:
            track: MUSDB track
            start_sample: Start sample index
            end_sample: End sample index
            
        Returns:
            Tuple of (mixture, target) arrays
        """
        all_sources = ['vocals', 'drums', 'bass', 'other']
        
        # Load all sources
        source_audios = {}
        for source_name in all_sources:
            audio = track.sources[source_name].audio[start_sample:end_sample].T
            source_audios[source_name] = audio
        
        # Random gain for each source
        mixture = np.zeros_like(source_audios['vocals'])
        for source_name in all_sources:
            gain = random.uniform(0.7, 1.3)
            mixture += source_audios[source_name] * gain
        
        # Target is the isolated source(s)
        target = np.zeros_like(mixture)
        for source_name in self.sources:
            target += source_audios[source_name]
        
        return mixture, target


class InferenceDataset(Dataset):
    """Dataset for inference on single audio files.
    
    Args:
        audio_files: List of audio file paths
        segment_length: Segment length in seconds (default: 8)
        sample_rate: Target sample rate (default: 44100)
    """
    
    def __init__(
        self,
        audio_files: List[str],
        segment_length: int = 8,
        sample_rate: int = 44100,
    ):
        super().__init__()
        self.audio_files = audio_files
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.segment_samples = segment_length * sample_rate
        
        # Load and store all audio files
        self.audio_data = []
        self.num_segments = []
        
        for audio_file in audio_files:
            audio, sr = sf.read(audio_file, always_2d=True)
            
            # Resample if needed
            if sr != sample_rate:
                from librosa import resample
                audio = resample(audio.T, orig_sr=sr, target_sr=sample_rate).T
            
            # Convert to (channels, samples)
            audio = audio.T
            
            self.audio_data.append(audio)
            
            # Calculate number of segments
            num_seg = (audio.shape[1] + self.segment_samples - 1) // self.segment_samples
            self.num_segments.append(num_seg)
    
    def __len__(self) -> int:
        """Get total number of segments."""
        return sum(self.num_segments)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """Get a single segment.
        
        Args:
            idx: Segment index
            
        Returns:
            Tuple of (audio_segment, file_idx, segment_idx)
        """
        # Find file and segment indices
        file_idx = 0
        segment_idx = idx
        
        for i, num_seg in enumerate(self.num_segments):
            if segment_idx < num_seg:
                file_idx = i
                break
            segment_idx -= num_seg
        
        # Get segment
        audio = self.audio_data[file_idx]
        start = segment_idx * self.segment_samples
        end = min(start + self.segment_samples, audio.shape[1])
        
        segment = audio[:, start:end]
        
        # Pad if necessary
        if segment.shape[1] < self.segment_samples:
            padding = self.segment_samples - segment.shape[1]
            segment = np.pad(segment, ((0, 0), (0, padding)), mode='constant')
        
        segment = torch.from_numpy(segment).float()
        
        return segment, file_idx, segment_idx
