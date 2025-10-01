"""Training script for BSMamba2.

This module provides the main training loop and utilities for training BSMamba2.
"""

import os
import logging
from typing import Optional, Dict
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.bsmamba2 import BSMamba2
from data.dataset import MUSDB18Dataset
from data.transforms import get_transforms
from utils.audio_utils import stft, istft
from training.loss import BSMamba2Loss
from training.metrics import MetricsCalculator


logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for BSMamba2 model.
    
    Args:
        model: BSMamba2 model
        config: Configuration dictionary
        device: Device to train on
        output_dir: Output directory for checkpoints and logs
    """
    
    def __init__(
        self,
        model: BSMamba2,
        config: Dict,
        device: torch.device,
        output_dir: str = 'outputs',
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        self.criterion = BSMamba2Loss(
            lambda_time=config['training']['lambda_time'],
            fft_sizes=config['loss']['stft_windows'],
            stft_hop=config['loss']['stft_hop'],
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
        )
        
        # Gradient scaler for mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if config['training']['precision'] == 'bf16' else None
        
        # Tensorboard writer
        self.writer = SummaryWriter(log_dir=self.output_dir / 'logs')
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        
        # Checkpoint settings
        self.save_every_n_epochs = config['training'].get('save_every_n_epochs', 1)
        self.keep_last_n_checkpoints = config['training'].get('keep_last_n_checkpoints', None)
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, (mixture, target) in enumerate(pbar):
            mixture = mixture.to(self.device)  # (batch, channels, samples)
            target = target.to(self.device)
            
            # Apply STFT
            mixture_spec = self._audio_to_spec(mixture)  # (batch, time, freq, 2)
            
            # Forward pass
            with torch.amp.autocast('cuda', enabled=self.scaler is not None):
                pred_spec = self.model(mixture_spec)
                
                # Convert back to time domain
                pred_audio = self._spec_to_audio(pred_spec, mixture.shape[-1])
                
                # Compute loss (normalized by accumulation steps)
                loss = self.criterion(pred_audio, target) / accumulation_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item() * accumulation_steps  # Denormalize for logging
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item() * accumulation_steps})
            
            # Log to tensorboard
            if self.global_step % 100 == 0:
                self.writer.add_scalar('train/loss', loss.item() * accumulation_steps, self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        metrics_calc = MetricsCalculator(
            sample_rate=self.config['audio']['sample_rate'],
            use_museval=False,
        )
        
        with torch.no_grad():
            for mixture, target in tqdm(val_loader, desc='Validation'):
                mixture = mixture.to(self.device)
                target = target.to(self.device)
                
                # Apply STFT
                mixture_spec = self._audio_to_spec(mixture)
                
                # Forward pass
                pred_spec = self.model(mixture_spec)
                
                # Convert back to time domain
                pred_audio = self._spec_to_audio(pred_spec, mixture.shape[-1])
                
                # Compute loss
                loss = self.criterion(pred_audio, target)
                total_loss += loss.item()
                num_batches += 1
                
                # Update metrics
                metrics_calc.update(pred_audio, target)
        
        # Handle empty validation set
        if num_batches == 0:
            logger.warning("Validation set is empty! Skipping validation.")
            return {'loss': float('inf'), 'sdr': 0.0}
        
        avg_loss = total_loss / num_batches
        metrics = metrics_calc.compute()
        metrics['loss'] = avg_loss
        
        return metrics
    
    def _audio_to_spec(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio to complex spectrogram (optimized).
        
        Args:
            audio: Audio of shape (batch, channels, samples)
            
        Returns:
            Spectrogram of shape (batch, time, freq, 2)
        """
        batch, channels, samples = audio.shape
        
        # Average channels first, then compute STFT once (much faster)
        audio_mono = audio.mean(dim=1)  # (batch, samples)
        
        spec = stft(
            audio_mono,
            n_fft=self.config['audio']['n_fft'],
            hop_length=self.config['audio']['hop_length'],
        )
        
        return spec
    
    def _spec_to_audio(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        """Convert spectrogram to audio.
        
        Args:
            spec: Spectrogram of shape (batch, time, freq, 2)
            length: Target audio length
            
        Returns:
            Audio of shape (batch, channels, samples)
        """
        audio = istft(
            spec,
            n_fft=self.config['audio']['n_fft'],
            hop_length=self.config['audio']['hop_length'],
            length=length,
        )
        
        # Expand to stereo
        audio = audio.unsqueeze(1).repeat(1, 2, 1)
        
        return audio
    
    def save_checkpoint(self, filename: str = 'checkpoint.pt', is_best: bool = False):
        """Save training checkpoint.
        
        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config,
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = self.output_dir / filename
        torch.save(checkpoint, checkpoint_path)
        logger.info(f'Saved checkpoint to {filename}')
        
        # Save as last checkpoint if not best model
        if not is_best:
            last_checkpoint_path = self.output_dir / 'last_checkpoint.pt'
            torch.save(checkpoint, last_checkpoint_path)
            logger.info(f'Saved last checkpoint')
        
        # Clean up old checkpoints if limit is set
        if self.keep_last_n_checkpoints is not None and not is_best:
            self._cleanup_old_checkpoints()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f'Loaded checkpoint from {checkpoint_path}')
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoint files, keeping only the last N."""
        # Find all epoch checkpoint files
        checkpoint_files = sorted(
            self.output_dir.glob('checkpoint_epoch_*.pt'),
            key=lambda p: int(p.stem.split('_')[-1])
        )
        
        # Remove oldest checkpoints
        if len(checkpoint_files) > self.keep_last_n_checkpoints:
            for checkpoint_file in checkpoint_files[:-self.keep_last_n_checkpoints]:
                checkpoint_file.unlink()
                logger.info(f'Removed old checkpoint: {checkpoint_file.name}')
    
    def train(self, num_epochs: int, train_loader: DataLoader, val_loader: DataLoader):
        """Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        logger.info(f'Starting training for {num_epochs} epochs')
        logger.info(f'Model parameters: {self.model.get_num_parameters():,}')
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            logger.info(f'Epoch {epoch}: train_loss={train_loss:.4f}')
            
            # Save checkpoint every N epochs (before validation)
            if (epoch + 1) % self.save_every_n_epochs == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
            
            # Validate
            val_metrics = self.validate(val_loader)
            logger.info(f'Epoch {epoch}: val_loss={val_metrics["loss"]:.4f}, '
                       f'cSDR={val_metrics["cSDR"]:.2f}, uSDR={val_metrics["uSDR"]:.2f}')
            
            # Log to tensorboard
            self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('val/cSDR', val_metrics['cSDR'], epoch)
            self.writer.add_scalar('val/uSDR', val_metrics['uSDR'], epoch)
            
            # Update learning rate
            self.scheduler.step(val_metrics['cSDR'])
            
            # Save best model
            if val_metrics['cSDR'] > self.best_metric:
                self.best_metric = val_metrics['cSDR']
                self.save_checkpoint('best_model.pt', is_best=True)
                logger.info(f'New best model with cSDR={self.best_metric:.2f}')
        
        logger.info('Training completed!')


def train_model(config: Dict, device: Optional[torch.device] = None):
    """Main training function.
    
    Args:
        config: Configuration dictionary
        device: Device to train on
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    # Device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Create model
    model = BSMamba2(
        n_fft=config['audio']['n_fft'],
        hop_length=config['audio']['hop_length'],
        num_subbands=config['model']['num_subbands'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        d_state=config['model'].get('d_state', 64),
        d_conv=config['model'].get('d_conv', 4),
        dropout=config['training'].get('dropout', 0.0),
        use_gradient_checkpointing=config['model'].get('use_gradient_checkpointing', False),
    )
    
    # Create datasets
    train_dataset = MUSDB18Dataset(
        root=config['data']['root'],
        subset='train',
        segment_length=config['audio']['segment_length'],
        sample_rate=config['audio']['sample_rate'],
        sources=['vocals'],
        random_mix=True,
        transform=get_transforms('train', config['audio']['sample_rate']),
    )
    
    # Use test subset for validation (as MUSDB18 doesn't have a separate valid set)
    val_dataset = MUSDB18Dataset(
        root=config['data']['root'],
        subset='test',
        segment_length=config['audio']['segment_length'],
        sample_rate=config['audio']['sample_rate'],
        sources=['vocals'],
        random_mix=False,
        transform=get_transforms('valid', config['audio']['sample_rate']),
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,  # Increased for better throughput
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,  # Keep workers alive between epochs
        drop_last=True,  # Drop incomplete batches for consistent training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        output_dir=config['training']['output_dir'],
    )
    
    # Load checkpoint if specified
    if 'resume_from' in config['training'] and config['training']['resume_from']:
        trainer.load_checkpoint(config['training']['resume_from'])
    
    # Train
    trainer.train(
        num_epochs=config['training']['num_epochs'],
        train_loader=train_loader,
        val_loader=val_loader,
    )
