"""Performance profiling script for BSMamba2 training.

This script helps identify performance bottlenecks in the training pipeline.
"""

import os
import yaml
import torch
import torch.profiler as profiler
from pathlib import Path
from torch.utils.data import DataLoader

from models.bsmamba2 import BSMamba2
from data.dataset import MUSDB18Dataset
from data.transforms import get_transforms
from training.train import Trainer
from utils.config import load_config


def profile_forward_pass(model, batch, device):
    """Profile a single forward pass."""
    mixture, target = batch
    mixture = mixture.to(device)
    target = target.to(device)
    
    # STFT
    from utils.audio_utils import stft
    mixture_spec = stft(
        mixture.mean(dim=1),  # Mono
        n_fft=2048,
        hop_length=441,
    )
    
    # Forward
    with torch.no_grad():
        pred_spec = model(mixture_spec)
    
    return pred_spec


def profile_training_step(trainer, train_loader, num_steps=10):
    """Profile training steps with PyTorch profiler."""
    
    print(f"Profiling {num_steps} training steps...")
    
    activities = [
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ]
    
    with profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        
        with profiler.record_function("training_steps"):
            trainer.model.train()
            
            for step, (mixture, target) in enumerate(train_loader):
                if step >= num_steps:
                    break
                
                mixture = mixture.to(trainer.device)
                target = target.to(trainer.device)
                
                # Forward
                with profiler.record_function("stft"):
                    mixture_spec = trainer._audio_to_spec(mixture)
                
                with profiler.record_function("model_forward"):
                    pred_spec = trainer.model(mixture_spec)
                
                with profiler.record_function("istft"):
                    pred_audio = trainer._spec_to_audio(pred_spec, mixture.shape[-1])
                
                with profiler.record_function("loss"):
                    loss = trainer.criterion(pred_audio, target)
                
                with profiler.record_function("backward"):
                    loss.backward()
                
                with profiler.record_function("optimizer_step"):
                    trainer.optimizer.step()
                    trainer.optimizer.zero_grad()
                
                print(f"Step {step + 1}/{num_steps} completed")
    
    return prof


def print_profiler_results(prof):
    """Print profiler results in various formats."""
    
    print("\n" + "="*80)
    print("PROFILING RESULTS")
    print("="*80)
    
    # CPU time
    print("\n### Top 10 operations by CPU time ###")
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=10,
    ))
    
    # CUDA time
    print("\n### Top 10 operations by CUDA time ###")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=10,
    ))
    
    # Memory
    print("\n### Top 10 operations by memory usage ###")
    print(prof.key_averages().table(
        sort_by="self_cuda_memory_usage",
        row_limit=10,
    ))
    
    # Grouped by function
    print("\n### Grouped by function name ###")
    print(prof.key_averages(group_by_input_shape=False).table(
        sort_by="cuda_time_total",
        row_limit=10,
    ))


def analyze_model_layers(model):
    """Analyze model layer sizes and parameters."""
    
    print("\n" + "="*80)
    print("MODEL ANALYSIS")
    print("="*80)
    
    total_params = 0
    trainable_params = 0
    
    print(f"\n{'Layer':<50} {'Parameters':<15} {'Trainable':<10}")
    print("-" * 75)
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        
        print(f"{name:<50} {num_params:>12,} {str(param.requires_grad):<10}")
    
    print("-" * 75)
    print(f"{'Total':<50} {total_params:>12,}")
    print(f"{'Trainable':<50} {trainable_params:>12,}")
    print(f"{'Non-trainable':<50} {total_params - trainable_params:>12,}")
    
    # Memory estimate
    param_size_mb = total_params * 4 / (1024 ** 2)  # Assuming float32
    print(f"\nEstimated model size: {param_size_mb:.2f} MB")


def profile_data_loading(train_loader, num_batches=20):
    """Profile data loading speed."""
    
    print("\n" + "="*80)
    print("DATA LOADING ANALYSIS")
    print("="*80)
    
    import time
    
    times = []
    
    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break
        
        start = time.time()
        mixture, target = batch
        # Move to GPU
        mixture_gpu = mixture.cuda()
        target_gpu = target.cuda()
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        times.append(elapsed)
        print(f"Batch {i+1}: {elapsed*1000:.2f} ms")
    
    avg_time = sum(times) / len(times)
    print(f"\nAverage loading time: {avg_time*1000:.2f} ms per batch")
    print(f"Estimated throughput: {1/avg_time:.2f} batches/sec")


def main():
    """Main profiling function."""
    
    # Load config
    config_path = Path(__file__).parent / 'configs' / 'bsmamba2.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override for profiling
    config['training']['batch_size'] = 1
    config['audio']['segment_length'] = 4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Create model
    print("\nCreating model...")
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
    
    # Analyze model
    analyze_model_layers(model)
    
    # Create small dataset for profiling
    print("\nCreating dataset...")
    try:
        train_dataset = MUSDB18Dataset(
            root=config['data']['root'],
            subset='train',
            segment_length=config['audio']['segment_length'],
            sample_rate=config['audio']['sample_rate'],
            sources=['vocals'],
            random_mix=False,
            transform=None,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        
        # Profile data loading
        profile_data_loading(train_loader, num_batches=10)
        
    except Exception as e:
        print(f"Warning: Could not load dataset: {e}")
        print("Skipping data loading profiling...")
        
        # Create dummy data
        dummy_mixture = torch.randn(1, 2, 44100 * 4)  # 4 seconds
        dummy_target = torch.randn(1, 2, 44100 * 4)
        train_loader = [(dummy_mixture, dummy_target)] * 10
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        output_dir='outputs/profiling',
    )
    
    # Profile training
    prof = profile_training_step(trainer, train_loader, num_steps=5)
    
    # Print results
    print_profiler_results(prof)
    
    # Export trace
    output_dir = Path('outputs/profiling')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trace_path = output_dir / 'trace.json'
    prof.export_chrome_trace(str(trace_path))
    print(f"\nTrace exported to: {trace_path}")
    print(f"View in Chrome: chrome://tracing")
    
    # Memory summary
    if torch.cuda.is_available():
        print("\n" + "="*80)
        print("CUDA MEMORY SUMMARY")
        print("="*80)
        print(torch.cuda.memory_summary())


if __name__ == '__main__':
    main()
