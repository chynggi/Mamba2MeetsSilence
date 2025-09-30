"""Quick benchmark script to compare before/after optimization.

Usage:
    python benchmark_speed.py
"""

import time
import torch
import yaml
from pathlib import Path

from models.bsmamba2 import BSMamba2
from training.loss import BSMamba2Loss
from utils.audio_utils import stft, istft


def benchmark_forward_pass(model, batch_size, seq_length, device, num_runs=10):
    """Benchmark model forward pass."""
    
    # Create dummy input
    time_steps = seq_length * 44100 // 441  # ~400 for 4 seconds
    freq_bins = 1025
    
    dummy_spec = torch.randn(batch_size, time_steps, freq_bins, 2).to(device)
    
    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_spec)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            output = model(dummy_spec)
            torch.cuda.synchronize()
    
    elapsed = time.time() - start
    avg_time = elapsed / num_runs
    
    return avg_time


def benchmark_loss_computation(loss_fn, batch_size, seq_length, device, num_runs=10):
    """Benchmark loss computation."""
    
    # Create dummy audio
    samples = seq_length * 44100
    dummy_pred = torch.randn(batch_size, 2, samples).to(device)
    dummy_target = torch.randn(batch_size, 2, samples).to(device)
    
    # Warmup
    for _ in range(3):
        _ = loss_fn(dummy_pred, dummy_target)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_runs):
        loss = loss_fn(dummy_pred, dummy_target)
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    avg_time = elapsed / num_runs
    
    return avg_time


def benchmark_stft(batch_size, seq_length, device, num_runs=10):
    """Benchmark STFT transformation."""
    
    samples = seq_length * 44100
    dummy_audio = torch.randn(batch_size, samples).to(device)
    
    # Warmup
    for _ in range(3):
        _ = stft(dummy_audio, n_fft=2048, hop_length=441)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_runs):
        spec = stft(dummy_audio, n_fft=2048, hop_length=441)
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    avg_time = elapsed / num_runs
    
    return avg_time


def print_benchmark_results(results):
    """Print benchmark results in a nice format."""
    
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    print(f"\n{'Component':<30} {'Time (ms)':<15} {'Time (s)':<15}")
    print("-" * 60)
    
    total_time = 0
    for name, time_s in results.items():
        time_ms = time_s * 1000
        total_time += time_s
        print(f"{name:<30} {time_ms:>10.2f} {time_s:>10.3f}")
    
    print("-" * 60)
    print(f"{'TOTAL (1 step estimate)':<30} {total_time*1000:>10.2f} {total_time:>10.3f}")
    print(f"{'Steps per minute':<30} {60/total_time:>10.1f}")
    print(f"{'Steps per hour':<30} {3600/total_time:>10.0f}")


def main():
    """Main benchmark function."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load config
    config_path = Path(__file__).parent / 'configs' / 'bsmamba2.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Parameters
    batch_size = 1
    seq_length = 4  # seconds
    num_runs = 5
    
    print(f"\nBenchmark parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_length} seconds")
    print(f"  Number of runs: {num_runs}")
    
    results = {}
    
    # 1. Benchmark STFT
    print("\nBenchmarking STFT transformation...")
    stft_time = benchmark_stft(batch_size, seq_length, device, num_runs)
    results['STFT (mono)'] = stft_time
    print(f"  Average time: {stft_time*1000:.2f} ms")
    
    # 2. Benchmark model
    print("\nBenchmarking model forward pass...")
    model = BSMamba2(
        n_fft=config['audio']['n_fft'],
        hop_length=config['audio']['hop_length'],
        num_subbands=config['model']['num_subbands'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        d_state=config['model'].get('d_state', 64),
        d_conv=config['model'].get('d_conv', 4),
        dropout=0.0,
        use_gradient_checkpointing=False,  # Disable for benchmarking
    ).to(device)
    
    model_time = benchmark_forward_pass(model, batch_size, seq_length, device, num_runs)
    results['Model forward'] = model_time
    print(f"  Average time: {model_time*1000:.2f} ms")
    
    # 3. Benchmark ISTFT
    print("\nBenchmarking ISTFT transformation...")
    # ISTFT is roughly same as STFT
    results['ISTFT (stereo)'] = stft_time
    
    # 4. Benchmark loss
    print("\nBenchmarking loss computation...")
    loss_fn = BSMamba2Loss(
        lambda_time=config['training']['lambda_time'],
        fft_sizes=config['loss']['stft_windows'],
        stft_hop=config['loss']['stft_hop'],
    ).to(device)
    
    loss_time = benchmark_loss_computation(loss_fn, batch_size, seq_length, device, num_runs)
    results['Loss computation'] = loss_time
    print(f"  Average time: {loss_time*1000:.2f} ms")
    
    # 5. Estimate backward pass (roughly 2x forward)
    results['Backward pass (est.)'] = model_time * 2
    
    # 6. Estimate optimizer step (small)
    results['Optimizer step (est.)'] = 0.05  # 50ms estimate
    
    # Print results
    print_benchmark_results(results)
    
    # Memory info
    if torch.cuda.is_available():
        print("\n" + "="*80)
        print("MEMORY USAGE")
        print("="*80)
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    total_time = sum(results.values())
    
    if total_time > 30:
        print("⚠️  Warning: Training is very slow (>30s per step)")
        print("\nSuggestions:")
        print("  1. Reduce model size (num_layers, hidden_dim)")
        print("  2. Reduce sequence length")
        print("  3. Use gradient checkpointing sparingly")
        print("  4. Consider model compilation (torch.compile)")
    elif total_time > 10:
        print("⚠️  Training is somewhat slow (>10s per step)")
        print("\nSuggestions:")
        print("  1. Verify GPU is being utilized (check nvidia-smi)")
        print("  2. Increase batch size if memory allows")
        print("  3. Use mixed precision training")
    else:
        print("✅ Training speed looks reasonable!")
        print("\nOptional optimizations:")
        print("  1. Increase batch size for better GPU utilization")
        print("  2. Use torch.compile for additional speedup")
        print("  3. Consider increasing num_workers for data loading")


if __name__ == '__main__':
    main()
