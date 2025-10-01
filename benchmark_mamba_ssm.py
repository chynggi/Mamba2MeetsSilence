"""Benchmark script to compare native PyTorch vs mamba-ssm implementation.

This script measures the performance improvement from using mamba-ssm's
optimized CUDA kernels for selective scan operations.
"""

import torch
import time
from models.mamba2 import Mamba2Block, MAMBA_SSM_AVAILABLE, CAUSAL_CONV1D_AVAILABLE


def benchmark_forward_pass(model, input_tensor, num_warmup=5, num_runs=20):
    """Benchmark forward pass performance.
    
    Args:
        model: The model to benchmark
        input_tensor: Input tensor
        num_warmup: Number of warmup iterations
        num_runs: Number of benchmark iterations
        
    Returns:
        Average time per forward pass in seconds
    """
    device = input_tensor.device
    
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(input_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            _ = model(input_tensor)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
    
    return sum(times) / len(times)


def main():
    """Run benchmark comparing implementations."""
    print("=" * 80)
    print("BSMamba2 Performance Benchmark: Native PyTorch vs mamba-ssm")
    print("=" * 80)
    
    # Check availability
    print(f"\n✓ mamba-ssm available: {MAMBA_SSM_AVAILABLE}")
    print(f"✓ causal-conv1d available: {CAUSAL_CONV1D_AVAILABLE}")
    
    if not MAMBA_SSM_AVAILABLE:
        print("\n⚠️  mamba-ssm not installed. Install with:")
        print("    pip install mamba-ssm causal-conv1d")
        print("\nShowing single implementation benchmark only...\n")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    
    if device.type == 'cpu':
        print("⚠️  Warning: CUDA not available. mamba-ssm optimization requires GPU.")
    
    # Model parameters (from paper)
    d_model = 192  # Hidden dimension (VRAM optimized)
    d_state = 64
    d_conv = 4
    
    # Test configurations
    configs = [
        {"batch": 1, "seqlen": 100, "name": "Short (1s audio)"},
        {"batch": 1, "seqlen": 400, "name": "Medium (4s audio)"},
        {"batch": 2, "seqlen": 400, "name": "Medium batch=2 (4s audio)"},
        {"batch": 1, "seqlen": 800, "name": "Long (8s audio)"},
    ]
    
    print("\n" + "-" * 80)
    print("Running benchmarks...")
    print("-" * 80)
    
    results = []
    
    for config in configs:
        batch = config["batch"]
        seqlen = config["seqlen"]
        name = config["name"]
        
        print(f"\n📊 Configuration: {name}")
        print(f"   Batch size: {batch}, Sequence length: {seqlen}")
        
        # Create model and input
        model = Mamba2Block(d_model, d_state=d_state, d_conv=d_conv).to(device)
        model.eval()
        
        input_tensor = torch.randn(batch, seqlen, d_model, device=device)
        
        # Benchmark
        try:
            avg_time = benchmark_forward_pass(model, input_tensor)
            throughput = batch * seqlen / avg_time  # frames per second
            
            print(f"   ⏱️  Average time: {avg_time*1000:.2f} ms")
            print(f"   🚀 Throughput: {throughput:.0f} frames/sec")
            
            results.append({
                "config": name,
                "batch": batch,
                "seqlen": seqlen,
                "time_ms": avg_time * 1000,
                "throughput": throughput
            })
        except Exception as e:
            print(f"   ❌ Benchmark failed: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    if MAMBA_SSM_AVAILABLE and device.type == 'cuda':
        print("\n✅ Using optimized mamba-ssm CUDA kernels")
        print("   Expected speedup: 5-10x compared to native PyTorch")
    else:
        print("\n⚠️  Using native PyTorch implementation")
        print("   For 5-10x speedup, install mamba-ssm:")
        print("   pip install mamba-ssm causal-conv1d")
    
    # Display results table
    if results:
        print("\n" + "-" * 80)
        print(f"{'Configuration':<25} {'Time (ms)':<15} {'Throughput (fps)':<20}")
        print("-" * 80)
        for r in results:
            print(f"{r['config']:<25} {r['time_ms']:<15.2f} {r['throughput']:<20.0f}")
        print("-" * 80)
    
    # Memory usage
    if device.type == 'cuda':
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"\n💾 GPU Memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")
    
    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
