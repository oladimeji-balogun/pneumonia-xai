import time
from typing import Dict

import torch
import numpy as np

from src.models.base import BaseModel

# Input size matching our preprocessing pipeline
_INPUT_SHAPE = (1, 3, 224, 224)
_WARMUP_RUNS = 10
_BENCHMARK_RUNS = 50


def _count_flops(model: BaseModel, device: torch.device) -> int:
    """
    Estimate FLOPs using a hook-based approach.
    Counts multiply-accumulate operations (MACs) for Conv2d and Linear layers,
    then multiplies by 2 to get FLOPs (1 MAC = 2 FLOPs).
    """
    flops = [0]

    def conv_hook(module, input, output):
        batch = input[0].size(0)
        in_c = input[0].size(1)
        out_c = output.size(1)
        out_h, out_w = output.size(2), output.size(3)
        kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        groups = module.groups
        # MACs = out_h * out_w * out_c * (in_c/groups) * kh * kw
        macs = out_h * out_w * out_c * (in_c // groups) * kh * kw
        flops[0] += 2 * macs * batch

    def linear_hook(module, input, output):
        batch = input[0].size(0)
        macs = module.in_features * module.out_features
        flops[0] += 2 * macs * batch

    hooks = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        dummy = torch.randn(_INPUT_SHAPE).to(device)
        model(dummy)

    for h in hooks:
        h.remove()

    return flops[0]


def _measure_inference_time(
    model: BaseModel,
    device: torch.device,
    batch_size: int = 1,
) -> Dict[str, float]:
    """
    Measure inference latency in milliseconds.

    Runs warmup iterations first to eliminate CUDA initialization overhead,
    then benchmarks over multiple runs and reports mean and std.

    Args:
        model: Trained model.
        device: Device to benchmark on.
        batch_size: Batch size for inference. Default: 1 (single-sample latency).

    Returns:
        Dict with keys: mean_ms, std_ms.
    """
    model.eval()
    dummy = torch.randn(batch_size, 3, 224, 224).to(device)
    timings = []

    # Warmup
    with torch.no_grad():
        for _ in range(_WARMUP_RUNS):
            _ = model(dummy)

    # Synchronize before timing if on CUDA
    if device.type == "cuda":
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(_BENCHMARK_RUNS):
            start = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append((end - start) * 1000)  # convert to ms

    return {
        "mean_ms": float(np.mean(timings)),
        "std_ms":  float(np.std(timings)),
    }


def benchmark_model(
    model: BaseModel,
    device: torch.device = None,
) -> Dict:
    """
    Run full computational benchmark for a model.

    Reports:
    - Total and trainable parameter count
    - FLOPs (floating point operations) for a single forward pass
    - Inference latency on GPU (mean ± std over 50 runs)
    - Inference latency on CPU (mean ± std over 50 runs)

    Args:
        model: Any BaseModel subclass.
        device: Primary device (used for GPU timing). Auto-detected if None.

    Returns:
        Dict with benchmark results.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = model.param_count
    flops = _count_flops(model, device)
    gpu_timing = _measure_inference_time(model, device, batch_size=1)

    # Always benchmark CPU separately for resource-constrained deployment context
    cpu_device = torch.device("cpu")
    model_cpu = model.to(cpu_device)
    cpu_timing = _measure_inference_time(model_cpu, cpu_device, batch_size=1)
    model.to(device)  # restore to original device

    return {
        "model_name":        model.model_name,
        "total_params":      params["total"],
        "trainable_params":  params["trainable"],
        "flops":             flops,
        "gpu_mean_ms":       gpu_timing["mean_ms"],
        "gpu_std_ms":        gpu_timing["std_ms"],
        "cpu_mean_ms":       cpu_timing["mean_ms"],
        "cpu_std_ms":        cpu_timing["std_ms"],
    }


def print_benchmark(results: Dict):
    """Pretty-print benchmark results."""
    print(f"\nBenchmark — {results['model_name']}")
    print("-" * 45)
    print(f"  Parameters (trainable): {results['trainable_params']:>12,}")
    print(f"  FLOPs:                  {results['flops']:>12,}")
    print(f"  GPU inference:          {results['gpu_mean_ms']:>8.2f} ± {results['gpu_std_ms']:.2f} ms")
    print(f"  CPU inference:          {results['cpu_mean_ms']:>8.2f} ± {results['cpu_std_ms']:.2f} ms")