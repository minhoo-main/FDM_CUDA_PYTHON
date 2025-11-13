"""
성능 비교 벤치마크: CPU vs GPU vs Optimized GPU

Phase 1 최적화 효과 검증:
1. CPU baseline
2. 기존 GPU (순차 for loop)
3. Optimized GPU (batched solver + vectorized)
"""

import numpy as np
import time
from src.models.els_product import create_sample_els
from src.pricing.els_pricer import price_els
from src.pricing.gpu_els_pricer import price_els_gpu
from src.pricing.gpu_els_pricer_optimized import price_els_optimized

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  CuPy가 설치되지 않았습니다. GPU 벤치마크를 건너뜁니다.")


def benchmark_cpu(product, N1, N2, Nt, runs=3):
    """CPU 벤치마크"""
    print(f"\n{'='*60}")
    print(f"CPU Benchmark: {N1}×{N2} grid, {Nt} time steps")
    print(f"{'='*60}")

    times = []
    prices = []

    for i in range(runs):
        start = time.time()
        result = price_els(product, N1=N1, N2=N2, Nt=Nt, verbose=False)
        elapsed = time.time() - start

        times.append(elapsed)
        prices.append(result['price'])

        print(f"Run {i+1}/{runs}: {elapsed:.4f}s, Price: {result['price']:.4f}")

    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_price = np.mean(prices)

    print(f"\nAverage: {avg_time:.4f}s ± {std_time:.4f}s")
    print(f"Price: {avg_price:.4f}")

    return {
        'times': times,
        'avg_time': avg_time,
        'std_time': std_time,
        'price': avg_price
    }


def benchmark_gpu_original(product, N1, N2, Nt, runs=3):
    """기존 GPU 벤치마크 (순차 for loop)"""
    if not GPU_AVAILABLE:
        return None

    print(f"\n{'='*60}")
    print(f"Original GPU Benchmark: {N1}×{N2} grid, {Nt} time steps")
    print(f"{'='*60}")

    times = []
    prices = []

    # Warm-up
    _ = price_els_gpu(product, N1=40, N2=40, Nt=50, use_gpu=True, verbose=False)
    cp.cuda.Stream.null.synchronize()

    for i in range(runs):
        cp.cuda.Stream.null.synchronize()
        start = time.time()

        result = price_els_gpu(product, N1=N1, N2=N2, Nt=Nt, use_gpu=True, verbose=False)

        cp.cuda.Stream.null.synchronize()
        elapsed = time.time() - start

        times.append(elapsed)
        prices.append(result['price'])

        print(f"Run {i+1}/{runs}: {elapsed:.4f}s, Price: {result['price']:.4f}")

    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_price = np.mean(prices)

    print(f"\nAverage: {avg_time:.4f}s ± {std_time:.4f}s")
    print(f"Price: {avg_price:.4f}")

    return {
        'times': times,
        'avg_time': avg_time,
        'std_time': std_time,
        'price': avg_price
    }


def benchmark_gpu_optimized(product, N1, N2, Nt, runs=3):
    """최적화된 GPU 벤치마크 (batched + vectorized)"""
    if not GPU_AVAILABLE:
        return None

    print(f"\n{'='*60}")
    print(f"🚀 Optimized GPU Benchmark: {N1}×{N2} grid, {Nt} time steps")
    print(f"{'='*60}")

    times = []
    prices = []

    # Warm-up
    _ = price_els_optimized(product, N1=40, N2=40, Nt=50, use_gpu=True, verbose=False)
    cp.cuda.Stream.null.synchronize()

    for i in range(runs):
        cp.cuda.Stream.null.synchronize()
        start = time.time()

        result = price_els_optimized(product, N1=N1, N2=N2, Nt=Nt, use_gpu=True, verbose=False)

        cp.cuda.Stream.null.synchronize()
        elapsed = time.time() - start

        times.append(elapsed)
        prices.append(result['price'])

        print(f"Run {i+1}/{runs}: {elapsed:.4f}s, Price: {result['price']:.4f}")

    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_price = np.mean(prices)

    print(f"\nAverage: {avg_time:.4f}s ± {std_time:.4f}s")
    print(f"Price: {avg_price:.4f}")

    return {
        'times': times,
        'avg_time': avg_time,
        'std_time': std_time,
        'price': avg_price
    }


def print_comparison(cpu_result, gpu_result, gpu_opt_result):
    """결과 비교 출력"""
    print(f"\n{'='*60}")
    print("📊 Performance Comparison Summary")
    print(f"{'='*60}\n")

    print(f"{'Method':<20} {'Time (s)':<15} {'Speedup':<15} {'Price':<10}")
    print("-" * 60)

    cpu_time = cpu_result['avg_time']
    cpu_price = cpu_result['price']

    print(f"{'CPU':<20} {cpu_time:>10.4f}s     {1.0:>10.1f}x     {cpu_price:>10.4f}")

    if gpu_result:
        gpu_time = gpu_result['avg_time']
        gpu_speedup = cpu_time / gpu_time
        gpu_price = gpu_result['price']
        print(f"{'GPU (Original)':<20} {gpu_time:>10.4f}s     {gpu_speedup:>10.1f}x     {gpu_price:>10.4f}")

    if gpu_opt_result:
        opt_time = gpu_opt_result['avg_time']
        opt_speedup = cpu_time / opt_time
        opt_price = gpu_opt_result['price']
        print(f"{'GPU (Optimized) 🚀':<20} {opt_time:>10.4f}s     {opt_speedup:>10.1f}x     {opt_price:>10.4f}")

        if gpu_result:
            gpu_to_opt = gpu_result['avg_time'] / opt_time
            print(f"\n🎯 GPU Optimization Gain: {gpu_to_opt:.1f}x faster than original GPU")

    print("\n" + "=" * 60)

    # 가격 일치 확인
    print("\n✓ Price Verification:")
    prices = [cpu_price]
    if gpu_result:
        prices.append(gpu_result['price'])
    if gpu_opt_result:
        prices.append(gpu_opt_result['price'])

    price_diff = max(prices) - min(prices)
    print(f"  Price range: {min(prices):.4f} ~ {max(prices):.4f}")
    print(f"  Max difference: {price_diff:.6f} ({price_diff/cpu_price*100:.4f}%)")

    if price_diff / cpu_price < 0.001:  # 0.1% 이내
        print("  ✅ All methods agree (< 0.1% difference)")
    else:
        print("  ⚠️  Price discrepancy detected!")


def run_benchmark_suite():
    """벤치마크 스위트 실행"""
    print("="*60)
    print("ELS FDM Pricer Performance Benchmark")
    print("="*60)
    print("\nPhase 1 Optimization Test:")
    print("1. Batched Tridiagonal Solver")
    print("2. Vectorized Early Redemption Check")
    print("3. Vectorized Terminal Payoff")
    print("="*60)

    # ELS 상품 생성
    product = create_sample_els()

    # 테스트 구성들
    test_configs = [
        # (N1, N2, Nt, name)
        (60, 60, 120, "Small"),
        (80, 80, 150, "Medium"),
        (100, 100, 200, "Large"),
    ]

    for N1, N2, Nt, name in test_configs:
        print(f"\n\n{'#'*60}")
        print(f"# Test: {name} Grid ({N1}×{N2}, {Nt} time steps)")
        print(f"{'#'*60}")

        # CPU 벤치마크
        cpu_result = benchmark_cpu(product, N1, N2, Nt, runs=2)

        # GPU 벤치마크
        gpu_result = benchmark_gpu_original(product, N1, N2, Nt, runs=3)

        # Optimized GPU 벤치마크
        gpu_opt_result = benchmark_gpu_optimized(product, N1, N2, Nt, runs=3)

        # 비교 출력
        print_comparison(cpu_result, gpu_result, gpu_opt_result)


if __name__ == "__main__":
    run_benchmark_suite()
