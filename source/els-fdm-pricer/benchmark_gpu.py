#!/usr/bin/env python3
"""
GPU vs CPU 성능 벤치마크

CUDA를 이용한 속도 개선 효과 측정
"""

import sys
import time
import numpy as np

sys.path.insert(0, '/home/minhoo/els-fdm-pricer')

from src.models.els_product import create_sample_els
from src.pricing.els_pricer import price_els

try:
    from src.pricing.gpu_els_pricer import price_els_gpu
    from src.solvers.gpu_adi_solver import check_gpu_available
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def check_system():
    """시스템 정보 확인"""
    print("\n" + "=" * 80)
    print(" " * 30 + "시스템 정보")
    print("=" * 80)

    # CPU 정보
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        print(f"CPU: {cpu_count}코어 ({cpu_count_logical} 논리 프로세서)")
    except:
        print("CPU: 정보 확인 불가")

    # GPU 정보
    if GPU_AVAILABLE:
        gpu_info = check_gpu_available()
        if gpu_info['gpu_available']:
            print(f"GPU: {gpu_info['gpu_name']}")
            print(f"GPU 메모리: {gpu_info['gpu_memory']}")
            print("✓ CUDA 가속 사용 가능")
        else:
            print("⚠️  GPU가 감지되지 않았습니다")
    else:
        print("⚠️  CuPy가 설치되지 않았습니다")
        print("   설치 방법:")
        print("   - CUDA 11.x: pip install cupy-cuda11x")
        print("   - CUDA 12.x: pip install cupy-cuda12x")

    print("=" * 80)


def benchmark_single(grid_size: int, use_gpu: bool = False):
    """
    단일 벤치마크 실행

    Args:
        grid_size: 그리드 크기 (N1 = N2 = grid_size)
        use_gpu: GPU 사용 여부

    Returns:
        (가격, 계산 시간)
    """
    product = create_sample_els()

    start = time.time()

    if use_gpu and GPU_AVAILABLE:
        result = price_els_gpu(
            product,
            N1=grid_size,
            N2=grid_size,
            Nt=grid_size * 2,
            verbose=False,
            use_gpu=True
        )
    else:
        result = price_els(
            product,
            N1=grid_size,
            N2=grid_size,
            Nt=grid_size * 2,
            verbose=False
        )

    elapsed = time.time() - start

    return result['price'], elapsed


def run_benchmark():
    """전체 벤치마크 실행"""
    print("\n" + "=" * 80)
    print(" " * 25 + "GPU vs CPU 성능 벤치마크")
    print("=" * 80)

    product = create_sample_els()
    print(f"\n상품: Step-Down ELS (3년, 6개월 단위)")
    print(f"기초자산: S1={product.S1_0}, S2={product.S2_0}")
    print(f"변동성: σ1={product.sigma1}, σ2={product.sigma2}, ρ={product.rho}")

    # 테스트할 그리드 크기들
    grid_sizes = [40, 60, 80, 100, 150, 200]

    print(f"\n{'=' * 80}")
    print(f"{'Grid':<12} {'CPU Time':<12} {'GPU Time':<12} {'Speedup':<12} {'Price':<12}")
    print(f"{'Size':<12} {'(sec)':<12} {'(sec)':<12} {'(x배)':<12} {'Diff':<12}")
    print("=" * 80)

    results = []

    for N in grid_sizes:
        print(f"{N}x{N:<7}", end="", flush=True)

        # CPU 벤치마크
        try:
            price_cpu, time_cpu = benchmark_single(N, use_gpu=False)
            print(f" {time_cpu:>10.2f}s", end="", flush=True)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        # GPU 벤치마크
        if GPU_AVAILABLE:
            try:
                price_gpu, time_gpu = benchmark_single(N, use_gpu=True)
                speedup = time_cpu / time_gpu
                price_diff = abs(price_cpu - price_gpu)

                print(f"  {time_gpu:>10.2f}s  {speedup:>10.1f}x  {price_diff:>10.4f}")

                results.append({
                    'grid_size': N,
                    'cpu_time': time_cpu,
                    'gpu_time': time_gpu,
                    'speedup': speedup,
                    'price_cpu': price_cpu,
                    'price_gpu': price_gpu
                })

            except Exception as e:
                print(f"  GPU ERROR: {e}")
        else:
            print(f"  N/A       N/A       N/A")

    print("=" * 80)

    # 요약
    if results and GPU_AVAILABLE:
        print("\n" + "=" * 80)
        print("요약")
        print("=" * 80)

        avg_speedup = np.mean([r['speedup'] for r in results])
        max_speedup = max([r['speedup'] for r in results])
        best_grid = max(results, key=lambda x: x['speedup'])

        print(f"평균 속도 향상: {avg_speedup:.1f}배")
        print(f"최대 속도 향상: {max_speedup:.1f}배 (Grid {best_grid['grid_size']}x{best_grid['grid_size']})")

        # 가격 정확도 체크
        max_price_diff = max([abs(r['price_cpu'] - r['price_gpu']) for r in results])
        print(f"최대 가격 차이: {max_price_diff:.6f} (수치 오차)")

        print("\n권장 사항:")
        if avg_speedup > 5:
            print(f"✓ GPU 사용 강력 권장! ({avg_speedup:.1f}배 빠름)")
        elif avg_speedup > 2:
            print(f"✓ GPU 사용 권장 ({avg_speedup:.1f}배 빠름)")
        else:
            print(f"⚠️  작은 그리드에서는 GPU 오버헤드로 인해 개선 효과 제한적")
            print(f"   큰 그리드(150x150 이상)에서 GPU 사용 권장")

        print("=" * 80)


def quick_comparison():
    """빠른 비교 (80x80 그리드)"""
    print("\n" + "=" * 80)
    print(" " * 30 + "빠른 비교 테스트")
    print("=" * 80)

    N = 80
    print(f"\nGrid: {N}x{N}, Time steps: {N*2}")

    product = create_sample_els()

    # CPU
    print("\n[CPU] 계산 중...", end="", flush=True)
    price_cpu, time_cpu = benchmark_single(N, use_gpu=False)
    print(f" 완료!")
    print(f"  시간: {time_cpu:.2f}초")
    print(f"  가격: {price_cpu:.4f}")

    # GPU
    if GPU_AVAILABLE:
        print("\n[GPU] 계산 중...", end="", flush=True)
        price_gpu, time_gpu = benchmark_single(N, use_gpu=True)
        print(f" 완료!")
        print(f"  시간: {time_gpu:.2f}초")
        print(f"  가격: {price_gpu:.4f}")

        speedup = time_cpu / time_gpu
        print(f"\n✓ 속도 향상: {speedup:.1f}배")
        print(f"  가격 차이: {abs(price_cpu - price_gpu):.6f}")

    print("=" * 80)


def main():
    """메인 실행"""
    check_system()

    if not GPU_AVAILABLE:
        print("\n" + "!" * 80)
        print("GPU 벤치마크를 실행하려면 CuPy를 설치하세요:")
        print("")
        print("  pip install cupy-cuda11x  # CUDA 11.x")
        print("  pip install cupy-cuda12x  # CUDA 12.x")
        print("")
        print("CPU 전용 벤치마크만 실행됩니다.")
        print("!" * 80)
        input("\n계속하려면 Enter를 누르세요...")

    print("\n벤치마크 옵션:")
    print("  1. 빠른 비교 (80x80 그리드)")
    print("  2. 전체 벤치마크 (여러 그리드 크기)")
    print("  3. 종료")

    choice = input("\n선택 (1-3): ").strip()

    if choice == '1':
        quick_comparison()
    elif choice == '2':
        run_benchmark()
    else:
        print("종료합니다.")


if __name__ == "__main__":
    main()
