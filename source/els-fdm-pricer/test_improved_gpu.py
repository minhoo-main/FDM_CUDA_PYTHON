#!/usr/bin/env python3
"""
개선된 GPU Solver 테스트

작은 그리드로 먼저 검증
"""

import sys
import time
import numpy as np

sys.path.insert(0, '/home/minhoo/els-fdm-pricer')

from src.models.els_product import create_sample_els
from src.pricing.els_pricer import price_els

try:
    import cupy as cp
    from src.solvers.gpu_adi_solver_improved import ImprovedGPUADISolver
    from src.grid.grid_2d import create_adaptive_grid
    GPU_AVAILABLE = True
except ImportError as e:
    GPU_AVAILABLE = False
    print(f"GPU not available: {e}")
    print("Install CuPy: pip install cupy-cuda12x")
    sys.exit(1)


def test_small_grid():
    """작은 그리드로 테스트"""

    print("=" * 80)
    print("개선된 GPU Solver 테스트 (작은 그리드)")
    print("=" * 80)
    print()

    product = create_sample_els()

    # 작은 그리드부터 시작
    test_configs = [
        (30, 30, 60, "매우 작음"),
        (50, 50, 100, "작음"),
        (80, 80, 160, "중간"),
    ]

    results = []

    for N1, N2, Nt, label in test_configs:
        print(f"\n{'=' * 80}")
        print(f"테스트: {N1}×{N2}×{Nt} ({label})")
        print(f"{'=' * 80}")

        # CPU 기준
        print(f"\n[CPU] 계산 중...", end="", flush=True)
        start = time.time()
        result_cpu = price_els(product, N1=N1, N2=N2, Nt=Nt, verbose=False)
        time_cpu = time.time() - start
        price_cpu = result_cpu['price']
        print(f" {time_cpu:.3f}초")
        print(f"  가격: {price_cpu:.4f}")

        # 개선된 GPU
        print(f"\n[GPU Improved] 계산 중...", end="", flush=True)

        # Grid 생성
        grid = create_adaptive_grid(
            product.S1_0, product.S2_0, product.maturity,
            N1, N2, Nt, space_factor=3.0
        )

        # 개선된 Solver 사용
        solver = ImprovedGPUADISolver(
            grid, product.r, product.q1, product.q2,
            product.sigma1, product.sigma2, product.rho,
            use_gpu=True
        )

        # 만기 페이오프
        V_T = np.zeros((N1, N2))
        for i in range(N1):
            for j in range(N2):
                S1 = grid.S1_mesh[i, j]
                S2 = grid.S2_mesh[i, j]

                # 만기 페이오프
                perf1 = S1 / product.S1_0
                perf2 = S2 / product.S2_0
                worst_perf = min(perf1, perf2) if product.worst_of else max(perf1, perf2)

                # 조기상환 체크 (만기)
                last_barrier = product.redemption_barriers[-1]
                if worst_perf >= last_barrier:
                    V_T[i, j] = product.principal + product.coupons[-1]
                else:
                    # Knock-in 체크
                    if worst_perf < product.ki_barrier:
                        V_T[i, j] = product.principal * min(1.0, worst_perf)
                    else:
                        V_T[i, j] = product.principal + product.coupons[-1]

        # Solve
        start = time.time()
        V_0 = solver.solve(V_T, early_exercise_callback=None)  # 일단 조기상환 없이
        time_gpu = time.time() - start

        # 가격 (초기 지점)
        i_mid = N1 // 2
        j_mid = N2 // 2
        price_gpu = V_0[i_mid, j_mid]

        print(f" {time_gpu:.3f}초")
        print(f"  가격: {price_gpu:.4f}")

        # 결과 비교
        speedup = time_cpu / time_gpu if time_gpu > 0 else 0
        price_diff = abs(price_cpu - price_gpu)

        print(f"\n비교:")
        print(f"  속도 향상: {speedup:.1f}배")
        print(f"  가격 차이: {price_diff:.4f} ({price_diff/price_cpu*100:.2f}%)")

        results.append({
            'label': label,
            'N1': N1, 'N2': N2, 'Nt': Nt,
            'time_cpu': time_cpu,
            'time_gpu': time_gpu,
            'speedup': speedup,
            'price_cpu': price_cpu,
            'price_gpu': price_gpu
        })

        if speedup > 1:
            print(f"  ✓ GPU가 빠름!")
        else:
            print(f"  ⚠️ CPU가 빠름 (최적화 더 필요)")

    # 요약
    print(f"\n{'=' * 80}")
    print("테스트 요약")
    print(f"{'=' * 80}")
    print()
    print(f"{'크기':<15} {'CPU':<10} {'GPU':<10} {'가속비':<10} {'상태':<10}")
    print("-" * 80)

    for r in results:
        status = "✓ 빠름" if r['speedup'] > 1 else "⚠️ 느림"
        print(f"{r['label']:<15} {r['time_cpu']:>8.3f}s {r['time_gpu']:>8.3f}s {r['speedup']:>8.1f}x  {status}")

    print()

    # 평균 가속비
    avg_speedup = np.mean([r['speedup'] for r in results])
    print(f"평균 가속비: {avg_speedup:.1f}배")

    if avg_speedup > 5:
        print("\n🚀 개선 성공! GPU가 상당히 빠름!")
    elif avg_speedup > 1:
        print("\n✓ GPU가 빠름. 추가 최적화 가능.")
    else:
        print("\n⚠️ GPU가 여전히 느림. 디버깅 필요.")

    return results


if __name__ == "__main__":
    if not GPU_AVAILABLE:
        print("GPU/CuPy not available!")
        sys.exit(1)

    results = test_small_grid()
