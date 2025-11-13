#!/usr/bin/env python3
"""
GPU 성능 프로파일링

병목 구간을 정확히 찾아서 개선하기
"""

import sys
import time
import numpy as np

sys.path.insert(0, '/home/minhoo/els-fdm-pricer')

from src.models.els_product import create_sample_els
from src.pricing.els_pricer import price_els

try:
    import cupy as cp
    from src.pricing.gpu_els_pricer import GPUELSPricer
    from src.solvers.gpu_adi_solver import GPUADISolver
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not installed. Install with: pip install cupy-cuda12x")
    sys.exit(1)


def profile_gpu_components():
    """GPU 각 컴포넌트별 성능 측정"""

    print("=" * 80)
    print("GPU 성능 프로파일링")
    print("=" * 80)
    print()

    # 작은 그리드로 시작
    N1, N2, Nt = 50, 50, 100

    product = create_sample_els()

    print(f"테스트 그리드: {N1}×{N2}×{Nt}")
    print()

    # GPU Pricer 생성
    from src.grid.grid_2d import create_adaptive_grid
    grid = create_adaptive_grid(
        product.S1_0, product.S2_0, product.maturity,
        N1, N2, Nt, space_factor=3.0
    )

    pricer = GPUELSPricer(product, grid, use_gpu=True)
    solver = pricer.solver

    print("=" * 80)
    print("1. 데이터 전송 테스트")
    print("=" * 80)

    # CPU -> GPU 전송
    V_cpu = np.random.rand(N1, N2)

    start = time.time()
    for _ in range(100):
        V_gpu = cp.array(V_cpu)
    elapsed_h2d = (time.time() - start) / 100

    # GPU -> CPU 전송
    start = time.time()
    for _ in range(100):
        V_back = cp.asnumpy(V_gpu)
    elapsed_d2h = (time.time() - start) / 100

    print(f"CPU → GPU (100회 평균): {elapsed_h2d*1000:.3f}ms")
    print(f"GPU → CPU (100회 평균): {elapsed_d2h*1000:.3f}ms")
    print(f"왕복 시간: {(elapsed_h2d + elapsed_d2h)*1000:.3f}ms")
    print()

    print("=" * 80)
    print("2. Thomas 알고리즘 테스트")
    print("=" * 80)

    # 삼중대각 시스템 준비
    lower = cp.random.rand(N1-1)
    diag = cp.random.rand(N1) + 2.0  # 대각 우세
    upper = cp.random.rand(N1-1)
    rhs = cp.random.rand(N1)

    # 단일 Thomas 알고리즘
    start = time.time()
    for _ in range(100):
        x = solver._solve_tridiagonal_gpu(lower, diag, upper, rhs)
        cp.cuda.Stream.null.synchronize()  # GPU 완료 대기
    elapsed_thomas = (time.time() - start) / 100

    print(f"단일 Thomas solve (크기 {N1}): {elapsed_thomas*1000:.3f}ms")
    print()

    print("=" * 80)
    print("3. S1 방향 solve 테스트 (순차 루프)")
    print("=" * 80)

    V = cp.random.rand(N1, N2)

    start = time.time()
    V_new = solver._solve_S1_direction_gpu(V)
    cp.cuda.Stream.null.synchronize()
    elapsed_s1 = time.time() - start

    print(f"S1 방향 solve ({N2}개 시스템 순차): {elapsed_s1*1000:.1f}ms")
    print(f"  - 단일 시스템당: {elapsed_s1/N2*1000:.3f}ms")
    print(f"  - 예상 (병렬화 시): {elapsed_thomas*1000:.3f}ms (이론적 {N2}배 향상)")
    print()

    print("=" * 80)
    print("4. 경계 조건 적용")
    print("=" * 80)

    start = time.time()
    for _ in range(100):
        V_bc = solver._apply_boundary_conditions_gpu(V)
        cp.cuda.Stream.null.synchronize()
    elapsed_bc = (time.time() - start) / 100

    print(f"경계 조건 (100회 평균): {elapsed_bc*1000:.3f}ms")
    print()

    print("=" * 80)
    print("5. 전체 타임스텝")
    print("=" * 80)

    V_init = cp.random.rand(N1, N2)

    start = time.time()
    V_half = solver._solve_S1_direction_gpu(V_init)
    V_full = solver._solve_S2_direction_gpu(V_half)
    V_final = solver._apply_boundary_conditions_gpu(V_full)
    cp.cuda.Stream.null.synchronize()
    elapsed_timestep = time.time() - start

    print(f"단일 타임스텝: {elapsed_timestep*1000:.1f}ms")
    print(f"  - S1 solve: ~{elapsed_s1*1000:.1f}ms")
    print(f"  - S2 solve: ~{elapsed_s1*1000:.1f}ms")
    print(f"  - 경계조건: ~{elapsed_bc*1000:.3f}ms")
    print()

    print("=" * 80)
    print("병목 분석")
    print("=" * 80)
    print()

    total_time = elapsed_timestep * Nt

    print(f"전체 예상 시간 ({Nt} 타임스텝): {total_time:.2f}초")
    print()
    print("시간 분해:")
    print(f"  S1+S2 solve: {elapsed_s1*2*Nt:.2f}초 ({elapsed_s1*2*Nt/total_time*100:.1f}%)")
    print(f"  경계조건:    {elapsed_bc*Nt:.2f}초 ({elapsed_bc*Nt/total_time*100:.1f}%)")
    print()

    print("🐌 병목:")
    print(f"  1. Python for loop로 {N2}개 시스템을 순차 실행")
    print(f"     현재: {elapsed_s1*1000:.1f}ms")
    print(f"     이상: {elapsed_thomas*1000:.3f}ms (batched)")
    print(f"     잠재적 향상: {elapsed_s1/elapsed_thomas:.0f}배")
    print()

    # CPU와 비교
    print("=" * 80)
    print("CPU vs GPU 비교")
    print("=" * 80)
    print()

    print("[CPU] 계산 중...", end="", flush=True)
    start = time.time()
    result_cpu = price_els(product, N1=N1, N2=N2, Nt=Nt, verbose=False)
    time_cpu = time.time() - start
    print(f" {time_cpu:.2f}초")

    print("[GPU] 계산 중...", end="", flush=True)
    start = time.time()
    result_gpu = pricer.price(verbose=False)
    time_gpu = time.time() - start
    print(f" {time_gpu:.2f}초")

    print()
    print(f"CPU 시간: {time_cpu:.2f}초")
    print(f"GPU 시간: {time_gpu:.2f}초")

    if time_cpu < time_gpu:
        print(f"⚠️  CPU가 {time_gpu/time_cpu:.1f}배 빠름! GPU 최적화 필요!")
    else:
        print(f"✓ GPU가 {time_cpu/time_gpu:.1f}배 빠름")

    print()
    print("=" * 80)
    print("개선 방안")
    print("=" * 80)
    print()
    print("1. Batched tridiagonal solver 구현")
    print("   - 현재: Python for loop로 순차 실행")
    print("   - 개선: GPU에서 병렬로 실행")
    print(f"   - 예상 향상: ~{elapsed_s1/elapsed_thomas:.0f}배")
    print()
    print("2. 조기상환 체크 GPU vectorize")
    print("   - 현재: CPU로 전송 후 처리")
    print("   - 개선: GPU에서 직접 처리")
    print("   - 예상 향상: CPU↔GPU 전송 제거")
    print()
    print("3. 커널 퓨전")
    print("   - 현재: 여러 작은 커널 호출")
    print("   - 개선: 하나의 큰 커널로 합치기")
    print("   - 예상 향상: launch 오버헤드 감소")
    print()


if __name__ == "__main__":
    if not GPU_AVAILABLE:
        print("GPU/CuPy not available!")
        sys.exit(1)

    profile_gpu_components()
