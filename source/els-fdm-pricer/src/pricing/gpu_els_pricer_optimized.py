"""
Optimized GPU-accelerated ELS Pricer

Phase 1 최적화 적용:
1. Batched tridiagonal solver (solver에서 처리)
2. Vectorized 조기상환 체크 - GPU에서 병렬 처리
3. Vectorized 만기 페이오프 - GPU에서 병렬 처리

예상 성능: 기존 GPU 대비 10-20배 향상
"""

import numpy as np
from typing import Dict, Optional

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from ..models.els_product import ELSProduct
from ..grid.grid_2d import Grid2D, create_adaptive_grid
from ..solvers.gpu_adi_solver_optimized import OptimizedGPUADISolver


class OptimizedGPUELSPricer:
    """
    Optimized GPU-accelerated Step-Down ELS Pricer

    주요 개선사항:
    1. Vectorized 만기 페이오프 초기화
       - 기존: 중첩 for loop (10,000번 반복)
       - 개선: GPU vectorized 연산 (1회)

    2. Vectorized 조기상환 체크
       - 기존: CPU에서 중첩 loop + GPU↔CPU 전송
       - 개선: GPU에서 vectorized 연산
    """

    def __init__(self, product: ELSProduct, grid: Grid2D, use_gpu: bool = True):
        self.product = product
        self.grid = grid

        # Optimized GPU FDM Solver 생성
        self.solver = OptimizedGPUADISolver(
            grid=grid,
            r=product.r,
            q1=product.q1,
            q2=product.q2,
            sigma1=product.sigma1,
            sigma2=product.sigma2,
            rho=product.rho,
            use_gpu=use_gpu
        )

        self.use_gpu = self.solver.use_gpu
        self.xp = cp if self.use_gpu else np

    def price(self, verbose: bool = True) -> Dict:
        """
        Optimized GPU ELS 가격 평가

        Returns:
            평가 결과 딕셔너리
        """
        if verbose:
            print("=" * 60)
            print("🚀 Optimized GPU ELS FDM Pricing" if self.use_gpu else "CPU ELS FDM Pricing")
            print("=" * 60)
            print(self.product)
            print(self.grid)

        # 1. 만기 페이오프 설정 (Vectorized!)
        V_T = self._initialize_terminal_payoff_vectorized()

        # 2. Optimized GPU로 FDM 풀기
        results = self.solver.solve_with_callbacks(
            V_T=V_T,
            observation_dates=self.product.observation_dates,
            redemption_callback=self._early_redemption_callback_vectorized
        )

        # 3. 현재가에서 가격 추출
        V_0 = results['V_0']
        price = self.grid.get_value_at_point(V_0, self.product.S1_0, self.product.S2_0)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"ELS Price: {price:.4f}")
            print(f"{'=' * 60}")

        return {
            'price': price,
            'V_0': V_0,
            'V_T': V_T,
            'snapshots': results.get('V_snapshots', {}),
            'redemption_flags': results.get('redemption_flags', {}),
            'grid': self.grid,
            'product': self.product,
            'use_gpu': self.use_gpu
        }

    def _initialize_terminal_payoff_vectorized(self) -> np.ndarray:
        """
        Vectorized 만기 페이오프 초기화 (핵심 최적화!)

        기존: 중첩 for loop (N1 × N2 반복)
        개선: GPU vectorized 연산 (병렬)

        예상 성능: 10배 향상
        """
        if not self.use_gpu:
            # CPU fallback
            return self._initialize_terminal_payoff_sequential()

        xp = self.xp

        # GPU 메모리에서 직접 계산
        S1_mesh = xp.array(self.grid.S1_mesh)
        S2_mesh = xp.array(self.grid.S2_mesh)

        # 1. Performance 계산 (vectorized)
        perf1 = S1_mesh / self.product.S1_0
        perf2 = S2_mesh / self.product.S2_0

        # 2. Worst-of 계산
        if self.product.worst_of:
            worst_perf = xp.minimum(perf1, perf2)
        else:
            worst_perf = xp.maximum(perf1, perf2)

        # 3. 마지막 조기상환 체크
        last_obs_idx = len(self.product.observation_dates) - 1
        redemption_barrier = self.product.redemption_barriers[last_obs_idx]
        coupon = self.product.coupons[last_obs_idx]

        is_redeemed = worst_perf >= redemption_barrier
        V_redeemed = self.product.principal + coupon

        # 4. Knock-In 체크 (전 구간)
        ki_barrier = self.product.ki_barrier
        ki_occurred = worst_perf < ki_barrier

        # 5. 만기 페이오프 계산
        # KI 발생: principal × min(1, worst_perf)
        # KI 미발생: principal + final coupon
        V_ki = self.product.principal * xp.minimum(1.0, worst_perf)
        V_no_ki = self.product.principal + coupon

        # 6. 조건부 페이오프 (nested where)
        V_T = xp.where(
            is_redeemed,
            V_redeemed,
            xp.where(ki_occurred, V_ki, V_no_ki)
        )

        # CPU로 반환
        return cp.asnumpy(V_T) if self.use_gpu else V_T

    def _initialize_terminal_payoff_sequential(self) -> np.ndarray:
        """Sequential fallback (기존 방식)"""
        N1, N2 = self.grid.N1, self.grid.N2
        V_T = np.zeros((N1, N2))

        S1_mesh = self.grid.S1_mesh
        S2_mesh = self.grid.S2_mesh

        for i in range(N1):
            for j in range(N2):
                S1 = S1_mesh[i, j]
                S2 = S2_mesh[i, j]

                last_obs_idx = len(self.product.observation_dates) - 1
                is_redeemed, payoff = self.product.check_early_redemption(S1, S2, last_obs_idx)

                if is_redeemed:
                    V_T[i, j] = payoff
                else:
                    ki_occurred = self.product.check_knock_in(S1, S2)
                    payoff_array = self.product.payoff_at_maturity(
                        np.array([S1]), np.array([S2]), ki_occurred
                    )
                    V_T[i, j] = float(np.squeeze(payoff_array))

        return V_T

    def _early_redemption_callback_vectorized(self, V: np.ndarray,
                                               S1_mesh: np.ndarray,
                                               S2_mesh: np.ndarray,
                                               obs_idx: int) -> np.ndarray:
        """
        Vectorized 조기상환 콜백 (핵심 최적화!)

        기존: CPU 중첩 loop (N1 × N2 반복)
        개선: GPU vectorized 연산

        예상 성능: 50배 향상
        """
        if not self.use_gpu:
            # CPU fallback
            return self._early_redemption_callback_sequential(V, S1_mesh, S2_mesh, obs_idx)

        xp = self.xp

        # GPU로 전송
        V_gpu = xp.array(V)
        S1_mesh_gpu = xp.array(S1_mesh)
        S2_mesh_gpu = xp.array(S2_mesh)

        # 1. Performance 계산 (vectorized)
        perf1 = S1_mesh_gpu / self.product.S1_0
        perf2 = S2_mesh_gpu / self.product.S2_0

        # 2. Worst-of 계산
        if self.product.worst_of:
            worst_perf = xp.minimum(perf1, perf2)
        else:
            worst_perf = xp.maximum(perf1, perf2)

        # 3. 조기상환 조건 체크 (vectorized)
        redemption_barrier = self.product.redemption_barriers[obs_idx]
        is_redeemed = worst_perf >= redemption_barrier

        # 4. 조기상환 페이오프
        coupon = self.product.coupons[obs_idx]
        redemption_value = self.product.principal + coupon

        # 5. 조건부 업데이트 (vectorized)
        V_new = xp.where(is_redeemed, redemption_value, V_gpu)

        # CPU로 반환
        return cp.asnumpy(V_new) if self.use_gpu else V_new

    def _early_redemption_callback_sequential(self, V, S1_mesh, S2_mesh, obs_idx):
        """Sequential fallback (기존 방식)"""
        V_adjusted = V.copy()
        N1, N2 = V.shape

        for i in range(N1):
            for j in range(N2):
                S1 = S1_mesh[i, j]
                S2 = S2_mesh[i, j]

                is_redeemed, payoff = self.product.check_early_redemption(S1, S2, obs_idx)

                if is_redeemed:
                    V_adjusted[i, j] = payoff

        return V_adjusted


def price_els_optimized(product: ELSProduct,
                        N1: int = 100,
                        N2: int = 100,
                        Nt: int = 200,
                        space_factor: float = 3.0,
                        use_gpu: bool = True,
                        verbose: bool = True) -> Dict:
    """
    Optimized GPU로 ELS 가격 평가 (간편 인터페이스)

    Args:
        product: ELS 상품
        N1: S1 방향 그리드 수
        N2: S2 방향 그리드 수
        Nt: 시간 스텝 수
        space_factor: 공간 범위 배율
        use_gpu: GPU 사용 여부
        verbose: 상세 출력 여부

    Returns:
        평가 결과
    """
    grid = create_adaptive_grid(
        S1_0=product.S1_0,
        S2_0=product.S2_0,
        T=product.maturity,
        N1=N1,
        N2=N2,
        Nt=Nt,
        space_factor=space_factor
    )

    pricer = OptimizedGPUELSPricer(product, grid, use_gpu=use_gpu)
    return pricer.price(verbose=verbose)
