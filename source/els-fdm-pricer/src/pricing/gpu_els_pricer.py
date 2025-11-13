"""
GPU-accelerated ELS Pricer

CUDA를 이용한 고속 ELS 가격 평가
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
from ..solvers.gpu_adi_solver import GPUADISolver


class GPUELSPricer:
    """GPU-accelerated Step-Down ELS Pricer"""

    def __init__(self, product: ELSProduct, grid: Grid2D, use_gpu: bool = True):
        """
        Args:
            product: ELS 상품
            grid: 2D 그리드
            use_gpu: GPU 사용 여부 (False면 CPU 사용)
        """
        self.product = product
        self.grid = grid

        # GPU FDM Solver 생성
        self.solver = GPUADISolver(
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

    def price(self, verbose: bool = True) -> Dict:
        """
        GPU로 ELS 가격 평가

        Returns:
            평가 결과 딕셔너리
        """
        if verbose:
            print("=" * 60)
            print("GPU-Accelerated ELS FDM Pricing" if self.use_gpu else "CPU ELS FDM Pricing")
            print("=" * 60)
            print(self.product)
            print(self.grid)

        # 1. 만기 페이오프 설정
        V_T = self._initialize_terminal_payoff()

        # 2. GPU로 FDM 풀기
        results = self.solver.solve_with_callbacks(
            V_T=V_T,
            observation_dates=self.product.observation_dates,
            redemption_callback=self._early_redemption_callback
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

    def _initialize_terminal_payoff(self) -> np.ndarray:
        """만기 페이오프 초기화 (CPU에서)"""
        N1, N2 = self.grid.N1, self.grid.N2
        V_T = np.zeros((N1, N2))

        S1_mesh = self.grid.S1_mesh
        S2_mesh = self.grid.S2_mesh

        for i in range(N1):
            for j in range(N2):
                S1 = S1_mesh[i, j]
                S2 = S2_mesh[i, j]

                # 마지막 조기상환 체크
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

    def _early_redemption_callback(self, V: np.ndarray,
                                    S1_mesh: np.ndarray,
                                    S2_mesh: np.ndarray,
                                    obs_idx: int) -> np.ndarray:
        """조기상환 콜백 (CPU에서 실행)"""
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


def price_els_gpu(product: ELSProduct,
                  N1: int = 100,
                  N2: int = 100,
                  Nt: int = 200,
                  space_factor: float = 3.0,
                  use_gpu: bool = True,
                  verbose: bool = True) -> Dict:
    """
    GPU로 ELS 가격 평가 (간편 인터페이스)

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
    # 그리드 생성
    grid = create_adaptive_grid(
        S1_0=product.S1_0,
        S2_0=product.S2_0,
        T=product.maturity,
        N1=N1,
        N2=N2,
        Nt=Nt,
        space_factor=space_factor
    )

    # GPU 프라이서 생성 및 평가
    pricer = GPUELSPricer(product, grid, use_gpu=use_gpu)
    return pricer.price(verbose=verbose)
