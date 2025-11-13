"""
ELS 상품 FDM 가격 평가 엔진
"""

import numpy as np
from typing import Dict, Optional
from ..models.els_product import ELSProduct
from ..grid.grid_2d import Grid2D, create_adaptive_grid
from ..solvers.adi_solver import ADISolver


class ELSPricer:
    """Step-Down ELS FDM 프라이서"""

    def __init__(self, product: ELSProduct, grid: Grid2D):
        """
        Args:
            product: ELS 상품 정의
            grid: 2D 그리드
        """
        self.product = product
        self.grid = grid

        # FDM Solver 생성
        self.solver = ADISolver(
            grid=grid,
            r=product.r,
            q1=product.q1,
            q2=product.q2,
            sigma1=product.sigma1,
            sigma2=product.sigma2,
            rho=product.rho
        )

        # 낙인 추적 그리드
        self.ki_grid = np.zeros((grid.N1, grid.N2), dtype=bool)

    def price(self, verbose: bool = True) -> Dict:
        """
        ELS 가격 평가

        Returns:
            평가 결과 딕셔너리
        """
        if verbose:
            print("=" * 60)
            print("ELS FDM Pricing")
            print("=" * 60)
            print(self.product)
            print(self.grid)

        # 1. 만기 페이오프 설정
        V_T = self._initialize_terminal_payoff()

        # 2. FDM으로 역방향 풀기 (조기상환 포함)
        results = self.solver.solve_with_callbacks(
            V_T=V_T,
            observation_dates=self.product.observation_dates,
            redemption_callback=self._early_redemption_callback
        )

        # 3. 현재가(S1_0, S2_0)에서 가격 추출
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
            'product': self.product
        }

    def _initialize_terminal_payoff(self) -> np.ndarray:
        """
        만기(T) 페이오프 초기화

        만기에는 조기상환이 마지막 기회이므로:
        1. 조기상환 조건 만족 -> 원금 + 쿠폰
        2. 조기상환 실패 + 낙인 미발생 -> 원금 + 쿠폰
        3. 조기상환 실패 + 낙인 발생 -> min(원금, 원금 * performance)
        """
        N1, N2 = self.grid.N1, self.grid.N2
        V_T = np.zeros((N1, N2))

        S1_mesh = self.grid.S1_mesh
        S2_mesh = self.grid.S2_mesh

        # 전체 경로에서 낙인 발생 여부 체크 (간단히 만기 시점만 체크)
        # 실제로는 경로 추적 필요하지만, 여기서는 보수적으로 처리
        for i in range(N1):
            for j in range(N2):
                S1 = S1_mesh[i, j]
                S2 = S2_mesh[i, j]

                # 마지막 조기상환 기회 체크
                last_obs_idx = len(self.product.observation_dates) - 1
                is_redeemed, payoff = self.product.check_early_redemption(S1, S2, last_obs_idx)

                if is_redeemed:
                    V_T[i, j] = payoff
                else:
                    # 낙인 체크
                    ki_occurred = self.product.check_knock_in(S1, S2)

                    # 만기 페이오프
                    payoff_array = self.product.payoff_at_maturity(
                        np.array([S1]), np.array([S2]), ki_occurred
                    )
                    # numpy array에서 스칼라 추출
                    V_T[i, j] = float(np.squeeze(payoff_array))

        return V_T

    def _early_redemption_callback(self, V: np.ndarray,
                                    S1_mesh: np.ndarray,
                                    S2_mesh: np.ndarray,
                                    obs_idx: int) -> np.ndarray:
        """
        조기상환 콜백 함수

        현재 시점에서 조기상환 조건을 만족하면 즉시 상환

        Args:
            V: 현재 continuation value (N1 x N2)
            S1_mesh: S1 메쉬 그리드
            S2_mesh: S2 메쉬 그리드
            obs_idx: 관찰일 인덱스

        Returns:
            조정된 가치 그리드
        """
        V_adjusted = V.copy()
        N1, N2 = V.shape

        for i in range(N1):
            for j in range(N2):
                S1 = S1_mesh[i, j]
                S2 = S2_mesh[i, j]

                # 조기상환 조건 체크
                is_redeemed, payoff = self.product.check_early_redemption(S1, S2, obs_idx)

                if is_redeemed:
                    # 조기상환 조건 만족 시, continuation value와 비교하여 큰 값
                    # (하지만 ELS는 자동 상환이므로 무조건 payoff)
                    V_adjusted[i, j] = payoff

        return V_adjusted

    def compute_greeks(self, bump_size: float = 0.01) -> Dict:
        """
        Greeks 계산 (유한차분법)

        Args:
            bump_size: 범핑 크기 (1%)

        Returns:
            Greeks 딕셔너리
        """
        base_price = self.price(verbose=False)['price']

        # Delta (S1)
        product_bump_S1 = self._bump_product(S1_bump=bump_size)
        price_S1_up = ELSPricer(product_bump_S1, self.grid).price(verbose=False)['price']
        delta_S1 = (price_S1_up - base_price) / (self.product.S1_0 * bump_size)

        # Delta (S2)
        product_bump_S2 = self._bump_product(S2_bump=bump_size)
        price_S2_up = ELSPricer(product_bump_S2, self.grid).price(verbose=False)['price']
        delta_S2 = (price_S2_up - base_price) / (self.product.S2_0 * bump_size)

        # Vega (sigma1)
        product_bump_sigma1 = self._bump_product(sigma1_bump=0.01)  # 1% vol bump
        price_sigma1_up = ELSPricer(product_bump_sigma1, self.grid).price(verbose=False)['price']
        vega_S1 = price_sigma1_up - base_price

        # Vega (sigma2)
        product_bump_sigma2 = self._bump_product(sigma2_bump=0.01)
        price_sigma2_up = ELSPricer(product_bump_sigma2, self.grid).price(verbose=False)['price']
        vega_S2 = price_sigma2_up - base_price

        # Rho (correlation)
        product_bump_rho = self._bump_product(rho_bump=0.05)  # 5% correlation bump
        price_rho_up = ELSPricer(product_bump_rho, self.grid).price(verbose=False)['price']
        rho_sensitivity = (price_rho_up - base_price) / 0.05

        return {
            'delta_S1': delta_S1,
            'delta_S2': delta_S2,
            'vega_S1': vega_S1,
            'vega_S2': vega_S2,
            'rho_sensitivity': rho_sensitivity,
        }

    def _bump_product(self, S1_bump: float = 0, S2_bump: float = 0,
                      sigma1_bump: float = 0, sigma2_bump: float = 0,
                      rho_bump: float = 0) -> ELSProduct:
        """제품 파라미터 범핑"""
        import copy
        product_copy = copy.deepcopy(self.product)

        if S1_bump != 0:
            product_copy.S1_0 *= (1 + S1_bump)
        if S2_bump != 0:
            product_copy.S2_0 *= (1 + S2_bump)
        if sigma1_bump != 0:
            product_copy.sigma1 += sigma1_bump
        if sigma2_bump != 0:
            product_copy.sigma2 += sigma2_bump
        if rho_bump != 0:
            product_copy.rho = np.clip(product_copy.rho + rho_bump, -0.99, 0.99)

        return product_copy


def price_els(product: ELSProduct,
              N1: int = 100,
              N2: int = 100,
              Nt: int = 200,
              space_factor: float = 3.0,
              verbose: bool = True) -> Dict:
    """
    ELS 가격 평가 (간편 인터페이스)

    Args:
        product: ELS 상품
        N1: S1 방향 그리드 수
        N2: S2 방향 그리드 수
        Nt: 시간 스텝 수
        space_factor: 공간 범위 배율
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

    # 프라이서 생성 및 평가
    pricer = ELSPricer(product, grid)
    return pricer.price(verbose=verbose)
