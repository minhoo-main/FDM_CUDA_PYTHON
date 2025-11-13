"""
ADI (Alternating Direction Implicit) Solver

2D 문제를 효율적으로 풀기 위한 ADI 방법:
- 각 시간 스텝을 2개의 half-step으로 분할
- 첫 번째 half-step: S1 방향으로 implicit, S2 방향으로 explicit
- 두 번째 half-step: S2 방향으로 implicit, S1 방향으로 explicit
- 각 half-step에서 삼중대각 행렬만 풀면 되므로 효율적
"""

import numpy as np
from typing import Callable, Optional
from .fdm_solver_base import FDMSolver2D, solve_tridiagonal
from ..grid.grid_2d import Grid2D


class ADISolver(FDMSolver2D):
    """ADI (Alternating Direction Implicit) Solver"""

    def __init__(self, grid: Grid2D, r: float, q1: float, q2: float,
                 sigma1: float, sigma2: float, rho: float):
        super().__init__(grid, r, q1, q2, sigma1, sigma2, rho)

        # ADI 계수 미리 계산
        self._precompute_coefficients()

    def _precompute_coefficients(self):
        """ADI 계수 미리 계산"""
        N1, N2 = self.N1, self.N2
        dS1, dS2 = self.grid.dS1, self.grid.dS2
        dt = self.dt

        # S1 방향 계수 행렬 (각 j에 대해 동일)
        self.alpha1 = np.zeros(N1 - 1)
        self.beta1 = np.zeros(N1)
        self.gamma1 = np.zeros(N1 - 1)

        # S2 방향 계수 행렬 (각 i에 대해 동일)
        self.alpha2 = np.zeros(N2 - 1)
        self.beta2 = np.zeros(N2)
        self.gamma2 = np.zeros(N2 - 1)

        # S1 방향 (i 인덱스)
        for i in range(1, N1 - 1):
            S1 = self.grid.S1[i]

            a1 = 0.5 * self.sigma1**2 * S1**2 / dS1**2
            b1 = (self.r - self.q1) * S1 / (2 * dS1)

            # 삼중대각 계수
            self.alpha1[i - 1] = -0.5 * dt * (a1 - b1)  # V[i-1]
            self.beta1[i] = 1.0 + dt * (a1 + 0.5 * self.r)  # V[i]
            self.gamma1[i] = -0.5 * dt * (a1 + b1)  # V[i+1]

        # 경계 조건
        self.beta1[0] = 1.0
        self.beta1[-1] = 1.0

        # S2 방향 (j 인덱스)
        for j in range(1, N2 - 1):
            S2 = self.grid.S2[j]

            a2 = 0.5 * self.sigma2**2 * S2**2 / dS2**2
            b2 = (self.r - self.q2) * S2 / (2 * dS2)

            # 삼중대각 계수
            self.alpha2[j - 1] = -0.5 * dt * (a2 - b2)
            self.beta2[j] = 1.0 + dt * (a2 + 0.5 * self.r)
            self.gamma2[j] = -0.5 * dt * (a2 + b2)

        # 경계 조건
        self.beta2[0] = 1.0
        self.beta2[-1] = 1.0

    def solve(self, V_T: np.ndarray,
              early_exercise_callback: Optional[Callable] = None) -> np.ndarray:
        """
        ADI 방법으로 PDE 풀기

        Args:
            V_T: 만기 페이오프 (N1 x N2)
            early_exercise_callback: 조기상환 체크 함수
                callback(V, t_idx) -> V_adjusted

        Returns:
            V_0: t=0에서의 가격 (N1 x N2)
        """
        V = V_T.copy()
        N1, N2 = self.N1, self.N2
        dS1, dS2 = self.grid.dS1, self.grid.dS2

        # 시간 역방향 진행 (T -> 0)
        for n in range(self.Nt - 1, -1, -1):
            t = self.grid.t[n]

            # Half-step 1: S1 방향 implicit
            V_half = self._solve_S1_direction(V)

            # Half-step 2: S2 방향 implicit
            V = self._solve_S2_direction(V_half)

            # 경계 조건 적용
            V = self.apply_boundary_conditions(V)

            # 조기상환 체크 (콜백이 있으면)
            if early_exercise_callback is not None:
                V = early_exercise_callback(V, n, t)

        return V

    def _solve_S1_direction(self, V: np.ndarray) -> np.ndarray:
        """
        S1 방향으로 implicit solve

        각 j에 대해 독립적으로 삼중대각 시스템 풀기
        """
        N1, N2 = self.N1, self.N2
        V_new = V.copy()

        # 각 S2 슬라이스에 대해 S1 방향으로 풀기
        for j in range(N2):
            # RHS 구성 (explicit part)
            rhs = V[:, j].copy()

            # 경계 조건
            rhs[0] = 0.0  # S1 = 0
            rhs[-1] = V[-1, j]  # S1 = S1_max

            # 삼중대각 시스템 풀기
            V_new[:, j] = solve_tridiagonal(self.alpha1, self.beta1, self.gamma1, rhs)

        return V_new

    def _solve_S2_direction(self, V: np.ndarray) -> np.ndarray:
        """
        S2 방향으로 implicit solve

        각 i에 대해 독립적으로 삼중대각 시스템 풀기
        """
        N1, N2 = self.N1, self.N2
        V_new = V.copy()

        # 각 S1 슬라이스에 대해 S2 방향으로 풀기
        for i in range(N1):
            # RHS 구성
            rhs = V[i, :].copy()

            # 경계 조건
            rhs[0] = 0.0  # S2 = 0
            rhs[-1] = V[i, -1]  # S2 = S2_max

            # 삼중대각 시스템 풀기
            V_new[i, :] = solve_tridiagonal(self.alpha2, self.beta2, self.gamma2, rhs)

        return V_new

    def solve_with_callbacks(self, V_T: np.ndarray,
                             observation_dates: list,
                             redemption_callback: Callable) -> dict:
        """
        조기상환 날짜에서 콜백 함수를 호출하며 풀기

        Args:
            V_T: 만기 페이오프
            observation_dates: 조기상환 평가일 리스트 (년 단위)
            redemption_callback: 조기상환 체크 함수
                callback(V, S1_mesh, S2_mesh, obs_idx) -> V_adjusted

        Returns:
            결과 딕셔너리 (V_0, 중간 결과 등)
        """
        V = V_T.copy()
        N1, N2 = self.N1, self.N2

        # 관찰일을 시간 인덱스로 변환
        obs_time_indices = []
        for obs_date in observation_dates:
            idx = int(np.argmin(np.abs(self.grid.t - obs_date)))
            obs_time_indices.append(idx)

        obs_idx = len(observation_dates) - 1  # 역방향이므로 마지막부터

        # 중간 결과 저장
        results = {
            'V_0': None,
            'V_snapshots': {},
            'redemption_flags': {}
        }

        # 시간 역방향 진행
        for n in range(self.Nt - 1, -1, -1):
            t = self.grid.t[n]

            # ADI 스텝
            V_half = self._solve_S1_direction(V)
            V = self._solve_S2_direction(V_half)
            V = self.apply_boundary_conditions(V)

            # 조기상환 체크
            if n in obs_time_indices and obs_idx >= 0:
                V_before = V.copy()
                V = redemption_callback(V, self.grid.S1_mesh, self.grid.S2_mesh, obs_idx)

                # 조기상환 플래그 저장
                results['redemption_flags'][t] = (V != V_before)
                obs_idx -= 1

            # 스냅샷 저장
            if n % (self.Nt // 10) == 0:  # 10개 정도만 저장
                results['V_snapshots'][t] = V.copy()

        results['V_0'] = V
        return results
