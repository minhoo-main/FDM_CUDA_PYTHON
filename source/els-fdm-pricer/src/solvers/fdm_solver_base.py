"""
FDM Solver 기본 클래스 및 유틸리티
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Callable
from ..grid.grid_2d import Grid2D


class FDMSolver2D(ABC):
    """2D FDM Solver 추상 클래스"""

    def __init__(self, grid: Grid2D, r: float, q1: float, q2: float,
                 sigma1: float, sigma2: float, rho: float):
        """
        Args:
            grid: 2D 그리드
            r: 무위험이자율
            q1: 기초자산 1 배당률
            q2: 기초자산 2 배당률
            sigma1: 기초자산 1 변동성
            sigma2: 기초자산 2 변동성
            rho: 상관계수
        """
        self.grid = grid
        self.r = r
        self.q1 = q1
        self.q2 = q2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho = rho

        # 그리드 속성
        self.N1 = grid.N1
        self.N2 = grid.N2
        self.Nt = grid.Nt
        self.dt = grid.dt

    @abstractmethod
    def solve(self, V_T: np.ndarray,
              boundary_condition: Callable = None) -> np.ndarray:
        """
        PDE를 풀어서 t=0에서의 가격 계산

        Args:
            V_T: 만기 페이오프 (N1 x N2)
            boundary_condition: 경계 조건 함수

        Returns:
            V_0: t=0에서의 가격 (N1 x N2)
        """
        pass

    def apply_boundary_conditions(self, V: np.ndarray) -> np.ndarray:
        """
        경계 조건 적용

        Args:
            V: 현재 가격 그리드 (N1 x N2)

        Returns:
            경계 조건이 적용된 가격 그리드
        """
        V_new = V.copy()

        # S1 = 0 경계: V = 0 (주식 가격이 0이면 옵션 가치 없음)
        V_new[0, :] = 0.0

        # S2 = 0 경계: V = 0
        V_new[:, 0] = 0.0

        # S1 = S1_max 경계: 선형 외삽 또는 Neumann 조건
        # dV/dS1 = (V[-1] - V[-2]) / dS1 유지
        V_new[-1, :] = 2 * V_new[-2, :] - V_new[-3, :]

        # S2 = S2_max 경계
        V_new[:, -1] = 2 * V_new[:, -2] - V_new[:, -3]

        return V_new

    def get_coefficients(self, i: int, j: int) -> dict:
        """
        (i, j) 위치에서 FDM 계수 계산

        2D Black-Scholes PDE:
        dV/dt + 0.5*σ1²*S1²*d²V/dS1² + 0.5*σ2²*S2²*d²V/dS2²
              + ρ*σ1*σ2*S1*S2*d²V/dS1dS2
              + (r-q1)*S1*dV/dS1 + (r-q2)*S2*dV/dS2 - r*V = 0

        Returns:
            계수 딕셔너리
        """
        S1 = self.grid.S1[i]
        S2 = self.grid.S2[j]
        dS1 = self.grid.dS1
        dS2 = self.grid.dS2

        # S1 방향 계수
        a1 = 0.5 * self.sigma1**2 * S1**2 / dS1**2
        b1 = (self.r - self.q1) * S1 / (2 * dS1)

        # S2 방향 계수
        a2 = 0.5 * self.sigma2**2 * S2**2 / dS2**2
        b2 = (self.r - self.q2) * S2 / (2 * dS2)

        # 교차항 계수
        c_cross = 0.25 * self.rho * self.sigma1 * self.sigma2 * S1 * S2 / (dS1 * dS2)

        return {
            'a1': a1,  # d²V/dS1² 계수
            'b1': b1,  # dV/dS1 계수
            'a2': a2,  # d²V/dS2² 계수
            'b2': b2,  # dV/dS2 계수
            'c': c_cross,  # d²V/dS1dS2 계수
            'r': self.r
        }


def create_tridiagonal_matrix(N: int, alpha: np.ndarray, beta: np.ndarray,
                               gamma: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    삼중대각 행렬 생성

    Args:
        N: 행렬 크기
        alpha: 하부 대각선 (N-1,)
        beta: 주 대각선 (N,)
        gamma: 상부 대각선 (N-1,)

    Returns:
        (lower, diag, upper) 대각선들
    """
    lower = alpha.copy()
    diag = beta.copy()
    upper = gamma.copy()

    return lower, diag, upper


def solve_tridiagonal(lower: np.ndarray, diag: np.ndarray,
                      upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """
    삼중대각 선형 시스템 풀기 (Thomas 알고리즘)

    A * x = rhs

    Args:
        lower: 하부 대각선
        diag: 주 대각선
        upper: 상부 대각선
        rhs: 우변

    Returns:
        해 벡터 x
    """
    N = len(diag)
    c_prime = np.zeros(N - 1)
    d_prime = np.zeros(N)
    x = np.zeros(N)

    # Forward sweep
    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]

    for i in range(1, N - 1):
        denom = diag[i] - lower[i - 1] * c_prime[i - 1]
        c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom

    d_prime[N - 1] = (rhs[N - 1] - lower[N - 2] * d_prime[N - 2]) / \
                     (diag[N - 1] - lower[N - 2] * c_prime[N - 2])

    # Backward substitution
    x[N - 1] = d_prime[N - 1]
    for i in range(N - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x
