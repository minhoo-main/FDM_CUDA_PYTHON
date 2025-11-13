"""
2D 그리드 생성 및 관리
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Grid2D:
    """2차원 균일 그리드"""

    # S1 방향 (기초자산 1)
    S1_min: float
    S1_max: float
    N1: int  # S1 방향 그리드 포인트 수

    # S2 방향 (기초자산 2)
    S2_min: float
    S2_max: float
    N2: int  # S2 방향 그리드 포인트 수

    # 시간 방향
    T: float  # 만기
    Nt: int  # 시간 스텝 수

    def __post_init__(self):
        """그리드 포인트 생성"""
        # 공간 그리드
        self.S1 = np.linspace(self.S1_min, self.S1_max, self.N1)
        self.S2 = np.linspace(self.S2_min, self.S2_max, self.N2)

        # 메쉬 그리드 (2D)
        self.S1_mesh, self.S2_mesh = np.meshgrid(self.S1, self.S2, indexing='ij')

        # 공간 스텝
        self.dS1 = (self.S1_max - self.S1_min) / (self.N1 - 1)
        self.dS2 = (self.S2_max - self.S2_min) / (self.N2 - 1)

        # 시간 그리드 (역방향: T -> 0)
        self.dt = self.T / self.Nt
        self.t = np.linspace(0, self.T, self.Nt + 1)

    def get_index(self, S1: float, S2: float) -> Tuple[int, int]:
        """
        주어진 (S1, S2)에 가장 가까운 그리드 인덱스 반환

        Args:
            S1: 기초자산 1 가격
            S2: 기초자산 2 가격

        Returns:
            (i, j) 인덱스
        """
        i = int(np.argmin(np.abs(self.S1 - S1)))
        j = int(np.argmin(np.abs(self.S2 - S2)))
        return i, j

    def get_value_at_point(self, V: np.ndarray, S1: float, S2: float) -> float:
        """
        선형 보간하여 임의 점에서 값 추정

        Args:
            V: 그리드 상의 값 (N1 x N2)
            S1: 기초자산 1 가격
            S2: 기초자산 2 가격

        Returns:
            보간된 값
        """
        from scipy.interpolate import RectBivariateSpline

        # 2D 스플라인 보간
        interp = RectBivariateSpline(self.S1, self.S2, V, kx=1, ky=1)
        return float(interp(S1, S2))

    def __repr__(self):
        return f"""
Grid2D Configuration
====================
S1: [{self.S1_min:.1f}, {self.S1_max:.1f}] with {self.N1} points (dS1={self.dS1:.3f})
S2: [{self.S2_min:.1f}, {self.S2_max:.1f}] with {self.N2} points (dS2={self.dS2:.3f})
Time: [0, {self.T:.1f}] with {self.Nt} steps (dt={self.dt:.4f})
Total grid points: {self.N1 * self.N2:,}
"""


def create_adaptive_grid(S1_0: float, S2_0: float,
                         T: float,
                         N1: int = 100,
                         N2: int = 100,
                         Nt: int = 200,
                         space_factor: float = 3.0) -> Grid2D:
    """
    기초자산 초기가 기준 적응형 그리드 생성

    Args:
        S1_0: 기초자산 1 초기가
        S2_0: 기초자산 2 초기가
        T: 만기
        N1: S1 방향 그리드 포인트 수
        N2: S2 방향 그리드 포인트 수
        Nt: 시간 스텝 수
        space_factor: 공간 범위 배율 (초기가의 몇 배까지)

    Returns:
        Grid2D 객체
    """
    # 넓은 범위 설정 (0부터 초기가의 space_factor 배까지)
    S1_min = 0.0
    S1_max = S1_0 * space_factor

    S2_min = 0.0
    S2_max = S2_0 * space_factor

    return Grid2D(
        S1_min=S1_min,
        S1_max=S1_max,
        N1=N1,
        S2_min=S2_min,
        S2_max=S2_max,
        N2=N2,
        T=T,
        Nt=Nt
    )


def check_stability(grid: Grid2D, sigma1: float, sigma2: float, r: float) -> dict:
    """
    FDM 안정성 조건 체크

    Args:
        grid: 그리드 객체
        sigma1: 기초자산 1 변동성
        sigma2: 기초자산 2 변동성
        r: 무위험이자율

    Returns:
        안정성 정보 딕셔너리
    """
    # Explicit 방법의 안정성 조건
    # dt <= min(dS1^2/(sigma1^2 * S1_max^2), dS2^2/(sigma2^2 * S2_max^2))

    dt_max_S1 = grid.dS1**2 / (sigma1**2 * grid.S1_max**2)
    dt_max_S2 = grid.dS2**2 / (sigma2**2 * grid.S2_max**2)
    dt_max = min(dt_max_S1, dt_max_S2)

    # CFL 조건
    CFL_S1 = sigma1**2 * grid.S1_max**2 * grid.dt / grid.dS1**2
    CFL_S2 = sigma2**2 * grid.S2_max**2 * grid.dt / grid.dS2**2

    return {
        'dt': grid.dt,
        'dt_max_explicit': dt_max,
        'is_explicit_stable': grid.dt <= dt_max,
        'CFL_S1': CFL_S1,
        'CFL_S2': CFL_S2,
        'CFL_condition': max(CFL_S1, CFL_S2) <= 0.5,  # 일반적 안정 조건
    }
