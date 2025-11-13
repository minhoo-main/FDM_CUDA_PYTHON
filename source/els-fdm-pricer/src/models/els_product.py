"""
ELS 상품 정의
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ELSProduct:
    """Step-Down ELS 상품"""

    # 기본 정보
    principal: float = 100.0  # 원금
    maturity: float = 3.0  # 만기 (년)

    # 조기상환 조건
    observation_dates: List[float] = None  # 조기상환 평가일 (년)
    redemption_barriers: List[float] = None  # 조기상환 배리어 (%)
    coupons: List[float] = None  # 각 조기상환 시 쿠폰 (%)

    # 낙인 조건
    ki_barrier: float = 0.50  # 낙인 배리어 (50%)
    ki_observation_start: float = 0.0  # 낙인 관찰 시작
    ki_observation_end: float = None  # 낙인 관찰 종료 (None이면 만기)

    # 기초자산 정보
    S1_0: float = 100.0  # 기초자산 1 초기가
    S2_0: float = 100.0  # 기초자산 2 초기가
    sigma1: float = 0.25  # 기초자산 1 변동성
    sigma2: float = 0.30  # 기초자산 2 변동성
    rho: float = 0.50  # 상관계수

    # 시장 파라미터
    r: float = 0.03  # 무위험이자율
    q1: float = 0.02  # 기초자산 1 배당률
    q2: float = 0.015  # 기초자산 2 배당률

    # 페이오프 타입
    worst_of: bool = True  # Worst-of (True) or Best-of (False)

    def __post_init__(self):
        """기본값 설정"""
        if self.observation_dates is None:
            # 6개월 단위 조기상환 (0.5, 1.0, ..., 3.0)
            self.observation_dates = [0.5 * (i + 1) for i in range(6)]

        if self.redemption_barriers is None:
            # Step-Down 배리어: 95%, 95%, 90%, 85%, 80%, 75%
            self.redemption_barriers = [0.95, 0.95, 0.90, 0.85, 0.80, 0.75]

        if self.coupons is None:
            # 연 8% 쿠폰, 조기상환 시 경과기간에 비례
            annual_coupon = 8.0
            self.coupons = [annual_coupon * t for t in self.observation_dates]

        if self.ki_observation_end is None:
            self.ki_observation_end = self.maturity

        # 검증
        assert len(self.observation_dates) == len(self.redemption_barriers)
        assert len(self.observation_dates) == len(self.coupons)

    def payoff_at_maturity(self, S1: np.ndarray, S2: np.ndarray, ki_occurred: bool) -> np.ndarray:
        """
        만기 페이오프 계산

        Args:
            S1: 기초자산 1 가격
            S2: 기초자산 2 가격
            ki_occurred: 낙인 발생 여부

        Returns:
            만기 페이오프
        """
        # Worst-of 퍼포먼스
        if self.worst_of:
            performance = np.minimum(S1 / self.S1_0, S2 / self.S2_0)
        else:
            performance = np.maximum(S1 / self.S1_0, S2 / self.S2_0)

        # 낙인 미발생 시: 원금 + 쿠폰
        # 낙인 발생 시: min(원금, 원금 * performance)
        if ki_occurred:
            payoff = self.principal * np.minimum(1.0, performance)
        else:
            payoff = self.principal + self.coupons[-1]

        return payoff

    def check_early_redemption(self, S1: float, S2: float, obs_idx: int) -> tuple:
        """
        조기상환 조건 체크

        Args:
            S1: 기초자산 1 가격
            S2: 기초자산 2 가격
            obs_idx: 관찰일 인덱스

        Returns:
            (조기상환 여부, 페이오프)
        """
        if obs_idx >= len(self.observation_dates):
            return False, 0.0

        # Worst-of 퍼포먼스
        if self.worst_of:
            performance = min(S1 / self.S1_0, S2 / self.S2_0)
        else:
            performance = max(S1 / self.S1_0, S2 / self.S2_0)

        barrier = self.redemption_barriers[obs_idx]

        if performance >= barrier:
            # 조기상환 발생
            payoff = self.principal + self.coupons[obs_idx]
            return True, payoff

        return False, 0.0

    def check_knock_in(self, S1: float, S2: float) -> bool:
        """
        낙인 조건 체크

        Args:
            S1: 기초자산 1 가격
            S2: 기초자산 2 가격

        Returns:
            낙인 발생 여부
        """
        # Worst-of 퍼포먼스
        if self.worst_of:
            performance = min(S1 / self.S1_0, S2 / self.S2_0)
        else:
            performance = max(S1 / self.S1_0, S2 / self.S2_0)

        return performance < self.ki_barrier

    def __repr__(self):
        """문자열 표현"""
        return f"""
ELS Product (Step-Down, Worst-of)
================================
Principal: {self.principal}
Maturity: {self.maturity} years
Observation Dates: {self.observation_dates}
Redemption Barriers: {[f'{b*100:.0f}%' for b in self.redemption_barriers]}
Coupons: {[f'{c:.1f}%' for c in self.coupons]}
KI Barrier: {self.ki_barrier*100:.0f}%

Underlying Assets:
- S1: {self.S1_0} (σ={self.sigma1}, q={self.q1})
- S2: {self.S2_0} (σ={self.sigma2}, q={self.q2})
- Correlation: {self.rho}

Market Parameters:
- Risk-free rate: {self.r}
"""


def create_sample_els() -> ELSProduct:
    """샘플 Step-Down ELS 생성"""
    return ELSProduct(
        principal=100.0,
        maturity=3.0,
        observation_dates=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        redemption_barriers=[0.95, 0.95, 0.90, 0.85, 0.80, 0.75],
        coupons=[4.0, 8.0, 12.0, 16.0, 20.0, 24.0],  # 연 8%
        ki_barrier=0.50,
        S1_0=100.0,
        S2_0=100.0,
        sigma1=0.25,
        sigma2=0.30,
        rho=0.50,
        r=0.03,
        q1=0.02,
        q2=0.015,
        worst_of=True
    )
