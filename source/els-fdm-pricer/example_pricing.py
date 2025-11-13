#!/usr/bin/env python3
"""
ELS FDM 프라이서 예제

Step-Down ELS 가격 평가 데모
"""

import sys
import numpy as np
import time

# 프로젝트 경로 추가
sys.path.insert(0, '/home/minhoo/els-fdm-pricer')

from src.models.els_product import ELSProduct, create_sample_els
from src.pricing.els_pricer import price_els
from src.grid.grid_2d import create_adaptive_grid, check_stability


def example_1_basic_pricing():
    """예제 1: 기본 가격 평가"""
    print("\n" + "=" * 80)
    print("예제 1: 기본 Step-Down ELS 가격 평가")
    print("=" * 80)

    # 샘플 ELS 상품 생성
    product = create_sample_els()

    # 가격 평가 (기본 그리드)
    start = time.time()
    result = price_els(
        product=product,
        N1=80,  # S1 방향 그리드
        N2=80,  # S2 방향 그리드
        Nt=150,  # 시간 스텝
        space_factor=3.0,  # 공간 범위: 0 ~ 3*S0
        verbose=True
    )
    elapsed = time.time() - start

    print(f"\n계산 시간: {elapsed:.2f}초")
    print(f"\nELS 가격: {result['price']:.4f} (원금 대비 {result['price']:.2f}%)")


def example_2_grid_convergence():
    """예제 2: 그리드 수렴성 테스트"""
    print("\n" + "=" * 80)
    print("예제 2: 그리드 수렴성 테스트")
    print("=" * 80)

    product = create_sample_els()

    grid_sizes = [40, 60, 80, 100]
    prices = []

    print("\n그리드 크기별 가격 비교:")
    print(f"{'Grid Size':<12} {'Price':<12} {'Time (s)':<10} {'Diff':<10}")
    print("-" * 50)

    for N in grid_sizes:
        start = time.time()
        result = price_els(product, N1=N, N2=N, Nt=N*2, verbose=False)
        elapsed = time.time() - start

        price = result['price']
        prices.append(price)

        diff = "" if len(prices) == 1 else f"{price - prices[-2]:.4f}"
        print(f"{N:<12} {price:<12.4f} {elapsed:<10.2f} {diff:<10}")

    # 수렴률 계산
    if len(prices) >= 2:
        print(f"\n최종 가격 차이: {abs(prices[-1] - prices[-2]):.6f}")
        print(f"수렴률: {abs(prices[-1] - prices[-2]) / prices[-1] * 100:.4f}%")


def example_3_parameter_sensitivity():
    """예제 3: 파라미터 민감도 분석"""
    print("\n" + "=" * 80)
    print("예제 3: 파라미터 민감도 분석")
    print("=" * 80)

    base_product = create_sample_els()
    base_price = price_els(base_product, N1=60, N2=60, Nt=120, verbose=False)['price']

    print(f"\n기준 가격: {base_price:.4f}\n")

    # 변동성 민감도
    print("=" * 50)
    print("변동성 민감도 (σ1)")
    print("-" * 50)
    print(f"{'σ1':<10} {'가격':<12} {'변화율':<10}")
    print("-" * 50)

    for sigma1 in [0.15, 0.20, 0.25, 0.30, 0.35]:
        product = create_sample_els()
        product.sigma1 = sigma1
        price = price_els(product, N1=60, N2=60, Nt=120, verbose=False)['price']
        change = (price - base_price) / base_price * 100
        print(f"{sigma1:<10.2f} {price:<12.4f} {change:>+9.2f}%")

    # 상관계수 민감도
    print("\n" + "=" * 50)
    print("상관계수 민감도 (ρ)")
    print("-" * 50)
    print(f"{'ρ':<10} {'가격':<12} {'변화율':<10}")
    print("-" * 50)

    for rho in [0.0, 0.25, 0.50, 0.75, 0.90]:
        product = create_sample_els()
        product.rho = rho
        price = price_els(product, N1=60, N2=60, Nt=120, verbose=False)['price']
        change = (price - base_price) / base_price * 100
        print(f"{rho:<10.2f} {price:<12.4f} {change:>+9.2f}%")

    # 배리어 민감도
    print("\n" + "=" * 50)
    print("조기상환 배리어 민감도 (첫 번째 배리어)")
    print("-" * 50)
    print(f"{'배리어':<10} {'가격':<12} {'변화율':<10}")
    print("-" * 50)

    for barrier in [0.85, 0.90, 0.95, 1.00]:
        product = create_sample_els()
        product.redemption_barriers[0] = barrier
        product.redemption_barriers[1] = barrier
        price = price_els(product, N1=60, N2=60, Nt=120, verbose=False)['price']
        change = (price - base_price) / base_price * 100
        print(f"{barrier:<10.2%} {price:<12.4f} {change:>+9.2f}%")


def example_4_stability_check():
    """예제 4: FDM 안정성 체크"""
    print("\n" + "=" * 80)
    print("예제 4: FDM 안정성 조건 체크")
    print("=" * 80)

    product = create_sample_els()

    # 여러 그리드 설정 테스트
    configs = [
        {'N1': 50, 'N2': 50, 'Nt': 100},
        {'N1': 100, 'N2': 100, 'Nt': 200},
        {'N1': 150, 'N2': 150, 'Nt': 300},
    ]

    for config in configs:
        grid = create_adaptive_grid(
            S1_0=product.S1_0,
            S2_0=product.S2_0,
            T=product.maturity,
            **config
        )

        stability = check_stability(grid, product.sigma1, product.sigma2, product.r)

        print(f"\n그리드: N1={config['N1']}, N2={config['N2']}, Nt={config['Nt']}")
        print(f"  dt = {stability['dt']:.6f}")
        print(f"  dt_max (Explicit) = {stability['dt_max_explicit']:.6f}")
        print(f"  Explicit 안정성: {'✓ 안정' if stability['is_explicit_stable'] else '✗ 불안정'}")
        print(f"  CFL(S1) = {stability['CFL_S1']:.4f}")
        print(f"  CFL(S2) = {stability['CFL_S2']:.4f}")
        print(f"  CFL 조건: {'✓ 만족' if stability['CFL_condition'] else '✗ 위반'}")


def example_5_custom_product():
    """예제 5: 커스텀 ELS 상품"""
    print("\n" + "=" * 80)
    print("예제 5: 커스텀 ELS 상품 평가")
    print("=" * 80)

    # 공격적인 Step-Down ELS 설계
    aggressive_els = ELSProduct(
        principal=100.0,
        maturity=2.0,  # 2년 만기
        observation_dates=[0.5, 1.0, 1.5, 2.0],  # 6개월 단위
        redemption_barriers=[0.90, 0.85, 0.80, 0.75],  # 빠른 step-down
        coupons=[5.0, 10.0, 15.0, 20.0],  # 연 10% (2년이면 20%)
        ki_barrier=0.45,  # 낮은 낙인 배리어 (45%)
        S1_0=100.0,
        S2_0=100.0,
        sigma1=0.30,  # 높은 변동성
        sigma2=0.35,
        rho=0.60,
        r=0.03,
        q1=0.02,
        q2=0.015,
        worst_of=True
    )

    print("\n공격적 Step-Down ELS (2년, 연 10% 쿠폰, 낮은 배리어)")
    result_aggressive = price_els(aggressive_els, N1=80, N2=80, Nt=160, verbose=True)

    # 보수적인 ELS 설계
    conservative_els = ELSProduct(
        principal=100.0,
        maturity=3.0,  # 3년 만기
        observation_dates=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        redemption_barriers=[0.95, 0.95, 0.95, 0.90, 0.90, 0.85],  # 높은 배리어
        coupons=[3.0, 6.0, 9.0, 12.0, 15.0, 18.0],  # 연 6%
        ki_barrier=0.55,  # 높은 낙인 배리어
        S1_0=100.0,
        S2_0=100.0,
        sigma1=0.20,  # 낮은 변동성
        sigma2=0.25,
        rho=0.40,
        r=0.03,
        q1=0.02,
        q2=0.015,
        worst_of=True
    )

    print("\n보수적 ELS (3년, 연 6% 쿠폰, 높은 배리어)")
    result_conservative = price_els(conservative_els, N1=80, N2=80, Nt=180, verbose=True)

    print("\n" + "=" * 80)
    print("비교 결과")
    print("=" * 80)
    print(f"공격적 ELS 가격: {result_aggressive['price']:.4f}")
    print(f"보수적 ELS 가격: {result_conservative['price']:.4f}")
    print(f"차이: {result_aggressive['price'] - result_conservative['price']:.4f}")


def main():
    """메인 실행 함수"""
    print("\n")
    print("=" * 80)
    print(" " * 20 + "ELS FDM 프라이서 예제")
    print("=" * 80)

    examples = [
        ("1", "기본 가격 평가", example_1_basic_pricing),
        ("2", "그리드 수렴성 테스트", example_2_grid_convergence),
        ("3", "파라미터 민감도 분석", example_3_parameter_sensitivity),
        ("4", "FDM 안정성 체크", example_4_stability_check),
        ("5", "커스텀 ELS 상품", example_5_custom_product),
    ]

    print("\n실행할 예제를 선택하세요:")
    for num, desc, _ in examples:
        print(f"  {num}. {desc}")
    print("  0. 전체 실행")
    print("  q. 종료")

    choice = input("\n선택: ").strip()

    if choice == 'q':
        return

    if choice == '0':
        # 전체 실행
        for num, desc, func in examples:
            func()
    else:
        # 선택한 예제 실행
        for num, desc, func in examples:
            if num == choice:
                func()
                break
        else:
            print("잘못된 선택입니다.")


if __name__ == "__main__":
    main()
