"""
ELS 가격 평가 결과 시각화 예제

다양한 시각화 생성:
1. 3D 가격 surface
2. 2D contour plot
3. 조기상환 경계면
4. 시간에 따른 가격 변화
5. 만기 페이오프 비교
"""

from src.models.els_product import create_sample_els
from src.pricing.els_pricer import price_els
from src.visualization.els_visualizer import ELSVisualizer

def main():
    print("="*60)
    print("ELS Pricing Visualization Example")
    print("="*60)

    # 1. ELS 상품 생성
    print("\n1. Creating ELS product...")
    product = create_sample_els()

    # 2. 가격 계산 (중간 결과 저장 포함)
    print("\n2. Pricing ELS...")
    result = price_els(
        product,
        N1=80,
        N2=80,
        Nt=150,
        verbose=True
    )

    print(f"\n   ELS Price: {result['price']:.4f}")

    # 3. 시각화 생성
    print("\n3. Generating visualizations...")
    visualizer = ELSVisualizer(result, output_dir="output/plots")

    # 개별 그래프 생성 (show=False로 저장만)
    print("\n   Creating individual plots...")

    print("   - 3D Price Surface...")
    visualizer.plot_price_surface_3d(save=True, show=False)

    print("   - 2D Price Contour...")
    visualizer.plot_price_contour(save=True, show=False)

    print("   - Early Redemption Boundaries...")
    visualizer.plot_early_redemption_boundary(save=True, show=False)

    print("   - Price Evolution...")
    visualizer.plot_price_evolution(save=True, show=False)

    print("   - Payoff Comparison...")
    visualizer.plot_payoff_comparison(save=True, show=False)

    print("\n✓ All visualizations saved to: output/plots/")
    print("\nGenerated files:")
    print("  - price_surface_3d.png")
    print("  - price_contour.png")
    print("  - early_redemption_boundary.png")
    print("  - price_evolution.png")
    print("  - payoff_comparison.png")

    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)


def quick_visualization():
    """빠른 시각화 (한 번에 모든 그래프 생성)"""
    print("Quick Visualization Mode")
    print("="*60)

    product = create_sample_els()
    result = price_els(product, N1=60, N2=60, Nt=100, verbose=False)

    # 모든 그래프 한 번에 생성
    visualizer = ELSVisualizer(result, output_dir="output/quick")
    visualizer.plot_all(save=True, show=False)


if __name__ == "__main__":
    # 기본 모드
    main()

    # 또는 빠른 모드
    # quick_visualization()
