"""
최적화된 GPU 구현 테스트

간단한 테스트로 최적화된 버전이 제대로 작동하는지 확인
"""

import numpy as np
from src.models.els_product import create_sample_els
from src.pricing.els_pricer import price_els
from src.pricing.gpu_els_pricer_optimized import price_els_optimized

def test_basic():
    """기본 동작 테스트"""
    print("="*60)
    print("Testing Optimized GPU Implementation")
    print("="*60)

    product = create_sample_els()

    # Small grid test
    N1, N2, Nt = 40, 40, 80

    print("\n1. CPU Baseline")
    print("-" * 60)
    cpu_result = price_els(product, N1=N1, N2=N2, Nt=Nt, verbose=True)

    print("\n2. Optimized GPU")
    print("-" * 60)
    try:
        gpu_result = price_els_optimized(product, N1=N1, N2=N2, Nt=Nt, use_gpu=True, verbose=True)

        # 가격 비교
        print("\n" + "="*60)
        print("Price Comparison")
        print("="*60)
        print(f"CPU Price:          {cpu_result['price']:.6f}")
        print(f"Optimized GPU Price: {gpu_result['price']:.6f}")
        price_diff = abs(cpu_result['price'] - gpu_result['price'])
        print(f"Difference:          {price_diff:.6f} ({price_diff/cpu_result['price']*100:.4f}%)")

        if price_diff / cpu_result['price'] < 0.01:  # 1% 이내
            print("\n✅ Test PASSED: Prices match within 1%")
        else:
            print("\n⚠️  Test WARNING: Price difference > 1%")

    except ImportError:
        print("\n⚠️  CuPy not available, skipping GPU test")
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic()
