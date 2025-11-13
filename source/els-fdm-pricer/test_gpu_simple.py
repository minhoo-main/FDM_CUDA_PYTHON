#!/usr/bin/env python3
"""
간단한 GPU 테스트

CuPy 없이도 작동하는지 확인
"""

import sys
sys.path.insert(0, '/home/minhoo/els-fdm-pricer')

print("\n" + "=" * 60)
print(" " * 20 + "GPU 기능 테스트")
print("=" * 60)

# 1. GPU 환경 확인
print("\n[1] GPU 환경 확인")
print("-" * 60)

try:
    from src.solvers.gpu_adi_solver import check_gpu_available

    info = check_gpu_available()
    print(f"CuPy 설치: {'✓' if info['cupy_installed'] else '✗'}")
    print(f"GPU 사용 가능: {'✓' if info['gpu_available'] else '✗'}")

    if info['gpu_available']:
        print(f"GPU: {info['gpu_name']}")
        print(f"메모리: {info['gpu_memory']}")
    else:
        print("→ CPU 모드로 작동합니다")

except Exception as e:
    print(f"⚠️  에러: {e}")

# 2. CPU 모드 테스트 (항상 작동해야 함)
print("\n[2] CPU 모드 테스트")
print("-" * 60)

try:
    from src.models.els_product import create_sample_els
    from src.pricing.els_pricer import price_els
    import time

    product = create_sample_els()

    print("40x40 그리드로 빠른 테스트...")
    start = time.time()
    result = price_els(product, N1=40, N2=40, Nt=80, verbose=False)
    elapsed = time.time() - start

    print(f"✓ CPU 평가 성공!")
    print(f"  가격: {result['price']:.4f}")
    print(f"  시간: {elapsed:.2f}초")

except Exception as e:
    print(f"✗ CPU 테스트 실패: {e}")
    import traceback
    traceback.print_exc()

# 3. GPU 모드 테스트 (CuPy가 있으면)
print("\n[3] GPU 모드 테스트")
print("-" * 60)

try:
    from src.pricing.gpu_els_pricer import price_els_gpu

    print("40x40 그리드로 GPU 테스트...")
    start = time.time()
    result = price_els_gpu(product, N1=40, N2=40, Nt=80, use_gpu=True, verbose=False)
    elapsed = time.time() - start

    if result.get('use_gpu'):
        print(f"✓ GPU 평가 성공!")
    else:
        print(f"✓ CPU 모드로 평가 성공! (GPU 없음)")

    print(f"  가격: {result['price']:.4f}")
    print(f"  시간: {elapsed:.2f}초")

except ImportError:
    print("⚠️  GPU 모듈을 import할 수 없습니다 (CuPy 미설치)")
    print("   → 정상입니다. CPU 모드로 사용하세요")
except Exception as e:
    print(f"✗ GPU 테스트 실패: {e}")
    import traceback
    traceback.print_exc()

# 4. 요약
print("\n" + "=" * 60)
print("요약")
print("=" * 60)

try:
    info = check_gpu_available()

    if info['gpu_available']:
        print("✓ GPU 가속 사용 가능!")
        print("  - benchmark_gpu.py로 성능 테스트 권장")
        print("  - GPU_GUIDE.md 참고")
    elif info['cupy_installed']:
        print("⚠️  CuPy는 설치되었지만 GPU가 감지되지 않습니다")
        print("  - nvidia-smi로 GPU 확인")
        print("  - CPU 모드로 정상 작동")
    else:
        print("⚠️  CuPy가 설치되지 않았습니다")
        print("  - CPU 모드로 정상 작동")
        print("  - GPU 가속을 원하면:")
        print("    pip install cupy-cuda11x  # CUDA 11.x")
        print("    pip install cupy-cuda12x  # CUDA 12.x")
except:
    print("✓ 시스템 정상 작동 (CPU 모드)")

print("=" * 60)
print("\n✅ 테스트 완료!")
