# GPU 환경에서 프로젝트 테스트 가이드

다른 컴퓨터(GPU 있는 환경)에서 이 프로젝트를 가져와서 테스트하는 방법입니다.

---

## 📋 준비 사항

### 필수 요구사항
- ✅ NVIDIA GPU (CUDA 지원)
- ✅ CUDA Toolkit 11.x 또는 12.x 설치
- ✅ Python 3.8 이상
- ✅ Git

### GPU 환경 확인
```bash
# GPU 확인
nvidia-smi

# CUDA 버전 확인
nvcc --version
# 또는
nvidia-smi  # 오른쪽 상단에 CUDA Version 표시
```

---

## 🚀 Step 1: 프로젝트 Clone

```bash
# 작업 디렉토리로 이동
cd ~

# GitHub에서 프로젝트 clone
git clone https://github.com/minhoo-main/FDM_CUDA.git

# 프로젝트 디렉토리로 이동
cd FDM_CUDA

# 파일 확인
ls -la
```

**예상 출력:**
```
drwxr-xr-x  src/
-rw-r--r--  README.md
-rw-r--r--  requirements.txt
-rw-r--r--  benchmark_optimized.py
-rw-r--r--  test_optimized.py
...
```

---

## 📦 Step 2: Python 환경 설정

### 방법 A: 가상환경 사용 (추천)

```bash
# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows

# pip 업그레이드
pip install --upgrade pip
```

### 방법 B: 시스템 Python 사용

```bash
# 시스템 Python 사용 (권장하지 않음)
# 아래 설치 명령에서 pip 대신 pip3 --break-system-packages 사용
```

---

## 🔧 Step 3: 의존성 설치

### 3-1. 기본 패키지 설치

```bash
# requirements.txt 설치
pip install -r requirements.txt
```

**설치되는 패키지:**
- numpy, scipy, pandas
- matplotlib (시각화)
- psutil

### 3-2. CuPy 설치 (GPU 가속)

**중요:** CUDA 버전에 맞는 CuPy를 설치하세요!

```bash
# CUDA 버전 확인
nvidia-smi  # 오른쪽 상단 CUDA Version 확인
```

**CUDA 11.x 사용하는 경우:**
```bash
pip install cupy-cuda11x
```

**CUDA 12.x 사용하는 경우:**
```bash
pip install cupy-cuda12x
```

**설치 확인:**
```bash
python3 -c "import cupy as cp; print('CuPy version:', cp.__version__); print('GPU:', cp.cuda.Device().name)"
```

**예상 출력:**
```
CuPy version: 12.3.0
GPU: NVIDIA GeForce RTX 3080
```

---

## ✅ Step 4: 정확성 테스트

먼저 알고리즘이 정확한지 확인합니다.

```bash
# CPU vs GPU 정확성 테스트
python3 test_optimized.py
```

**예상 출력:**
```
============================================================
Testing Optimized GPU Implementation
============================================================

1. CPU Baseline
------------------------------------------------------------
...
CPU Price:          106.655756

2. Optimized GPU
------------------------------------------------------------
✓ Optimized GPU 가속 활성화: NVIDIA GeForce RTX 3080
...
Optimized GPU Price: 106.655756
Difference:          0.000000 (0.0000%)

✅ Test PASSED: Prices match within 1%
```

**✅ 성공 기준:** Difference가 0.01% 이내

---

## 🏃 Step 5: 성능 벤치마크 (핵심!)

이제 실제 GPU 성능을 측정합니다.

```bash
# 전체 벤치마크 실행
python3 benchmark_optimized.py
```

**예상 실행 시간:** 5-10분

**예상 출력 (예시):**
```
============================================================
ELS FDM Pricer Performance Benchmark
============================================================

Phase 1 Optimization Test:
1. Batched Tridiagonal Solver
2. Vectorized Early Redemption Check
3. Vectorized Terminal Payoff
============================================================


############################################################
# Test: Small Grid (60×60, 120 time steps)
############################################################

CPU Benchmark: 60×60 grid, 120 time steps
------------------------------------------------------------
Run 1/2: 8.1234s, Price: 106.6558
Run 2/2: 8.0987s, Price: 106.6558

Average: 8.1111s ± 0.0124s
Price: 106.6558

Original GPU Benchmark: 60×60 grid, 120 time steps
------------------------------------------------------------
✓ GPU 가속 활성화: NVIDIA GeForce RTX 3080
Run 1/3: 0.3245s, Price: 106.6558
Run 2/3: 0.3198s, Price: 106.6558
Run 3/3: 0.3221s, Price: 106.6558

Average: 0.3221s ± 0.0019s
Price: 106.6558

🚀 Optimized GPU Benchmark: 60×60 grid, 120 time steps
------------------------------------------------------------
✓ Optimized GPU 가속 활성화: NVIDIA GeForce RTX 3080
Run 1/3: 0.0234s, Price: 106.6558
Run 2/3: 0.0228s, Price: 106.6558
Run 3/3: 0.0231s, Price: 106.6558

Average: 0.0231s ± 0.0002s
Price: 106.6558

📊 Performance Comparison Summary
============================================================

Method                Time (s)        Speedup        Price
------------------------------------------------------------
CPU                      8.11s          1.0x     106.6558
GPU (Original)           0.32s         25.2x     106.6558
GPU (Optimized) 🚀       0.02s        351.3x     106.6558

🎯 GPU Optimization Gain: 13.9x faster than original GPU

============================================================

✓ Price Verification:
  Price range: 106.6558 ~ 106.6558
  Max difference: 0.000000 (0.0000%)
  ✅ All methods agree (< 0.1% difference)


############################################################
# Test: Medium Grid (80×80, 150 time steps)
############################################################
...

############################################################
# Test: Large Grid (100×100, 200 time steps)
############################################################
...
```

**📊 결과 기록:**
- Small, Medium, Large 각각의 성능 수치 확인
- CPU 대비 GPU (Original) 속도 향상
- GPU (Original) 대비 GPU (Optimized) 속도 향상
- **예측(10-15배)과 비교**

---

## 📸 Step 6: 시각화 생성

```bash
# 시각화 예제 실행
python3 visualize_example.py
```

**생성되는 파일:**
```
output/plots/
├── price_surface_3d.png           # 3D 가격 surface
├── price_contour.png              # 2D contour
├── early_redemption_boundary.png  # 조기상환 경계
├── price_evolution.png            # 가격 변화
└── payoff_comparison.png          # V_0 vs V_T
```

**그래프 확인:**
```bash
# Linux GUI 환경
xdg-open output/plots/price_surface_3d.png

# Windows
start output/plots/price_surface_3d.png

# Mac
open output/plots/price_surface_3d.png
```

---

## 📊 Step 7: 결과 분석 및 기록

### 벤치마크 결과 정리

`benchmark_optimized.py` 실행 후, 다음 정보를 기록하세요:

#### GPU 환경 정보
```
GPU 모델: [예: NVIDIA RTX 3080]
CUDA Version: [예: 12.1]
CuPy Version: [예: 12.3.0]
Driver Version: [예: 535.54.03]
```

#### 성능 결과 (80×80 그리드 기준)

| Method | Time | CPU 대비 | 이전 대비 |
|--------|------|----------|-----------|
| CPU | ?s | 1x | - |
| GPU (Original) | ?s | ?x | - |
| GPU (Optimized) | ?s | ?x | ?x |

#### 실제 vs 예측 비교

| 항목 | 예측 | 실제 | 비고 |
|------|------|------|------|
| GPU (Original) 향상 | ~40배 | ?배 | CPU 대비 |
| GPU (Optimized) 추가 향상 | 10-15배 | ?배 | Original 대비 |

---

## 🔬 Step 8: 추가 실험 (선택)

### 실험 1: 다양한 그리드 크기 테스트

```python
# custom_benchmark.py 생성
from src.models.els_product import create_sample_els
from src.pricing.gpu_els_pricer_optimized import price_els_optimized
import time

product = create_sample_els()

grid_sizes = [
    (40, 40, 80),
    (60, 60, 120),
    (80, 80, 150),
    (100, 100, 200),
    (150, 150, 300),
    (200, 200, 400),
]

for N1, N2, Nt in grid_sizes:
    start = time.time()
    result = price_els_optimized(product, N1=N1, N2=N2, Nt=Nt, verbose=False)
    elapsed = time.time() - start
    print(f"{N1}×{N2} grid, {Nt} steps: {elapsed:.4f}s, Price: {result['price']:.4f}")
```

### 실험 2: 다양한 ELS 상품 테스트

```python
from src.models.els_product import ELSProduct

# 공격적인 상품 (높은 배리어)
aggressive_product = ELSProduct(
    redemption_barriers=[0.90, 0.90, 0.85, 0.80, 0.75, 0.70],
    coupons=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
    ki_barrier=0.45
)

# 보수적인 상품 (낮은 배리어)
conservative_product = ELSProduct(
    redemption_barriers=[0.95, 0.95, 0.95, 0.90, 0.90, 0.85],
    coupons=[3.0, 6.0, 9.0, 12.0, 15.0, 18.0],
    ki_barrier=0.55
)
```

---

## 🐛 문제 해결

### CuPy 설치 오류

**"CUDA not found" 오류:**
```bash
# CUDA Toolkit 설치 확인
nvcc --version

# 없으면 CUDA Toolkit 설치 필요
# Ubuntu: https://developer.nvidia.com/cuda-downloads
```

**"Incompatible CUDA version" 오류:**
```bash
# CUDA 버전 확인
nvidia-smi

# 맞는 CuPy 버전 설치
pip uninstall cupy-cuda11x cupy-cuda12x
pip install cupy-cuda12x  # 또는 cupy-cuda11x
```

### 메모리 부족 오류

**"out of memory" 오류:**
```python
# 그리드 크기 줄이기
result = price_els_optimized(product, N1=50, N2=50, Nt=100)

# 또는 GPU 메모리 확인
import cupy as cp
print(cp.cuda.Device().mem_info)  # (free, total)
```

### 성능이 예상보다 낮은 경우

**가능한 원인:**
1. GPU가 다른 프로세스 사용 중
2. GPU 메모리 부족으로 swap 발생
3. PCIe 대역폭 제한
4. 구형 GPU

**확인 방법:**
```bash
# GPU 사용률 모니터링
watch -n 1 nvidia-smi

# 다른 프로세스 종료
# nvidia-smi에서 PID 확인 후
kill <PID>
```

---

## 📝 Step 9: Claude와 결과 공유

다른 환경에서 Claude를 다시 실행할 때:

### 1. 프로젝트 경로 알려주기
```
저는 ~/FDM_CUDA 디렉토리에 프로젝트를 clone했습니다.
```

### 2. 벤치마크 결과 공유
```
벤치마크 결과입니다:

GPU: NVIDIA RTX 3080
CUDA: 12.1

80×80 그리드:
- CPU: 12.34s
- GPU (Original): 0.45s (27배)
- GPU (Optimized): 0.03s (411배, Original 대비 15배)

예측이 정확했습니다!
```

### 3. 추가 작업 요청
```
성능이 좋습니다. 이제 다음을 하고 싶습니다:
- 더 큰 그리드 테스트
- 다양한 ELS 상품 비교
- 결과 보고서 작성
```

---

## 📋 체크리스트

다음을 확인하세요:

- [ ] GPU 환경 준비 (nvidia-smi 확인)
- [ ] 프로젝트 clone
- [ ] 가상환경 생성 및 활성화
- [ ] requirements.txt 설치
- [ ] CuPy 설치 (CUDA 버전 맞게)
- [ ] test_optimized.py 실행 (정확성 확인)
- [ ] benchmark_optimized.py 실행 (성능 측정)
- [ ] 결과 기록
- [ ] 시각화 생성 (선택)
- [ ] 결과 분석

---

## 🎯 최종 목표

이 가이드를 따라하면:

1. ✅ GPU 환경에서 프로젝트 정상 작동 확인
2. ✅ 실제 GPU 성능 측정
3. ✅ 예측(10-15배) vs 실제 비교
4. ✅ 시각화 그래프 생성
5. ✅ 프로덕션 준비 완료 확인

**예상 소요 시간:** 30분 - 1시간

**성공 기준:**
- 정확성 테스트 통과 (가격 일치)
- GPU Optimized가 Original보다 5배 이상 빠름
- 멋진 시각화 그래프 생성

---

## 📞 도움이 필요한 경우

문제가 발생하면:

1. **에러 메시지 전체 복사**
2. **실행한 명령어 기록**
3. **환경 정보 수집:**
   ```bash
   python3 --version
   pip list
   nvidia-smi
   ```
4. **Claude에게 공유**

Claude가 문제를 해결하도록 도와드리겠습니다!

---

**행운을 빕니다! GPU에서 엄청난 속도를 경험하세요! 🚀**
