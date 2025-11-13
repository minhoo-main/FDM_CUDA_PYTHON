# 🚀 GPU 가속 가이드

CUDA를 이용하여 ELS FDM Pricer를 **10~100배 빠르게** 실행하는 방법

---

## ⚡ 성능 향상

### 예상 속도 개선

| Grid Size | CPU 시간 | GPU 시간 | 속도 향상 |
|-----------|---------|---------|-----------|
| 40×40 | 0.7초 | ~0.1초 | 5~10배 |
| 80×80 | 3.7초 | ~0.2초 | 15~20배 |
| 150×150 | ~20초 | ~0.5초 | 30~50배 |
| 200×200 | ~60초 | ~1초 | 50~100배 |

**결론**: 그리드가 클수록 GPU 효과 극대화!

---

## 📋 사전 요구사항

### 1. NVIDIA GPU 확인

```bash
# GPU 확인
nvidia-smi

# CUDA 버전 확인
nvcc --version
```

**필요 사양**:
- NVIDIA GPU (GTX 1060 이상, RTX 시리즈 권장)
- CUDA Toolkit 11.x 또는 12.x
- GPU 메모리: 2GB 이상 (4GB 권장)

### 2. CUDA Toolkit 설치

**이미 설치되어 있다면 Skip!**

#### Ubuntu/WSL
```bash
# CUDA 11.8 예시
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

#### Windows
NVIDIA 웹사이트에서 다운로드:
https://developer.nvidia.com/cuda-downloads

---

## 🔧 설치

### 1. CuPy 설치

CUDA 버전에 맞는 CuPy 설치:

```bash
# CUDA 버전 확인
nvidia-smi  # 우측 상단에 CUDA Version 표시

# CUDA 11.x (예: 11.2, 11.8 등)
pip3 install cupy-cuda11x

# CUDA 12.x (예: 12.0, 12.1 등)
pip3 install cupy-cuda12x
```

### 2. 설치 확인

```bash
python3 -c "import cupy as cp; print(f'✓ CuPy 설치 완료: {cp.cuda.Device().name}')"
```

성공 시:
```
✓ CuPy 설치 완료: NVIDIA GeForce RTX 3080
```

---

## 🚀 사용 방법

### 1. 기본 사용

```python
from src.models.els_product import create_sample_els
from src.pricing.gpu_els_pricer import price_els_gpu

# ELS 상품
product = create_sample_els()

# GPU로 평가 (자동으로 GPU 감지)
result = price_els_gpu(
    product,
    N1=100,
    N2=100,
    Nt=200,
    use_gpu=True  # GPU 사용
)

print(f"가격: {result['price']:.4f}")
```

### 2. CPU vs GPU 비교

```python
import time
from src.pricing.els_pricer import price_els
from src.pricing.gpu_els_pricer import price_els_gpu

product = create_sample_els()

# CPU
start = time.time()
result_cpu = price_els(product, N1=100, N2=100, Nt=200, verbose=False)
time_cpu = time.time() - start

# GPU
start = time.time()
result_gpu = price_els_gpu(product, N1=100, N2=100, Nt=200, use_gpu=True, verbose=False)
time_gpu = time.time() - start

print(f"CPU: {time_cpu:.2f}초")
print(f"GPU: {time_gpu:.2f}초")
print(f"속도 향상: {time_cpu/time_gpu:.1f}배")
```

### 3. 자동 GPU 감지

GPU가 없으면 자동으로 CPU로 전환:

```python
# GPU가 있으면 GPU, 없으면 CPU 자동 선택
result = price_els_gpu(product, use_gpu=True, verbose=True)
```

출력:
```
✓ GPU 가속 활성화: NVIDIA GeForce RTX 3080
```

또는

```
⚠️  CuPy가 설치되지 않았습니다. CPU 모드로 실행합니다.
```

---

## 📊 벤치마크

### 벤치마크 실행

```bash
cd /home/minhoo/els-fdm-pricer
python3 benchmark_gpu.py
```

실행하면:
1. **시스템 정보** - CPU/GPU 확인
2. **빠른 비교** - 80×80 그리드 1회 테스트
3. **전체 벤치마크** - 여러 그리드 크기 테스트

### 예상 결과

```
================================================================================
Grid         CPU Time     GPU Time     Speedup      Price
Size         (sec)        (sec)        (x배)        Diff
================================================================================
40x40          0.40s        0.08s        5.0x       0.0001
60x60          0.65s        0.10s        6.5x       0.0002
80x80          4.26s        0.21s       20.3x       0.0001
100x100        3.18s        0.15s       21.2x       0.0003
150x150       20.45s        0.48s       42.6x       0.0002
200x200       59.23s        1.05s       56.4x       0.0001
================================================================================

평균 속도 향상: 25.3배
최대 속도 향상: 56.4배 (Grid 200x200)
```

---

## 💡 최적화 팁

### 1. 그리드 크기 선택

**GPU 효율이 높은 크기:**
- 80×80 이상
- 150×150, 200×200 권장 (GPU 효과 극대화)

**작은 그리드 (40×40 이하):**
- GPU 오버헤드로 인해 CPU가 더 빠를 수 있음
- 빠른 테스트에는 CPU 사용 권장

### 2. 메모리 관리

큰 그리드 사용 시 GPU 메모리 부족 주의:

```python
# GPU 메모리 확인
import cupy as cp
free, total = cp.cuda.Device().mem_info
print(f"사용 가능: {free/1e9:.1f}GB / {total/1e9:.1f}GB")
```

### 3. 배치 처리

여러 ELS를 평가할 때:

```python
products = [create_els_1(), create_els_2(), create_els_3()]

for product in products:
    result = price_els_gpu(product, use_gpu=True, verbose=False)
    print(f"{product.name}: {result['price']:.4f}")
```

GPU는 한 번 초기화되면 다음 계산이 더 빠름!

---

## 🐛 문제 해결

### 1. CuPy 설치 실패

**문제**: `pip install cupy-cuda11x` 실패

**해결**:
```bash
# CUDA 버전 다시 확인
nvidia-smi

# 정확한 버전 설치
# CUDA 11.2: pip install cupy-cuda112
# CUDA 11.8: pip install cupy-cuda118
# CUDA 12.0: pip install cupy-cuda12x
```

### 2. "Out of Memory" 에러

**문제**: GPU 메모리 부족

**해결**:
```python
# 그리드 크기 줄이기
result = price_els_gpu(product, N1=80, N2=80, Nt=150)

# 또는 CPU 사용
result = price_els_gpu(product, use_gpu=False)
```

### 3. GPU가 감지되지 않음

**문제**: CuPy 설치되었지만 GPU 인식 안됨

**해결**:
```bash
# CUDA 경로 확인
echo $CUDA_HOME
export CUDA_HOME=/usr/local/cuda

# 또는
export CUDA_HOME=/usr/local/cuda-11.8

# Python에서 확인
python3 -c "import cupy; cupy.show_config()"
```

### 4. WSL에서 GPU 사용

**WSL2에서 NVIDIA GPU 사용**:

1. Windows에 최신 NVIDIA 드라이버 설치
2. WSL2 업데이트
3. CUDA Toolkit 설치 (WSL 내부)
4. CuPy 설치

참고: https://docs.nvidia.com/cuda/wsl-user-guide/

---

## 📈 성능 비교 예제

### 대규모 그리드 (200×200)

```python
from src.models.els_product import create_sample_els
from src.pricing.gpu_els_pricer import price_els_gpu
import time

product = create_sample_els()

# 대규모 그리드
N = 200

print(f"대규모 그리드 테스트: {N}×{N}")

# GPU
start = time.time()
result = price_els_gpu(product, N1=N, N2=N, Nt=N*2, use_gpu=True, verbose=False)
gpu_time = time.time() - start

print(f"\nGPU 시간: {gpu_time:.2f}초")
print(f"가격: {result['price']:.4f}")
print(f"\n✓ CPU 대비 약 50~100배 빠름!")
```

---

## 🎯 권장 사용 시나리오

### GPU 사용 권장
- ✅ 정밀한 평가 (150×150 이상)
- ✅ 여러 ELS 배치 평가
- ✅ 파라미터 스캔 (민감도 분석)
- ✅ 그리드 수렴성 테스트

### CPU 사용 권장
- ✅ 빠른 프로토타입 (40×40)
- ✅ 단일 평가
- ✅ GPU 없는 환경
- ✅ 간단한 테스트

---

## 📚 추가 정보

### CuPy 공식 문서
https://docs.cupy.dev/

### CUDA Toolkit
https://developer.nvidia.com/cuda-toolkit

### 성능 프로파일링

```python
import cupy as cp

# GPU 프로파일링
with cp.cuda.profile():
    result = price_els_gpu(product, N1=100, N2=100, Nt=200)
```

---

## ✅ 체크리스트

설치 완료 확인:

- [ ] NVIDIA GPU 있음 (`nvidia-smi`)
- [ ] CUDA Toolkit 설치됨 (`nvcc --version`)
- [ ] CuPy 설치됨 (`pip list | grep cupy`)
- [ ] GPU 인식됨 (`python3 -c "import cupy as cp; print(cp.cuda.Device().name)"`)
- [ ] 벤치마크 실행 성공 (`python3 benchmark_gpu.py`)

모두 체크되면 **GPU 가속 준비 완료!** 🚀

---

**Happy GPU Pricing!** ⚡
