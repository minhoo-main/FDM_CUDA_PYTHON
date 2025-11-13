# Google Colab에서 ELS Pricer 실행하기

무료 GPU로 성능 테스트하는 가장 쉬운 방법!

## 🚀 빠른 시작

### 1. Google Colab 접속
https://colab.research.google.com/

### 2. 새 노트북 생성
- `File` → `New notebook`

### 3. GPU 활성화
- `Runtime` → `Change runtime type`
- `Hardware accelerator`: **GPU** 선택
- `Save`

### 4. 코드 실행

**셀 1: 프로젝트 설치**
```python
# 프로젝트 다운로드 (GitHub에 있다면)
# !git clone https://github.com/your-repo/els-fdm-pricer.git
# %cd els-fdm-pricer

# 또는 직접 업로드
from google.colab import files
import zipfile

# 왼쪽 파일 탭에서 업로드 버튼으로 프로젝트 zip 업로드
# uploaded = files.upload()
# !unzip els-fdm-pricer.zip
# %cd els-fdm-pricer

# CuPy 설치 (Colab은 CUDA 11.8)
!pip install cupy-cuda11x -q

# 필요한 패키지
!pip install -r requirements.txt -q
```

**셀 2: GPU 확인**
```python
import cupy as cp

print(f"✓ GPU 사용 가능: {cp.cuda.is_available()}")
if cp.cuda.is_available():
    print(f"✓ GPU: {cp.cuda.Device().name}")
    print(f"✓ GPU 메모리: {cp.cuda.Device().mem_info[1] / 1024**3:.1f} GB")
```

**셀 3: 기본 테스트**
```python
from src.models.els_product import create_sample_els
from src.pricing.gpu_els_pricer import price_els_gpu

# 샘플 ELS 생성
product = create_sample_els()

# GPU로 가격 평가
result = price_els_gpu(
    product,
    N1=100,
    N2=100,
    Nt=200,
    use_gpu=True,
    verbose=True
)

print(f"\n✓ ELS 가격: {result['price']:.4f}")
print(f"✓ 계산 시간: {result['computation_time']:.3f}초")
```

**셀 4: GPU vs CPU 벤치마크**
```python
import time
from src.pricing.els_pricer import price_els

# CPU 버전
start = time.time()
result_cpu = price_els(product, N1=80, N2=80, Nt=160, verbose=False)
time_cpu = time.time() - start

# GPU 버전
start = time.time()
result_gpu = price_els_gpu(product, N1=80, N2=80, Nt=160, use_gpu=True, verbose=False)
time_gpu = time.time() - start

# 비교
speedup = time_cpu / time_gpu
print(f"CPU 시간: {time_cpu:.2f}초")
print(f"GPU 시간: {time_gpu:.2f}초")
print(f"속도 향상: {speedup:.1f}배")
print(f"가격 차이: {abs(result_cpu['price'] - result_gpu['price']):.6f}")
```

**셀 5: 시각화 (선택)**
```python
import matplotlib.pyplot as plt
import numpy as np

# 그리드 크기별 성능 비교
sizes = [40, 60, 80, 100]
times_cpu = []
times_gpu = []

for N in sizes:
    # CPU
    start = time.time()
    price_els(product, N1=N, N2=N, Nt=N*2, verbose=False)
    times_cpu.append(time.time() - start)

    # GPU
    start = time.time()
    price_els_gpu(product, N1=N, N2=N, Nt=N*2, use_gpu=True, verbose=False)
    times_gpu.append(time.time() - start)

# 플롯
plt.figure(figsize=(10, 6))
plt.plot(sizes, times_cpu, 'o-', label='CPU', linewidth=2)
plt.plot(sizes, times_gpu, 's-', label='GPU', linewidth=2)
plt.xlabel('Grid Size (N×N)')
plt.ylabel('Time (seconds)')
plt.title('GPU vs CPU Performance')
plt.legend()
plt.grid(True)
plt.show()

# 속도 향상
speedups = np.array(times_cpu) / np.array(times_gpu)
plt.figure(figsize=(10, 6))
plt.bar(sizes, speedups)
plt.xlabel('Grid Size (N×N)')
plt.ylabel('Speedup (×)')
plt.title('GPU Speedup over CPU')
plt.grid(True, axis='y')
plt.show()
```

## 💡 Colab 팁

### GPU 세션 제한
- 무료: 12시간/세션, 주당 제한 있음
- Colab Pro ($10/월): 24시간/세션, 더 나은 GPU

### GPU 타입 확인
```python
!nvidia-smi
```

일반적으로 받는 GPU:
- **Tesla T4** (16GB, 보통)
- **Tesla P100** (16GB, 빠름)
- **Tesla V100** (16GB, 매우 빠름) - 운이 좋으면

### 파일 업로드/다운로드
```python
from google.colab import files

# 업로드
uploaded = files.upload()

# 다운로드
files.download('results.csv')
```

### Google Drive 연동
```python
from google.colab import drive
drive.mount('/content/drive')

# 프로젝트를 Drive에 저장
!cp -r /content/els-fdm-pricer /content/drive/MyDrive/
```

## 🎯 예상 성능 (Google Colab T4)

| Grid Size | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| 40×40     | 0.7초    | 0.1초    | 7배     |
| 80×80     | 3.7초    | 0.2초    | 18배    |
| 100×100   | 8초      | 0.5초    | 16배    |
| 150×150   | 20초     | 1초      | 20배    |
| 200×200   | 60초     | 2초      | 30배    |

## ⚠️ 주의사항

1. **세션 끊김**: 90분 idle 시 연결 해제
   - 해결: 주기적으로 셀 실행

2. **GPU 할당 실패**: 사용량 많으면 GPU 못 받을 수 있음
   - 해결: 시간대 바꿔서 재시도

3. **파일 휘발성**: 세션 종료 시 파일 사라짐
   - 해결: Google Drive 연동

## 📚 참고

- [Colab 공식 가이드](https://colab.research.google.com/notebooks/intro.ipynb)
- [CuPy 공식 문서](https://docs.cupy.dev/)
