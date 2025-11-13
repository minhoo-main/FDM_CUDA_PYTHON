# 🚀 빠른 시작 가이드 (새 환경)

다른 컴퓨터에서 이 프로젝트를 빠르게 시작하는 방법입니다.

---

## ⚡ 5분 퀵 스타트

```bash
# 1. Clone
git clone https://github.com/minhoo-main/FDM_CUDA.git
cd FDM_CUDA

# 2. 가상환경 & 설치
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. GPU 환경이면 CuPy 설치
pip install cupy-cuda12x  # 또는 cupy-cuda11x

# 4. 테스트
python3 test_optimized.py

# 5. 벤치마크 (GPU 있으면)
python3 benchmark_optimized.py
```

---

## 📋 상세 가이드

**GPU 환경 전체 가이드:** `GPU_ENVIRONMENT_GUIDE.md` 참조

**필수 파일:**
- `GPU_ENVIRONMENT_GUIDE.md` - GPU 환경 완전 가이드
- `benchmark_optimized.py` - 성능 측정
- `test_optimized.py` - 정확성 검증
- `visualize_example.py` - 시각화

---

## 🎯 Claude와 함께 작업하기

### 1. Claude에게 알려주기

새 환경에서 Claude를 실행하면:

```
안녕하세요! 저는 https://github.com/minhoo-main/FDM_CUDA 프로젝트를
~/FDM_CUDA 디렉토리에 clone했습니다.

GPU 환경에서 벤치마크를 실행하려고 합니다.
GPU_ENVIRONMENT_GUIDE.md 파일을 참조해주세요.
```

### 2. 벤치마크 결과 공유

```
벤치마크 결과입니다:

GPU: NVIDIA RTX 3080
80×80 그리드:
- CPU: 12.34s
- GPU (Original): 0.45s (27배)
- GPU (Optimized): 0.03s (411배, 15배 추가 향상)

예측이 맞았습니다!
```

### 3. 다음 작업 요청

```
이제 다음을 하고 싶습니다:
- 200×200 그리드 테스트
- 결과 보고서 작성
- 논문용 그래프 생성
```

---

## 📊 기대 결과

### 정확성 (test_optimized.py)
```
✅ Test PASSED: Prices match within 1%
```

### 성능 (benchmark_optimized.py)
```
GPU (Optimized) 🚀  0.03s    400-600x    106.6558
🎯 GPU Optimization Gain: 10-15x faster than original GPU
```

### 시각화 (visualize_example.py)
```
✓ All visualizations saved to: output/plots/
```

---

## ⚠️ 중요

**GPU 없는 환경:**
- CPU 모드로 자동 fallback
- 정확성은 동일, 속도만 느림

**GPU 있는 환경:**
- CuPy 설치 필수
- 10-100배 속도 향상 경험!

---

**상세한 설명은 GPU_ENVIRONMENT_GUIDE.md를 참조하세요!**
