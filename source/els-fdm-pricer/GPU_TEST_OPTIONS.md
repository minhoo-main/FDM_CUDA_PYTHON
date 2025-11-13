# GPU 없이 테스트하는 모든 방법

GPU가 없어도 프로젝트를 테스트하고 성능을 측정할 수 있는 방법들

---

## 📊 방법 비교표

| 방법 | 비용 | 설정 난이도 | GPU 성능 | 세션 시간 | 추천도 |
|------|------|-------------|----------|-----------|--------|
| **로컬 CPU** | 무료 | ⭐ 쉬움 | ❌ 없음 | 무제한 | ⭐⭐⭐ 개발용 |
| **Google Colab** | 무료 | ⭐⭐ 쉬움 | ✅ T4/P100 | 12시간 | ⭐⭐⭐⭐⭐ 최고 |
| **Kaggle** | 무료 | ⭐⭐ 쉬움 | ✅ P100/T4 | 9시간 | ⭐⭐⭐⭐ 좋음 |
| **Paperspace** | $8/월~ | ⭐⭐⭐ 보통 | ✅ 다양 | 6시간~ | ⭐⭐⭐ 괜찮음 |
| **AWS EC2** | $0.5/시간~ | ⭐⭐⭐⭐ 어려움 | ✅ 최고 | 무제한 | ⭐⭐ 프로덕션 |

---

## 1️⃣ 로컬 CPU 모드 (현재 환경)

**장점:**
- ✓ 추가 설정 불필요
- ✓ 개발/디버깅에 최적
- ✓ 무제한 시간

**사용법:**
```bash
cd ~/els-fdm-pricer

# 기본 테스트
python3 example_pricing.py

# 벤치마크
python3 benchmark_gpu.py  # GPU 없어도 CPU 모드로 실행
```

**성능:**
- 40×40: ~0.7초
- 80×80: ~3.7초
- 100×100: ~8초

---

## 2️⃣ Google Colab (⭐ 최고 추천!)

**장점:**
- ✓ 완전 무료
- ✓ 설정 초간단 (2분)
- ✓ T4/P100 GPU 제공
- ✓ Jupyter 노트북 환경

**단점:**
- △ 12시간/세션 제한
- △ 90분 idle 시 연결 해제
- △ 파일 휘발성

**사용법:**
👉 **`COLAB_SETUP.md` 참고**

**빠른 시작:**
1. https://colab.research.google.com/
2. Runtime → Change runtime type → GPU
3. 코드 실행!

**예상 성능 (T4 GPU):**
- 40×40: ~0.1초 (7배 빠름)
- 80×80: ~0.2초 (18배 빠름)
- 100×100: ~0.5초 (16배 빠름)
- 200×200: ~2초 (30배 빠름)

---

## 3️⃣ Kaggle Notebooks

**장점:**
- ✓ 완전 무료
- ✓ P100 GPU (Colab보다 빠를 수 있음)
- ✓ 주당 30시간 GPU

**단점:**
- △ 9시간/세션 제한
- △ 인터넷 접근 제한 (일부 pip install 안됨)

**사용법:**

1. **Kaggle 가입**
   - https://www.kaggle.com/

2. **새 노트북 생성**
   - `Code` → `New Notebook`
   - Settings → Accelerator: **GPU T4 x2** 선택

3. **프로젝트 업로드**
   ```python
   # 파일 업로드 (왼쪽 데이터 탭)
   # 또는 GitHub에서 clone
   !git clone https://github.com/your-repo/els-fdm-pricer.git
   %cd els-fdm-pricer

   # CuPy 설치
   !pip install cupy-cuda11x -q
   !pip install -r requirements.txt -q
   ```

4. **실행**
   ```python
   from src.models.els_product import create_sample_els
   from src.pricing.gpu_els_pricer import price_els_gpu

   product = create_sample_els()
   result = price_els_gpu(product, N1=100, N2=100, Nt=200, use_gpu=True)
   print(f"가격: {result['price']:.4f}")
   ```

**GPU 타입:**
- Tesla P100 (16GB) - 주로 할당됨
- Tesla T4 x2 (16GB × 2) - 운 좋으면

---

## 4️⃣ Paperspace Gradient

**장점:**
- ✓ 무료 티어 있음 (제한적)
- ✓ 다양한 GPU 옵션
- ✓ Jupyter 환경

**단점:**
- △ 무료: 6시간/세션
- △ 대기 시간 있을 수 있음

**사용법:**

1. **가입**
   - https://www.paperspace.com/gradient
   - 무료 계정 생성

2. **노트북 생성**
   - Create → Notebook
   - Runtime: **Free-GPU** (또는 유료 옵션)

3. **설정 및 실행**
   ```bash
   # 터미널
   git clone your-repo
   cd els-fdm-pricer
   pip install cupy-cuda11x
   pip install -r requirements.txt

   # Jupyter에서
   python3 example_pricing.py
   ```

**가격 (유료 옵션):**
- Free-GPU: 무료 (제한적)
- P4000: $0.51/hour
- RTX4000: $0.76/hour
- V100: $2.30/hour

---

## 5️⃣ AWS EC2 (프로덕션용)

**장점:**
- ✓ 최고 성능
- ✓ 무제한 시간
- ✓ 다양한 GPU (T4, V100, A100)
- ✓ 프로덕션 배포 가능

**단점:**
- ✗ 비쌈
- ✗ 설정 복잡
- ✗ AWS 계정 필요

**사용법:**

1. **인스턴스 생성**
   ```
   EC2 → Launch Instance
   - AMI: Deep Learning AMI (Ubuntu)
   - Instance Type: g4dn.xlarge (T4, $0.526/hr)
   - Storage: 50GB
   ```

2. **접속 및 설정**
   ```bash
   ssh -i key.pem ubuntu@ec2-ip

   # CUDA 확인
   nvidia-smi

   # 프로젝트 설정
   git clone your-repo
   cd els-fdm-pricer
   conda create -n els python=3.9
   conda activate els
   pip install cupy-cuda11x
   pip install -r requirements.txt
   ```

3. **실행**
   ```bash
   python3 benchmark_gpu.py
   ```

**가격 (주요 인스턴스):**
| 타입 | GPU | vCPU | RAM | 가격/시간 |
|------|-----|------|-----|----------|
| g4dn.xlarge | T4 | 4 | 16GB | $0.526 |
| p3.2xlarge | V100 | 8 | 61GB | $3.06 |
| p4d.24xlarge | A100×8 | 96 | 1.1TB | $32.77 |

---

## 6️⃣ GCP / Azure (AWS 대안)

### Google Cloud Platform

```bash
# VM 생성 (gcloud CLI)
gcloud compute instances create gpu-instance \
  --zone=us-west1-b \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release

# 접속
gcloud compute ssh gpu-instance
```

### Microsoft Azure

```bash
# NC 시리즈 (Tesla T4/V100)
az vm create \
  --resource-group myResourceGroup \
  --name myGPUVM \
  --size Standard_NC6s_v3 \
  --image microsoft-dsvm:ubuntu-1804:1804-gen2:latest
```

---

## 🎯 추천 워크플로우

### 개발 단계
```
1. 로컬 CPU로 개발/디버깅 (무료, 무제한)
2. Google Colab으로 GPU 검증 (무료, 12시간)
3. 필요시 Kaggle로 추가 테스트 (무료, 9시간)
```

### 프로덕션 단계
```
1. AWS/GCP에 배포
2. 로드밸런서 + Auto Scaling
3. 비용 최적화 (Spot 인스턴스 활용)
```

---

## 💡 각 상황별 추천

| 상황 | 추천 방법 |
|------|----------|
| **처음 테스트** | Google Colab |
| **장시간 실험** | Kaggle (30시간/주) |
| **개발/디버깅** | 로컬 CPU |
| **프로덕션 배포** | AWS EC2 / GCP |
| **예산 제한** | Colab → Kaggle 번갈아 |
| **최고 성능** | AWS p4d (A100×8) |

---

## 🚀 지금 바로 시작하기

**가장 빠른 방법 (5분):**

1. https://colab.research.google.com/ 접속
2. 새 노트북 생성
3. Runtime → Change runtime type → GPU
4. 첫 셀에 붙여넣기:
   ```python
   # 프로젝트 준비 (GitHub 업로드 후)
   !git clone https://github.com/your-repo/els-fdm-pricer.git
   %cd els-fdm-pricer
   !pip install cupy-cuda11x -q

   # 테스트
   from src.models.els_product import create_sample_els
   from src.pricing.gpu_els_pricer import price_els_gpu

   product = create_sample_els()
   result = price_els_gpu(product, N1=100, N2=100, Nt=200, use_gpu=True)
   print(f"✓ ELS 가격: {result['price']:.4f}")
   ```
5. 실행! (Shift+Enter)

---

## 📞 문제 해결

### CuPy 설치 실패
```bash
# CUDA 버전 확인
nvidia-smi

# CUDA 11.x
pip install cupy-cuda11x

# CUDA 12.x
pip install cupy-cuda12x
```

### GPU 인식 안됨
```python
import cupy as cp
print(cp.cuda.is_available())  # True여야 함

# False면:
# 1. Runtime type이 GPU인지 확인
# 2. nvidia-smi로 GPU 확인
# 3. CuPy 재설치
```

### 메모리 부족
```python
# 그리드 크기 줄이기
result = price_els_gpu(product, N1=80, N2=80, Nt=160)  # 대신 100×100

# 또는 GPU 메모리 정리
import cupy as cp
cp.get_default_memory_pool().free_all_blocks()
```

---

**Happy Testing! 🎉**
