# 🚀 Google Colab GPU 테스트 - 5분 가이드

**완전 무료로 GPU 성능을 테스트하세요!**

---

## 📋 준비물

1. ✓ Google 계정 (Gmail)
2. ✓ 프로젝트 파일: `els-fdm-pricer-colab.tar.gz` (21KB)
3. ✓ Colab 노트북: `ELS_GPU_Test_Colab.ipynb`

**위치:**
- `/home/minhoo/els-fdm-pricer-colab.tar.gz`
- `/home/minhoo/els-fdm-pricer/ELS_GPU_Test_Colab.ipynb`

---

## 🎯 단계별 가이드

### Step 1: Colab 접속 (30초)

1. 브라우저에서 접속:
   ```
   https://colab.research.google.com/
   ```

2. Google 계정으로 로그인

---

### Step 2: 노트북 업로드 (1분)

#### 방법 A: 파일 업로드 (추천)

1. Colab 메인 화면에서:
   - `File` → `Upload notebook`

2. `ELS_GPU_Test_Colab.ipynb` 파일 선택

3. 업로드 완료!

#### 방법 B: GitHub에서 (나중에)

나중에 GitHub에 올리면 URL로 바로 열 수 있습니다.

---

### Step 3: GPU 활성화 (30초) ⚠️ 중요!

**반드시 GPU를 활성화해야 합니다!**

1. 메뉴: `Runtime` → `Change runtime type`

2. Hardware accelerator: **GPU** 선택

3. GPU type: **T4** (무료는 자동 선택됨)

4. `Save` 클릭

✓ 확인: 우측 상단에 "Connected" 표시되고 GPU 아이콘이 보이면 성공!

---

### Step 4: 셀 실행 (3분)

이제 노트북의 셀들을 순서대로 실행하면 됩니다!

#### 📦 셀 1: 프로젝트 파일 업로드

```python
from google.colab import files
uploaded = files.upload()
```

실행하면 **파일 선택 버튼**이 나타납니다.
→ `els-fdm-pricer-colab.tar.gz` 선택!

#### 🔧 셀 2: 패키지 설치

```python
!pip install -q cupy-cuda12x
```

자동으로 CuPy와 필요한 패키지를 설치합니다.
→ 1-2분 소요

#### 🎮 셀 3: GPU 확인

```python
!nvidia-smi
```

GPU 정보가 출력됩니다!

**예상 출력:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.x   |
|-------------------------------+----------------------+----------------------+
|   0  Tesla T4            On   | 00000000:00:04.0 Off |                    0 |
| N/A   xx°C    P8    xx W / 70W |      0MiB / 15360MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

✓ Tesla T4 (16GB) 또는 P100 받으면 좋음!

#### 🧪 셀 4-6: 테스트 실행

나머지 셀들을 순서대로 실행하면:

**셀 4:** 기본 테스트 (100×100)
**셀 5:** CPU vs GPU 벤치마크 ⭐
**셀 6:** 대규모 그리드 (200×200×1000) ⭐⭐⭐

각 셀을 클릭하고 `Shift + Enter`로 실행하거나
상단 메뉴: `Runtime` → `Run all`로 전체 실행!

---

## 🎉 예상 결과

### GPU 정보
```
✓ GPU: Tesla T4 (16GB)
✓ CUDA 사용 가능
```

### 성능 비교 (예상)
```
Grid Size      CPU Time    GPU Time    Speedup
────────────────────────────────────────────────
80×80×160      3.7초       0.2초       18배
100×100×200    8.0초       0.5초       16배
150×150×300    20초        0.5초       40배
200×200×400    35초        1.0초       35배
```

### 대규모 그리드 (200×200×1000)
```
✓ CPU: 78초 (측정값)
✓ GPU: ~1.3초 (예상)
✓ 속도 향상: ~60배!
```

---

## 💡 팁

### 1. 셀 실행 방법
- **한 셀만**: 클릭 후 `Shift + Enter`
- **전체 실행**: `Runtime` → `Run all`
- **중단**: `Runtime` → `Interrupt execution`

### 2. 오류 발생 시

**"GPU not found"**
- 해결: Runtime type이 GPU로 설정되었는지 확인
- `Runtime` → `Change runtime type` → GPU

**"CuPy import error"**
- 해결: 패키지 설치 셀을 다시 실행
- CUDA 버전 확인 (`!nvcc --version`)

**"파일 업로드 실패"**
- 해결: 파일 경로 확인
- 브라우저 새로고침 후 재시도

### 3. 세션 제한
- **무료**: 12시간/세션
- **Idle**: 90분 미사용 시 연결 해제
- **해결**: 주기적으로 셀 실행

### 4. 결과 저장
마지막 셀로 결과를 CSV로 다운로드:
```python
files.download('gpu_benchmark_results.csv')
```

---

## 📊 시각화

노트북에는 자동으로 그래프가 생성됩니다:

1. **CPU vs GPU 시간 비교** (막대 그래프)
2. **속도 향상** (막대 그래프)

---

## 🚨 주의사항

### GPU 할당 실패?

Google Colab 무료 버전은 GPU 할당이 보장되지 않습니다.

**증상:**
- GPU 버튼이 회색
- "GPU backend not available"

**해결:**
1. 시간대를 바꿔서 재시도 (미국 시간대 업무시간 피하기)
2. 몇 시간 후 재시도
3. Kaggle Notebooks 이용 (대안)

### 사용량 제한

무료 버전은 주당 GPU 사용량 제한이 있습니다.

**권장:**
- 필요한 테스트만 실행
- 완료 후 런타임 종료: `Runtime` → `Disconnect and delete runtime`

---

## 🎯 테스트 체크리스트

```
□ Colab 접속
□ 노트북 업로드
□ GPU 활성화 (중요!)
□ 프로젝트 파일 업로드
□ 패키지 설치
□ GPU 확인 (nvidia-smi)
□ 기본 테스트 실행
□ CPU vs GPU 벤치마크
□ 대규모 그리드 테스트
□ 결과 확인 및 저장
```

---

## 🔄 빠른 재실행

다음번에는 더 빠릅니다:

1. Colab에 이전 노트북이 저장되어 있음
2. `File` → `Open notebook` → `Recent` 탭
3. 이전 노트북 열기
4. GPU 활성화만 다시 하면 됨!
5. `Runtime` → `Run all`

---

## 📞 도움말

### GPU 타입 확인
```python
import cupy as cp
print(cp.cuda.Device().name)
```

**가능한 GPU:**
- **Tesla T4** (16GB) - 보통, 좋음
- **Tesla P100** (16GB) - 빠름, 매우 좋음
- **Tesla V100** (16GB) - 매우 빠름, 최고! (운 좋으면)

### 메모리 부족?
```python
# 그리드 크기 줄이기
N1, N2, Nt = 150, 150, 300  # 대신 200×200×1000
```

### 세션 유지
```python
# 주기적으로 실행 (90분 idle 방지)
import time
while True:
    print(".", end="", flush=True)
    time.sleep(60)  # 1분마다
```

---

## 🎉 완료 후

테스트가 성공했다면:

### ✓ 확인된 사항
1. GPU 60배 빠름 (기본)
2. 1초 이내 가능 (200×200×1000)
3. 실시간 프라이싱 가능성 확인

### 📝 다음 단계
1. GPU 최적화 구현
   - Batched tridiagonal solver
   - Vectorized callbacks
   - 예상: 추가 10-15배 향상

2. 최종 목표
   - < 0.1초 (실시간)
   - 배치 프라이싱 효율화

---

## 🌟 Colab Pro?

더 많이 사용하고 싶다면:

**Colab Pro** ($10/월):
- 24시간 세션
- 더 빠른 GPU (우선 할당)
- 더 많은 메모리
- 우선 접속

**Colab Pro+** ($50/월):
- 백그라운드 실행
- V100/A100 GPU 가능
- 더 긴 세션

하지만 **무료로도 충분합니다!**

---

## 📚 추가 자료

- [Colab 공식 가이드](https://colab.research.google.com/notebooks/intro.ipynb)
- [CuPy 문서](https://docs.cupy.dev/)
- [프로젝트 README](README.md)

---

**준비됐나요? 시작하세요! 🚀**

5분이면 GPU 성능을 직접 확인할 수 있습니다!
