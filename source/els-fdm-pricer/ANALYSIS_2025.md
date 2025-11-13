# ELS FDM Pricer 프로젝트 분석 (2025-11-03)

## 프로젝트 개요

**2개 기초자산 Step-Down ELS(주가연계증권) 가격 계산 시스템**
- 유한차분법(FDM)과 ADI(Alternating Direction Implicit) 방법 사용
- GPU 가속 지원 (CuPy 기반)
- 프로덕션 수준의 퀀트 금융 라이브러리

## 핵심 기능

### 1. ELS 상품 지원
- **Step-Down 조기상환 구조**: 6개월마다 관찰
- **Knock-In 배리어 보호**: 하방 보호 기능
- **2개 기초자산의 Worst-of 방식**: 두 자산 중 낮은 수익률 기준
- **조기상환 배리어 점진적 하락**: 예) 95% → 90% → 85% → 80% → 75%

### 2. 수학적 기법
- **2D Black-Scholes PDE 해법**
  ```
  ∂V/∂t + 0.5σ₁²S₁²∂²V/∂S₁² + 0.5σ₂²S₂²∂²V/∂S₂²
        + ρσ₁σ₂S₁S₂∂²V/∂S₁∂S₂
        + (r-q₁)S₁∂V/∂S₁ + (r-q₂)S₂∂V/∂S₂ - rV = 0
  ```
- **ADI 방법**: 무조건 안정적, O(N₁N₂Nₜ) 복잡도
- **역방향 시간 진행**: 만기 → 현재
- **상관관계 반영**: 두 자산 간 correlation 고려

### 3. GPU 가속 (선택사항)
- CuPy를 사용한 CUDA 구현
- 10~100배 속도 향상
- 자동 CPU 폴백 지원

## 기술 스택

```
Python 3.8+
├── numpy >= 1.24.0       # 핵심 계산
├── scipy >= 1.10.0       # 과학 계산
├── pandas >= 2.0.0       # 데이터 처리
├── matplotlib >= 3.7.0   # 시각화 (선택)
├── psutil >= 5.9.0       # 시스템 유틸리티
├── cupy-cuda11x/12x      # GPU 가속 (선택)
└── pytest >= 7.4.0       # 테스트
```

## 프로젝트 구조

```
els-fdm-pricer/
├── src/                          (1,525 lines)
│   ├── models/
│   │   └── els_product.py        # ELS 상품 정의 (182줄)
│   ├── grid/
│   │   └── grid_2d.py            # 2D 그리드 생성 (161줄)
│   ├── solvers/
│   │   ├── fdm_solver_base.py   # 기본 FDM 클래스 (180줄)
│   │   ├── adi_solver.py        # ADI 알고리즘 (214줄)
│   │   └── gpu_adi_solver.py    # GPU 구현 (329줄)
│   └── pricing/
│       ├── els_pricer.py        # 가격 계산 엔진 (258줄)
│       └── gpu_els_pricer.py    # GPU 가격 계산 (177줄)
│
├── example_pricing.py            # 인터랙티브 예제 (266줄)
├── benchmark_gpu.py              # GPU 벤치마크 (247줄)
├── test_gpu_simple.py            # GPU 테스트 (163줄)
│
├── README.md                     # 메인 문서 (401줄)
├── QUICK_START.md               # 빠른 시작 가이드 (346줄)
├── PROJECT_SUMMARY.md           # 프로젝트 요약 (393줄)
└── GPU_GUIDE.md                 # GPU 가이드 (368줄)
```

## 성능 벤치마크

### CPU 성능
- 40×40 그리드: ~0.7초
- 80×80 그리드: ~3.7초
- 100×100 그리드: ~8초
- 150×150 그리드: ~20초

### GPU 성능 (CUDA 지원 시)
- 80×80 그리드: ~0.2초 (18배 향상)
- 150×150 그리드: ~0.5초 (40배 향상)
- 200×200 그리드: ~1초 (60배 향상)

## 주요 알고리즘

### ADI (Alternating Direction Implicit) 방법

각 시간 스텝을 2개의 half-step으로 분할:

1. **Half-step 1**: S1 방향 implicit, S2 방향 explicit
2. **Half-step 2**: S2 방향 implicit, S1 방향 explicit

**장점:**
- 무조건 안정적 (dt 제약 없음)
- 효율적: O(N₁N₂Nₜ) vs O(N₁²N₂²) for full implicit
- 2차 정확도 (공간 및 시간)

**구현 위치:**
- CPU 버전: `src/solvers/adi_solver.py`
- GPU 버전: `src/solvers/gpu_adi_solver.py`

### 조기상환 로직

**위치**: `src/pricing/els_pricer.py` (line 122-156)

각 관찰일마다:
1. Worst-of 성과 ≥ 조기상환 배리어 체크
2. Yes → V = 원금 + 쿠폰 (즉시 상환)
3. No → V = 계속 보유 가치 (역방향 계산 지속)

### 만기 페이오프

**위치**: `src/pricing/els_pricer.py` (line 81-120)

만기 시점에서:
1. 최종 조기상환 기회 체크
2. 상환되지 않았다면:
   - Knock-In 발생: V = 원금 × min(1, worst-of 성과)
   - Knock-In 미발생: V = 원금 + 최종 쿠폰

## 사용 예시

### 기본 사용법

```python
from src.models.els_product import create_sample_els
from src.pricing.els_pricer import price_els

# ELS 상품 생성
product = create_sample_els()

# 가격 계산
result = price_els(product, N1=80, N2=80, Nt=150)
print(f"가격: {result['price']:.4f}")
```

### GPU 사용

```python
from src.pricing.gpu_els_pricer import price_els_gpu

result = price_els_gpu(
    product,
    N1=100,
    N2=100,
    Nt=200,
    use_gpu=True,
    verbose=True
)
```

### 인터랙티브 예제 실행

```bash
python3 example_pricing.py

# 옵션:
# 1. 기본 가격 계산
# 2. 그리드 수렴성 테스트
# 3. 파라미터 민감도 분석
# 4. FDM 안정성 체크
# 5. 커스텀 ELS 상품 비교
```

## 데이터 모델

### ELSProduct 클래스

```python
@dataclass
class ELSProduct:
    # 기본 정보
    principal: float = 100.0           # 원금
    maturity: float = 3.0              # 만기 (년)

    # 조기상환
    observation_dates: List[float]     # [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    redemption_barriers: List[float]   # [0.95, 0.95, 0.90, 0.85, 0.80, 0.75]
    coupons: List[float]               # [4.0, 8.0, 12.0, 16.0, 20.0, 24.0]

    # Knock-In
    ki_barrier: float = 0.50           # KI 배리어
    ki_observation_start: float = 0.0
    ki_observation_end: float = None

    # 기초자산
    S1_0: float = 100.0                # 초기 가격 1
    S2_0: float = 100.0                # 초기 가격 2
    sigma1: float = 0.25               # 변동성 1
    sigma2: float = 0.30               # 변동성 2
    rho: float = 0.50                  # 상관계수

    # 시장 파라미터
    r: float = 0.03                    # 무위험 이자율
    q1: float = 0.02                   # 배당 수익률 1
    q2: float = 0.015                  # 배당 수익률 2

    # 페이오프 타입
    worst_of: bool = True              # Worst-of vs Best-of
```

### Grid2D 클래스

```python
@dataclass
class Grid2D:
    # 공간 차원
    S1_min, S1_max: float              # 자산 1 범위
    N1: int                             # S1 그리드 포인트 수
    S2_min, S2_max: float              # 자산 2 범위
    N2: int                             # S2 그리드 포인트 수

    # 시간 차원
    T: float                            # 만기
    Nt: int                             # 시간 스텝 수

    # 생성된 속성
    S1: np.ndarray                      # S1 그리드 포인트
    S2: np.ndarray                      # S2 그리드 포인트
    S1_mesh, S2_mesh: np.ndarray       # 2D 메시 그리드
    dS1, dS2: float                     # 공간 간격
    dt: float                           # 시간 간격
    t: np.ndarray                       # 시간 그리드
```

## GPU 병렬화 분석

### ADI 방법의 병렬화 가능성

#### ✅ 공간 방향 병렬화 가능

**S1 방향 implicit 단계** (`adi_solver.py:111-132`):
```python
for j in range(N2):  # 각 S2 슬라이스
    # 각 j에 대해 독립적인 tridiagonal system
    V_new[:, j] = solve_tridiagonal(alpha1, beta1, gamma1, rhs)
```
- **N2개의 독립적인 tridiagonal systems**
- 각 S2 값에서 S1 방향으로 1D 문제 해결
- 서로 독립적 → **병렬 처리 가능**

**S2 방향 implicit 단계** (`adi_solver.py:134-155`):
```python
for i in range(N1):  # 각 S1 슬라이스
    # 각 i에 대해 독립적인 tridiagonal system
    V_new[i, :] = solve_tridiagonal(alpha2, beta2, gamma2, rhs)
```
- **N1개의 독립적인 tridiagonal systems**
- 병렬 처리 가능

#### ⚠️ 현재 GPU 구현의 한계

GPU 코드 (`gpu_adi_solver.py:168-176`):
```python
for j in range(N2):  # 여전히 순차적 루프!
    rhs = V[:, j].copy()
    V_new[:, j] = self._solve_tridiagonal_gpu(
        self.alpha1_gpu, self.beta1_gpu, self.gamma1_gpu, rhs
    )
```

**문제점:**
- 여전히 `for` loop로 하나씩 처리
- N2개의 tridiagonal systems를 **순차적으로** 풀고 있음
- 진정한 병렬화가 아님

**그럼에도 속도 향상이 있는 이유:**
1. `_solve_tridiagonal_gpu` 내부 연산이 GPU에서 실행
2. GPU 메모리 접근 속도가 빠름
3. 각 tridiagonal solve의 vector 연산이 빠름
4. 데이터 전송 오버헤드 감소

#### 🚀 개선 가능한 방법

**Batched Tridiagonal Solver** 사용:

```python
# 현재 방식 (순차적)
for j in range(N2):
    V_new[:, j] = solve_tridiagonal(...)  # 하나씩

# 이상적인 방식 (병렬)
V_new = solve_tridiagonal_batched(...)  # N2개 동시에!
```

**사용 가능한 알고리즘:**
1. **Cyclic Reduction (CR)**: O(log N) 병렬 복잡도
2. **Parallel Cyclic Reduction (PCR)**: GPU에 최적화
3. **Hybrid CR+PCR**: 큰 시스템에 효과적
4. **cuSPARSE의 gtsv2StridedBatch**: CUDA 공식 라이브러리

#### 📊 병렬화 정도 분석

```
예: 100×100 그리드
- S1 방향 implicit: 100개의 독립적 tridiagonal systems
- S2 방향 implicit: 100개의 독립적 tridiagonal systems

현재 구현: 100번 순차 실행 (각 실행은 GPU에서 빠름)
이상적 구현: 100개 동시 실행 → 이론적으로 100배 더 빠를 수 있음
```

#### 🎯 결론

**병렬화 가능성:**
1. **시간 방향**: ✗ 불가능 (역방향 순차 진행 필요)
2. **공간 방향**: ✓ 가능 (ADI의 핵심 장점!)
3. **현재 구현**: △ 부분적 병렬화 (GPU 연산만 활용)
4. **개선 여지**: ✓ Batched solver로 10~100배 추가 향상 가능

**현재 10-100배 속도 향상 요인:**
- GPU 메모리 대역폭
- Vectorized 연산
- 병렬 하드웨어 활용

**추가 개선 가능:**
- 진정한 병렬 tridiagonal solver 구현
- cuSPARSE 라이브러리 활용
- Custom CUDA kernel 작성

## 코드 품질 평가

### 장점
- ✅ 명확한 구조와 모듈 분리
- ✅ Type hints 전반적 사용
- ✅ 상세한 docstring과 주석
- ✅ 수렴성 및 안정성 테스트 포함
- ✅ 확장 가능한 아키텍처
- ✅ 우수한 문서화 (4개 마크다운 파일)
- ✅ 한글 주석으로 이해하기 쉬움

### 개선 가능한 점
- GPU batched tridiagonal solver 구현
- 단위 테스트 확충 (tests/ 디렉토리 비어있음)
- 3개 이상 기초자산 지원
- 다른 ELS 구조 (Phoenix, Reverse Convertible 등)
- 시각화 도구 추가

## 확장 가능성

아키텍처가 다음 기능 추가를 지원:
- 추가 solver 방법 (Explicit, Crank-Nicolson)
- 3D/4D 그리드 (3개 이상 기초자산)
- 다른 ELS 구조
- 시각화 도구 (가격 surface, Greeks plots)
- Monte Carlo 비교

## 전체 평가

**프로덕션 수준의 퀀트 금융 라이브러리**

이 프로젝트는 다음을 입증합니다:
1. **강력한 수학적 기반**: ADI 방법 정확한 구현
2. **실용적 사용성**: 잘 문서화되고 명확한 예제
3. **성능 최적화**: GPU 가속 옵션
4. **코드 품질**: 깨끗한 아키텍처, 적절한 문서화
5. **확장성**: 미래 개선을 위한 설계

학술 연구와 실무 금융공학 모두에 적합하며, Step-Down ELS 상품 가격 계산에 즉시 사용 가능합니다.

---

**분석 일자**: 2025-11-03
**분석자**: Claude Code Analysis
