# ELS FDM Pricer

**Finite Difference Method (FDM) 기반 2-기초자산 Step-Down ELS 가격 평가 시스템**

---

## 📋 개요

이 프로젝트는 유한차분법(FDM)을 사용하여 2개 기초자산을 가진 Step-Down 형태의 ELS(주가연계증권)를 평가하는 시스템입니다.

### 주요 기능

- ✅ **2D Black-Scholes PDE 솔버** - ADI(Alternating Direction Implicit) 방법
- ✅ **Step-Down ELS 평가** - 조기상환, 낙인(Knock-In), Worst-of 구조
- ✅ **유연한 그리드 시스템** - 가변적인 공간/시간 해상도
- ✅ **파라미터 민감도 분석** - 변동성, 상관계수, 배리어 등
- ✅ **안정성 체크** - CFL 조건 및 수렴성 검증

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
cd /home/minhoo/els-fdm-pricer

# 의존성 설치
pip3 install -r requirements.txt
```

### 2. 기본 예제 실행

```bash
# 대화형 예제 메뉴
python3 example_pricing.py

# 또는 직접 실행
python3 -c "
from src.models.els_product import create_sample_els
from src.pricing.els_pricer import price_els

product = create_sample_els()
result = price_els(product, N1=80, N2=80, Nt=150)
print(f'ELS 가격: {result[\"price\"]:.4f}')
"
```

---

## 📂 프로젝트 구조

```
els-fdm-pricer/
├── src/
│   ├── models/
│   │   └── els_product.py          # ELS 상품 정의
│   ├── grid/
│   │   └── grid_2d.py              # 2D 그리드 생성
│   ├── solvers/
│   │   ├── fdm_solver_base.py      # FDM Solver 기본 클래스
│   │   └── adi_solver.py           # ADI Solver 구현
│   └── pricing/
│       └── els_pricer.py           # ELS 가격 평가 엔진
├── example_pricing.py              # 예제 스크립트
├── requirements.txt                # Python 의존성
└── README.md                       # 이 문서
```

---

## ⚡ GPU 가속 (선택 사항)

**CUDA를 이용하여 10~100배 빠른 계산!**

```bash
# CuPy 설치 (CUDA 버전에 맞게)
pip3 install cupy-cuda11x  # CUDA 11.x
# 또는
pip3 install cupy-cuda12x  # CUDA 12.x

# GPU로 평가
from src.pricing.gpu_els_pricer import price_els_gpu

result = price_els_gpu(product, N1=100, N2=100, Nt=200, use_gpu=True)
```

**성능 비교**:
- 40×40 그리드: CPU 0.7초 → GPU 0.1초 (7배)
- 80×80 그리드: CPU 3.7초 → GPU 0.2초 (18배)
- 150×150 그리드: CPU ~20초 → GPU ~0.5초 (40배)
- 200×200 그리드: CPU ~60초 → GPU ~1초 (60배)

**자세한 내용**: `GPU_GUIDE.md` 참고

**GPU 없이도 작동**: GPU가 없으면 자동으로 CPU 모드로 전환됩니다!

---

## 🎯 ELS 상품 정의

### Step-Down ELS 구조

```python
from src.models.els_product import ELSProduct

product = ELSProduct(
    principal=100.0,              # 원금
    maturity=3.0,                 # 만기 (3년)

    # 조기상환 조건
    observation_dates=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],  # 6개월 단위
    redemption_barriers=[0.95, 0.95, 0.90, 0.85, 0.80, 0.75],  # Step-Down
    coupons=[4.0, 8.0, 12.0, 16.0, 20.0, 24.0],  # 연 8% 쿠폰

    # 낙인 조건
    ki_barrier=0.50,              # 낙인 배리어 (50%)

    # 기초자산
    S1_0=100.0,                   # 자산 1 초기가
    S2_0=100.0,                   # 자산 2 초기가
    sigma1=0.25,                  # 변동성 1
    sigma2=0.30,                  # 변동성 2
    rho=0.50,                     # 상관계수

    # 시장 파라미터
    r=0.03,                       # 무위험이자율
    q1=0.02,                      # 배당률 1
    q2=0.015,                     # 배당률 2

    worst_of=True                 # Worst-of 구조
)
```

---

## 🧮 FDM 방법론

### ADI (Alternating Direction Implicit) Solver

2D Black-Scholes PDE를 효율적으로 풀기 위해 ADI 방법 사용:

```
∂V/∂t + 0.5σ₁²S₁²∂²V/∂S₁² + 0.5σ₂²S₂²∂²V/∂S₂²
        + ρσ₁σ₂S₁S₂∂²V/∂S₁∂S₂
        + (r-q₁)S₁∂V/∂S₁ + (r-q₂)S₂∂V/∂S₂ - rV = 0
```

**ADI 알고리즘:**
1. 각 시간 스텝을 2개 half-step으로 분할
2. Half-step 1: S₁ 방향 implicit, S₂ 방향 explicit
3. Half-step 2: S₂ 방향 implicit, S₁ 방향 explicit
4. 각 half-step에서 삼중대각 행렬만 풀면 되므로 O(N) 효율

**장점:**
- 2D 문제를 1D 문제들로 분해
- 무조건 안정적 (Implicit)
- 계산 효율 O(N₁N₂) vs 일반 implicit O(N₁²N₂²)

---

## 📊 그리드 설정

### 적응형 그리드 생성

```python
from src.grid.grid_2d import create_adaptive_grid

grid = create_adaptive_grid(
    S1_0=100.0,           # 기초자산 1 초기가
    S2_0=100.0,           # 기초자산 2 초기가
    T=3.0,                # 만기
    N1=100,               # S1 방향 그리드 수
    N2=100,               # S2 방향 그리드 수
    Nt=200,               # 시간 스텝 수
    space_factor=3.0      # 공간 범위: [0, 3*S0]
)
```

### 안정성 조건

```python
from src.grid.grid_2d import check_stability

stability = check_stability(grid, sigma1=0.25, sigma2=0.30, r=0.03)
print(f"CFL 조건: {stability['CFL_condition']}")
print(f"Explicit 안정성: {stability['is_explicit_stable']}")
```

---

## 💡 사용 예제

### 1. 기본 가격 평가

```python
from src.models.els_product import create_sample_els
from src.pricing.els_pricer import price_els

# 샘플 ELS 생성
product = create_sample_els()

# 가격 평가
result = price_els(
    product=product,
    N1=80,              # S1 그리드 수
    N2=80,              # S2 그리드 수
    Nt=150,             # 시간 스텝 수
    space_factor=3.0,   # 공간 범위
    verbose=True
)

print(f"ELS 가격: {result['price']:.4f}")
```

### 2. 파라미터 민감도 분석

```python
# 변동성 변화에 따른 가격
for sigma1 in [0.15, 0.20, 0.25, 0.30, 0.35]:
    product.sigma1 = sigma1
    result = price_els(product, N1=60, N2=60, Nt=120, verbose=False)
    print(f"σ1={sigma1:.2f}: 가격={result['price']:.4f}")

# 상관계수 변화에 따른 가격
for rho in [0.0, 0.25, 0.50, 0.75, 0.90]:
    product.rho = rho
    result = price_els(product, N1=60, N2=60, Nt=120, verbose=False)
    print(f"ρ={rho:.2f}: 가격={result['price']:.4f}")
```

### 3. 그리드 수렴성 테스트

```python
for N in [40, 60, 80, 100]:
    result = price_els(product, N1=N, N2=N, Nt=N*2, verbose=False)
    print(f"Grid {N}x{N}: 가격={result['price']:.4f}")
```

### 4. 커스텀 ELS 설계

```python
custom_els = ELSProduct(
    principal=100.0,
    maturity=2.0,  # 2년 만기
    observation_dates=[0.5, 1.0, 1.5, 2.0],
    redemption_barriers=[0.90, 0.85, 0.80, 0.75],  # 공격적
    coupons=[5.0, 10.0, 15.0, 20.0],  # 연 10%
    ki_barrier=0.45,  # 낮은 낙인
    sigma1=0.30, sigma2=0.35, rho=0.60,
    # ... 기타 파라미터
)

result = price_els(custom_els, N1=80, N2=80, Nt=160)
```

---

## 🔬 예제 스크립트

`example_pricing.py` 스크립트는 5가지 예제를 제공합니다:

1. **기본 가격 평가** - Step-Down ELS 기본 평가
2. **그리드 수렴성 테스트** - 그리드 크기별 수렴성 확인
3. **파라미터 민감도 분석** - 변동성, 상관계수, 배리어 민감도
4. **FDM 안정성 체크** - CFL 조건 및 안정성 검증
5. **커스텀 ELS 상품** - 공격적 vs 보수적 ELS 비교

```bash
python3 example_pricing.py
```

실행 후 원하는 예제를 선택하거나 전체 실행 가능.

---

## 🧪 테스트

### 기본 동작 확인

```bash
cd /home/minhoo/els-fdm-pricer

# 간단한 테스트
python3 -c "
from src.models.els_product import create_sample_els
product = create_sample_els()
print(product)
"
```

### 전체 예제 실행

```bash
# 예제 1번 실행
python3 example_pricing.py
# 선택: 1
```

---

## 📈 성능 최적화

### 그리드 크기 권장사항

| 용도 | N1 x N2 | Nt | 계산 시간 |
|------|---------|----|---------:|
| 테스트 | 40 x 40 | 80 | ~0.5초 |
| 일반 | 80 x 80 | 150 | ~2초 |
| 정밀 | 100 x 100 | 200 | ~5초 |
| 매우 정밀 | 150 x 150 | 300 | ~15초 |

### 계산 효율

- ADI 방법: O(N₁N₂Nₜ) 복잡도
- 메모리: O(N₁N₂)
- 병렬화 가능 (각 슬라이스 독립적)

---

## 🔧 확장 가능성

이 시스템은 다음과 같이 확장 가능하도록 설계되었습니다:

### 1. 다양한 ELS 구조 지원
```python
# Step-Down 외 다른 구조
# - Reverse Convertible
# - Phoenix
# - Autocallable
# - Booster
```

### 2. 추가 FDM 방법
```python
# 현재: ADI Solver
# 추가 가능:
# - Explicit Solver
# - Crank-Nicolson Solver
# - Multi-step methods
```

### 3. 3개 이상 기초자산
```python
# 3D, 4D 그리드로 확장
# Rainbow ELS, Basket ELS 지원
```

### 4. 그리스(Greeks) 계산
```python
# Delta, Gamma, Vega, Rho 등
# 유한차분법으로 계산
```

---

## 📚 참고 문헌

### FDM 방법론
- Wilmott, P. (2006). "Paul Wilmott on Quantitative Finance"
- Tavella, D., & Randall, C. (2000). "Pricing Financial Instruments: The Finite Difference Method"

### ELS 구조
- Kwok, Y.K. (2008). "Mathematical Models of Financial Derivatives"
- Haug, E.G. (2007). "The Complete Guide to Option Pricing Formulas"

### ADI 알고리즘
- Peaceman, D.W., & Rachford, H.H. (1955). "The Numerical Solution of Parabolic and Elliptic Differential Equations"
- Douglas, J., & Rachford, H.H. (1956). "On the numerical solution of heat conduction problems in two and three space variables"

---

## 🛠️ 개발 환경

- **Python**: 3.8+
- **필수 패키지**: numpy, scipy
- **선택 패키지**: matplotlib (시각화), pytest (테스트)

---

## 📝 TODO

향후 개선 계획:

- [ ] 시각화 도구 (가격 서피스, Greeks 플롯)
- [ ] Monte Carlo 방법과 비교 검증
- [ ] 멀티프로세싱 병렬화
- [ ] 3-기초자산 ELS 지원
- [ ] 실시간 시장 데이터 연동
- [ ] 웹 대시보드 인터페이스

---

## 📞 문의

프로젝트 디렉토리: `/home/minhoo/els-fdm-pricer`

---

**마지막 업데이트**: 2025-11-03
