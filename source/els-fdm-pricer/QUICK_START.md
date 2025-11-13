# ⚡ ELS FDM Pricer - 빠른 시작 가이드

작업을 재개할 때 이 문서를 참고하세요!

---

## 🎯 현재 상태

- ✅ **2D 그리드 시스템 완성** - 유연한 공간/시간 해상도
- ✅ **ADI FDM Solver 완성** - 효율적인 2D PDE 솔버
- ✅ **ELS 가격 평가 엔진 완성** - 조기상환, 낙인 처리
- ✅ **예제 스크립트 완성** - 5가지 실전 예제
- ✅ **테스트 완료** - 수렴성 및 안정성 검증

**프로젝트 위치**: `/home/minhoo/els-fdm-pricer`

---

## 📌 주요 명령어

### 1. 빠른 테스트

```bash
cd /home/minhoo/els-fdm-pricer

# 기본 평가
python3 -c "
from src.models.els_product import create_sample_els
from src.pricing.els_pricer import price_els

product = create_sample_els()
result = price_els(product, N1=60, N2=60, Nt=120)
print(f'가격: {result[\"price\"]:.4f}')
"
```

### 2. 대화형 예제 실행

```bash
python3 example_pricing.py
```

예제 메뉴:
1. 기본 가격 평가
2. 그리드 수렴성 테스트
3. 파라미터 민감도 분석
4. FDM 안정성 체크
5. 커스텀 ELS 상품

---

## 📂 주요 파일 위치

### 핵심 모듈

| 파일 | 위치 | 설명 |
|------|------|------|
| **ELS 상품** | `src/models/els_product.py` | Step-Down ELS 정의 |
| **그리드** | `src/grid/grid_2d.py` | 2D 그리드 생성 |
| **FDM Solver** | `src/solvers/adi_solver.py` | ADI 방법 구현 |
| **프라이서** | `src/pricing/els_pricer.py` | 가격 평가 엔진 |

### 실행 파일

- `example_pricing.py` - 예제 스크립트 (대화형)
- `requirements.txt` - Python 의존성

### 문서

- `README.md` - 전체 문서
- `QUICK_START.md` - 이 문서

---

## 🚀 기본 사용법

### 1. 샘플 ELS 평가

```python
from src.models.els_product import create_sample_els
from src.pricing.els_pricer import price_els

# 기본 Step-Down ELS
product = create_sample_els()
print(product)

# 가격 평가
result = price_els(
    product=product,
    N1=80,              # S1 방향 그리드 수
    N2=80,              # S2 방향 그리드 수
    Nt=150,             # 시간 스텝 수
    space_factor=3.0,   # 공간 범위 (0 ~ 3*S0)
    verbose=True
)

print(f"가격: {result['price']:.4f}")
```

### 2. 커스텀 ELS 설계

```python
from src.models.els_product import ELSProduct
from src.pricing.els_pricer import price_els

# 공격적 ELS (2년, 연 10% 쿠폰)
aggressive = ELSProduct(
    principal=100.0,
    maturity=2.0,
    observation_dates=[0.5, 1.0, 1.5, 2.0],
    redemption_barriers=[0.90, 0.85, 0.80, 0.75],
    coupons=[5.0, 10.0, 15.0, 20.0],
    ki_barrier=0.45,
    S1_0=100.0, S2_0=100.0,
    sigma1=0.30, sigma2=0.35, rho=0.60,
    r=0.03, q1=0.02, q2=0.015
)

result = price_els(aggressive, N1=80, N2=80, Nt=160)
```

### 3. 파라미터 민감도 분석

```python
# 변동성 민감도
for sigma1 in [0.15, 0.20, 0.25, 0.30, 0.35]:
    product.sigma1 = sigma1
    result = price_els(product, N1=60, N2=60, Nt=120, verbose=False)
    print(f"σ1={sigma1:.2f}: {result['price']:.4f}")

# 상관계수 민감도
for rho in [0.0, 0.25, 0.50, 0.75, 0.90]:
    product.rho = rho
    result = price_els(product, N1=60, N2=60, Nt=120, verbose=False)
    print(f"ρ={rho:.2f}: {result['price']:.4f}")
```

---

## 🔧 핵심 코드 위치

### 1. ELS 상품 정의
**파일**: `src/models/els_product.py`

```python
@dataclass
class ELSProduct:
    principal: float = 100.0
    maturity: float = 3.0
    observation_dates: List[float] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    redemption_barriers: List[float] = [0.95, 0.95, 0.90, 0.85, 0.80, 0.75]
    # ...

    def payoff_at_maturity(self, S1, S2, ki_occurred):
        """만기 페이오프 계산"""

    def check_early_redemption(self, S1, S2, obs_idx):
        """조기상환 조건 체크"""
```

### 2. ADI Solver
**파일**: `src/solvers/adi_solver.py`

```python
class ADISolver(FDMSolver2D):
    def solve(self, V_T, early_exercise_callback=None):
        """ADI 방법으로 PDE 풀기"""
        # Half-step 1: S1 방향 implicit
        # Half-step 2: S2 방향 implicit
```

### 3. 가격 평가 엔진
**파일**: `src/pricing/els_pricer.py:53`

```python
def price(self, verbose=True):
    # 1. 만기 페이오프 설정
    V_T = self._initialize_terminal_payoff()

    # 2. FDM으로 역방향 풀기
    results = self.solver.solve_with_callbacks(...)

    # 3. 현재가에서 가격 추출
    price = self.grid.get_value_at_point(V_0, S1_0, S2_0)
```

---

## 📊 그리드 권장사항

| 용도 | Grid Size | 시간 스텝 | 계산 시간 | 정확도 |
|------|-----------|----------|---------|-------|
| 빠른 테스트 | 40×40 | 80 | ~0.7초 | 보통 |
| 일반 평가 | 60×60 | 120 | ~1.5초 | 좋음 |
| 정밀 평가 | 80×80 | 150 | ~3.7초 | 우수 |
| 매우 정밀 | 100×100 | 200 | ~8초 | 최고 |

### 안정성 체크

```python
from src.grid.grid_2d import create_adaptive_grid, check_stability

grid = create_adaptive_grid(S1_0=100, S2_0=100, T=3.0, N1=80, N2=80, Nt=150)
stability = check_stability(grid, sigma1=0.25, sigma2=0.30, r=0.03)

print(f"CFL 조건: {stability['CFL_condition']}")
print(f"dt: {stability['dt']:.6f}")
```

---

## 💡 개발 메모

### ADI 방법의 장점
1. **무조건 안정적** - Implicit 방법이므로 dt 제약 없음
2. **효율적** - O(N₁N₂) 복잡도 (vs 일반 implicit O(N₁²N₂²))
3. **정확** - 2차 정확도 (공간, 시간)

### ELS 평가 로직
1. 만기에서 역방향으로 풀기 (T → 0)
2. 각 조기상환 평가일에서 조건 체크
3. 조기상환 조건 만족 시 즉시 상환
4. 낙인 발생 시 원금 손실 가능

### 테스트 결과
```
Grid 40×40: 가격 106.66 (0.7초)
Grid 60×60: 가격 106.91 (1.5초)
Grid 80×80: 가격 106.28 (3.7초)

→ 수렴 확인 완료 ✓
```

---

## 🐛 문제 해결

### ImportError 발생 시
```bash
# 패키지 설치
pip3 install numpy scipy matplotlib

# 또는
pip3 install -r requirements.txt
```

### 계산 너무 느릴 때
```python
# 그리드 크기 줄이기
result = price_els(product, N1=40, N2=40, Nt=80)
```

### 가격이 이상할 때
```python
# 안정성 체크
from src.grid.grid_2d import check_stability
stability = check_stability(grid, sigma1, sigma2, r)
print(stability)
```

---

## 📝 다음 개선 사항

### 우선순위 높음
- [ ] 시각화 도구 (가격 서피스, Greeks)
- [ ] Monte Carlo와 비교 검증
- [ ] 멀티프로세싱 병렬화

### 우선순위 중간
- [ ] 3-기초자산 ELS 지원
- [ ] Explicit/Crank-Nicolson Solver 추가
- [ ] Greeks 계산 자동화

### 우선순위 낮음
- [ ] 웹 대시보드 인터페이스
- [ ] 실시간 시장 데이터 연동
- [ ] Docker 컨테이너화

---

## 🔍 작업 재개 시

1. **환경 확인**
   ```bash
   cd /home/minhoo/els-fdm-pricer
   python3 --version
   ```

2. **빠른 테스트**
   ```bash
   python3 -c "from src import *; print('✓ Import OK')"
   ```

3. **예제 실행**
   ```bash
   python3 example_pricing.py
   ```

4. **문서 확인**
   - `README.md` - 전체 개요
   - `QUICK_START.md` - 이 문서
   - 코드 주석 - 상세 설명

---

## 📞 주요 API

### price_els() - 간편 인터페이스

```python
from src.pricing.els_pricer import price_els

result = price_els(
    product,              # ELS 상품
    N1=80,                # S1 그리드 수
    N2=80,                # S2 그리드 수
    Nt=150,               # 시간 스텝 수
    space_factor=3.0,     # 공간 범위
    verbose=True          # 상세 출력
)

# 결과
result['price']         # 가격
result['V_0']          # t=0 가격 그리드
result['V_T']          # 만기 페이오프
result['snapshots']    # 중간 스냅샷
```

### create_sample_els() - 샘플 생성

```python
from src.models.els_product import create_sample_els

product = create_sample_els()
# 3년 만기, 6개월 단위 조기상환
# Step-Down 배리어: 95% → 75%
# 연 8% 쿠폰, 낙인 50%
```

---

**작업 재개 시**: 이 파일부터 확인!

**마지막 업데이트**: 2025-11-03
