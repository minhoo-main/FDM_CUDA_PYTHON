# 시각화 기능 설정 가이드

## 시각화 기능

ELS 가격 평가 결과를 다양한 그래프로 시각화할 수 있습니다:

1. **3D 가격 Surface** - S1, S2 평면에서 ELS 가격
2. **2D Contour Plot** - 등고선으로 가격 분포
3. **조기상환 경계면** - 각 관찰일의 조기상환 영역
4. **가격 변화** - 시간에 따른 가격 진화
5. **페이오프 비교** - 초기 가격 vs 만기 페이오프

---

## 설치 방법

### 방법 1: 시스템 패키지 (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3-matplotlib python3-tk
```

### 방법 2: pip (가상환경 사용)

```bash
# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate

# matplotlib 설치
pip install matplotlib

# 사용 후 비활성화
deactivate
```

### 방법 3: pip --break-system-packages (권장하지 않음)

```bash
pip3 install matplotlib --break-system-packages
```

---

## 사용 예제

### 기본 사용

```python
from src.models.els_product import create_sample_els
from src.pricing.els_pricer import price_els
from src.visualization.els_visualizer import ELSVisualizer

# ELS 가격 계산
product = create_sample_els()
result = price_els(product, N1=80, N2=80, Nt=150)

# 시각화 생성
visualizer = ELSVisualizer(result, output_dir="output/plots")

# 개별 그래프
visualizer.plot_price_surface_3d(save=True, show=False)
visualizer.plot_price_contour(save=True, show=False)
visualizer.plot_early_redemption_boundary(save=True, show=False)

# 또는 모든 그래프 한 번에
visualizer.plot_all(save=True, show=False)
```

### 간편 실행

```bash
# 예제 스크립트 실행
python3 visualize_example.py
```

---

## 생성되는 그래프

실행 후 `output/plots/` 디렉토리에 다음 파일들이 생성됩니다:

```
output/plots/
├── price_surface_3d.png           # 3D 가격 surface
├── price_contour.png              # 2D contour plot
├── early_redemption_boundary.png  # 조기상환 경계면
├── price_evolution.png            # 시간에 따른 가격 변화
└── payoff_comparison.png          # V_0 vs V_T 비교
```

---

## 문제 해결

### "No module named 'matplotlib'" 오류

matplotlib가 설치되지 않았습니다. 위의 설치 방법을 참조하세요.

### "_tkinter.TclError: no display name" 오류

GUI가 없는 서버 환경에서 발생합니다.

**해결 방법:**
```python
# 스크립트 최상단에 추가
import matplotlib
matplotlib.use('Agg')  # GUI 없는 백엔드 사용
```

또는:
```bash
# 환경변수 설정
export MPLBACKEND=Agg
python3 visualize_example.py
```

### 그래프가 보이지 않음

`show=False`로 설정하면 화면에 표시하지 않고 파일로만 저장됩니다.

```python
# 화면에 표시하려면
visualizer.plot_price_surface_3d(save=True, show=True)
```

---

## 시각화 예제

### 예제 1: 기본 그래프

```python
from src.models.els_product import create_sample_els
from src.pricing.els_pricer import price_els
from src.visualization.els_visualizer import ELSVisualizer

product = create_sample_els()
result = price_els(product, N1=80, N2=80, Nt=150, verbose=False)

vis = ELSVisualizer(result)
vis.plot_all(save=True, show=False)
```

### 예제 2: 특정 포인트 가격 변화

```python
# 특정 자산 가격에서 시간에 따른 변화
vis.plot_price_evolution(S1=100, S2=100, save=True, show=False)

# 다른 포인트
vis.plot_price_evolution(S1=110, S2=90, save=True, show=False)
```

### 예제 3: GPU 결과 시각화

```python
from src.pricing.gpu_els_pricer_optimized import price_els_optimized

# GPU로 가격 계산
result_gpu = price_els_optimized(product, N1=100, N2=100, Nt=200, use_gpu=True)

# 시각화
vis_gpu = ELSVisualizer(result_gpu, output_dir="output/gpu_plots")
vis_gpu.plot_all(save=True, show=False)
```

---

## 커스텀 시각화

### 커스텀 그래프 생성

```python
import matplotlib.pyplot as plt

# 결과 데이터 접근
V_0 = result['V_0']
grid = result['grid']

# 커스텀 플롯
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(V_0, cmap='viridis', origin='lower')
ax.set_xlabel('S2 index')
ax.set_ylabel('S1 index')
ax.set_title('Custom ELS Price Plot')
plt.colorbar(im, ax=ax)
plt.savefig('output/custom_plot.png', dpi=300)
```

---

## 현재 상태

- ✅ 시각화 코드 구현 완료
- ⏳ matplotlib 설치 필요
- 📁 코드 위치: `src/visualization/els_visualizer.py`
- 📄 예제: `visualize_example.py`

---

## 참고

시각화는 **선택적 기능**입니다. matplotlib 없이도 ELS 가격 계산은 정상적으로 작동합니다.
