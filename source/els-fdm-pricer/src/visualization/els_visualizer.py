"""
ELS 가격 평가 결과 시각화

다양한 시각화 기능:
1. 가격 surface plot (3D)
2. 가격 contour plot (2D)
3. 조기상환 경계면
4. 시간에 따른 가격 변화
5. Greeks (Delta, Gamma, Vega)
6. 그리드 수렴성 분석
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Optional, List
import os


class ELSVisualizer:
    """ELS 가격 평가 결과 시각화 클래스"""

    def __init__(self, result: Dict, output_dir: str = "output"):
        """
        Args:
            result: price_els 또는 price_els_gpu의 결과 딕셔너리
            output_dir: 그래프 저장 디렉토리
        """
        self.result = result
        self.output_dir = output_dir

        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        # 결과 추출
        self.price = result['price']
        self.V_0 = result['V_0']
        self.V_T = result['V_T']
        self.grid = result['grid']
        self.product = result['product']
        self.snapshots = result.get('snapshots', {})
        self.redemption_flags = result.get('redemption_flags', {})

    def plot_price_surface_3d(self, save: bool = True, show: bool = True):
        """
        3D 가격 surface plot

        S1, S2 평면에서 ELS 가격을 3D로 표시
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Mesh grid
        S1_mesh = self.grid.S1_mesh
        S2_mesh = self.grid.S2_mesh
        V = self.V_0

        # Surface plot
        surf = ax.plot_surface(S1_mesh, S2_mesh, V,
                               cmap=cm.viridis,
                               alpha=0.8,
                               linewidth=0,
                               antialiased=True)

        # 초기 포인트 표시
        S1_0, S2_0 = self.product.S1_0, self.product.S2_0
        ax.scatter([S1_0], [S2_0], [self.price],
                  color='red', s=100, marker='o',
                  label=f'Current: {self.price:.2f}')

        ax.set_xlabel('S1 (Asset 1)', fontsize=10)
        ax.set_ylabel('S2 (Asset 2)', fontsize=10)
        ax.set_zlabel('ELS Price', fontsize=10)
        ax.set_title('ELS Price Surface at t=0', fontsize=14, fontweight='bold')

        # Colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        ax.legend()

        if save:
            filepath = os.path.join(self.output_dir, 'price_surface_3d.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_price_contour(self, save: bool = True, show: bool = True):
        """
        2D 가격 contour plot

        등고선으로 가격 분포 표시
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        S1_mesh = self.grid.S1_mesh
        S2_mesh = self.grid.S2_mesh
        V = self.V_0

        # Contour plot
        levels = 20
        contour = ax.contourf(S1_mesh, S2_mesh, V,
                             levels=levels,
                             cmap='viridis')

        # Contour lines
        contour_lines = ax.contour(S1_mesh, S2_mesh, V,
                                   levels=levels,
                                   colors='black',
                                   alpha=0.3,
                                   linewidths=0.5)
        ax.clabel(contour_lines, inline=True, fontsize=8)

        # 초기 포인트
        S1_0, S2_0 = self.product.S1_0, self.product.S2_0
        ax.plot(S1_0, S2_0, 'r*', markersize=15,
               label=f'Current: S1={S1_0}, S2={S2_0}\nPrice={self.price:.2f}')

        # At-the-money 라인
        ax.axvline(S1_0, color='red', linestyle='--', alpha=0.3, linewidth=1)
        ax.axhline(S2_0, color='red', linestyle='--', alpha=0.3, linewidth=1)

        ax.set_xlabel('S1 (Asset 1)', fontsize=12)
        ax.set_ylabel('S2 (Asset 2)', fontsize=12)
        ax.set_title('ELS Price Contour at t=0', fontsize=14, fontweight='bold')

        plt.colorbar(contour, ax=ax, label='ELS Price')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        if save:
            filepath = os.path.join(self.output_dir, 'price_contour.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_early_redemption_boundary(self, save: bool = True, show: bool = True):
        """
        조기상환 경계면 시각화

        각 관찰일의 조기상환 발생 영역 표시
        """
        if not self.redemption_flags:
            print("⚠️  조기상환 플래그 정보가 없습니다.")
            return

        n_obs = len(self.redemption_flags)
        if n_obs == 0:
            print("⚠️  조기상환 데이터가 없습니다.")
            return

        # 서브플롯 레이아웃
        n_cols = 3
        n_rows = (n_obs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        S1_mesh = self.grid.S1_mesh
        S2_mesh = self.grid.S2_mesh

        for idx, (t, flags) in enumerate(sorted(self.redemption_flags.items())):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # 조기상환 영역 표시
            im = ax.contourf(S1_mesh, S2_mesh, flags.astype(int),
                           levels=[0, 0.5, 1],
                           colors=['lightblue', 'orange'],
                           alpha=0.6)

            # 경계선
            ax.contour(S1_mesh, S2_mesh, flags.astype(int),
                      levels=[0.5],
                      colors='red',
                      linewidths=2)

            # 초기 포인트
            S1_0, S2_0 = self.product.S1_0, self.product.S2_0
            ax.plot(S1_0, S2_0, 'r*', markersize=10)

            # 배리어 라인 (근사)
            obs_idx = list(sorted(self.redemption_flags.keys())).index(t)
            barrier = self.product.redemption_barriers[obs_idx]

            # Worst-of 배리어 라인
            S1_barrier = S1_0 * barrier
            S2_barrier = S2_0 * barrier
            ax.axvline(S1_barrier, color='green', linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(S2_barrier, color='green', linestyle='--', alpha=0.5, linewidth=1)

            ax.set_xlabel('S1')
            ax.set_ylabel('S2')
            ax.set_title(f't = {t:.2f} years\nBarrier: {barrier*100:.0f}%',
                        fontsize=10)
            ax.grid(True, alpha=0.3)

        # 빈 서브플롯 제거
        for idx in range(n_obs, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            fig.delaxes(axes[row, col])

        plt.suptitle('Early Redemption Boundaries', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, 'early_redemption_boundary.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_price_evolution(self, S1: float = None, S2: float = None,
                           save: bool = True, show: bool = True):
        """
        특정 포인트에서 시간에 따른 가격 변화

        Args:
            S1: Asset 1 가격 (None이면 S1_0)
            S2: Asset 2 가격 (None이면 S2_0)
        """
        if not self.snapshots:
            print("⚠️  스냅샷 정보가 없습니다.")
            return

        S1 = S1 or self.product.S1_0
        S2 = S2 or self.product.S2_0

        # 가장 가까운 그리드 포인트 찾기
        i = np.argmin(np.abs(self.grid.S1 - S1))
        j = np.argmin(np.abs(self.grid.S2 - S2))

        times = sorted(self.snapshots.keys())
        prices = [self.snapshots[t][i, j] for t in times]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times, prices, 'b-o', linewidth=2, markersize=6)

        # 관찰일 표시
        for obs_date in self.product.observation_dates:
            ax.axvline(obs_date, color='red', linestyle='--', alpha=0.3)

        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('ELS Price', fontsize=12)
        ax.set_title(f'Price Evolution at S1={S1:.2f}, S2={S2:.2f}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(['Price', 'Observation Dates'])

        if save:
            filepath = os.path.join(self.output_dir, 'price_evolution.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_payoff_comparison(self, save: bool = True, show: bool = True):
        """
        초기 가격(V_0)과 만기 페이오프(V_T) 비교
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        S1_mesh = self.grid.S1_mesh
        S2_mesh = self.grid.S2_mesh

        # V_0
        contour0 = axes[0].contourf(S1_mesh, S2_mesh, self.V_0,
                                   levels=20, cmap='viridis')
        axes[0].plot(self.product.S1_0, self.product.S2_0, 'r*', markersize=15)
        axes[0].set_xlabel('S1')
        axes[0].set_ylabel('S2')
        axes[0].set_title(f'Price at t=0\nCurrent: {self.price:.2f}', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(contour0, ax=axes[0])

        # V_T
        contourT = axes[1].contourf(S1_mesh, S2_mesh, self.V_T,
                                   levels=20, cmap='plasma')
        axes[1].plot(self.product.S1_0, self.product.S2_0, 'r*', markersize=15)
        axes[1].set_xlabel('S1')
        axes[1].set_ylabel('S2')
        axes[1].set_title('Terminal Payoff at t=T', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(contourT, ax=axes[1])

        plt.suptitle('Price vs Terminal Payoff', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, 'payoff_comparison.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_all(self, save: bool = True, show: bool = False):
        """모든 그래프 생성"""
        print("\n" + "="*60)
        print("ELS Pricing Visualization")
        print("="*60)

        self.plot_price_surface_3d(save=save, show=show)
        self.plot_price_contour(save=save, show=show)
        self.plot_early_redemption_boundary(save=save, show=show)
        self.plot_price_evolution(save=save, show=show)
        self.plot_payoff_comparison(save=save, show=show)

        print(f"\n✓ All visualizations saved to: {self.output_dir}/")
        print("="*60)


def plot_price_surface(result: Dict, output_dir: str = "output",
                      save: bool = True, show: bool = True):
    """간편 함수: 가격 surface plot"""
    vis = ELSVisualizer(result, output_dir)
    vis.plot_price_surface_3d(save=save, show=show)


def plot_early_redemption_boundary(result: Dict, output_dir: str = "output",
                                   save: bool = True, show: bool = True):
    """간편 함수: 조기상환 경계면"""
    vis = ELSVisualizer(result, output_dir)
    vis.plot_early_redemption_boundary(save=save, show=show)
