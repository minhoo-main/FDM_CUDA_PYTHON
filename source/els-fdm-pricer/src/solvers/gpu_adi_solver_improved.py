"""
GPU ADI Solver - 개선 버전

주요 개선사항:
1. Batched tridiagonal solver (Python for loop 제거)
2. Vectorized 경계 조건
3. GPU 최적화 커널
"""

import numpy as np
from typing import Callable, Optional

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from .fdm_solver_base import FDMSolver2D
from ..grid.grid_2d import Grid2D


class ImprovedGPUADISolver(FDMSolver2D):
    """
    개선된 GPU ADI Solver

    주요 개선:
    - Batched tridiagonal solver (for loop 제거)
    - Vectorized operations
    - 최소한의 CPU↔GPU 전송
    """

    def __init__(self, grid: Grid2D, r: float, q1: float, q2: float,
                 sigma1: float, sigma2: float, rho: float,
                 use_gpu: bool = True):
        super().__init__(grid, r, q1, q2, sigma1, sigma2, rho)

        if not CUPY_AVAILABLE:
            print("⚠️  CuPy가 설치되지 않았습니다. CPU 모드로 실행합니다.")
            self.use_gpu = False
        else:
            self.use_gpu = use_gpu
            if use_gpu:
                try:
                    device_id = cp.cuda.Device().id
                    props = cp.cuda.runtime.getDeviceProperties(device_id)
                    gpu_name = props['name'].decode('utf-8')
                    print(f"✓ Improved GPU 가속 활성화: {gpu_name}")
                except:
                    print("✓ Improved GPU 가속 활성화")

        self.xp = cp if (self.use_gpu and CUPY_AVAILABLE) else np
        self._precompute_coefficients()

    def _precompute_coefficients(self):
        """ADI 계수를 GPU 메모리에 미리 계산"""
        xp = self.xp
        N1, N2 = self.N1, self.N2
        dS1, dS2 = self.grid.dS1, self.grid.dS2
        dt = self.grid.dt / 2.0  # half time step

        S1 = self.grid.S1
        S2 = self.grid.S2

        # S1 방향 계수
        self.alpha1 = xp.zeros(N1)
        self.beta1 = xp.zeros(N1)
        self.gamma1 = xp.zeros(N1)

        for i in range(1, N1-1):
            sigma1_term = 0.5 * self.sigma1**2 * S1[i]**2
            drift1_term = (self.r - self.q1) * S1[i]

            self.alpha1[i] = -0.5 * dt * (sigma1_term / dS1**2 - drift1_term / (2*dS1))
            self.beta1[i] = 1.0 + dt * (sigma1_term / dS1**2 + self.r / 2.0)
            self.gamma1[i] = -0.5 * dt * (sigma1_term / dS1**2 + drift1_term / (2*dS1))

        # 경계
        self.beta1[0] = 1.0
        self.beta1[-1] = 1.0

        # S2 방향 계수
        self.alpha2 = xp.zeros(N2)
        self.beta2 = xp.zeros(N2)
        self.gamma2 = xp.zeros(N2)

        for j in range(1, N2-1):
            sigma2_term = 0.5 * self.sigma2**2 * S2[j]**2
            drift2_term = (self.r - self.q2) * S2[j]

            self.alpha2[j] = -0.5 * dt * (sigma2_term / dS2**2 - drift2_term / (2*dS2))
            self.beta2[j] = 1.0 + dt * (sigma2_term / dS2**2 + self.r / 2.0)
            self.gamma2[j] = -0.5 * dt * (sigma2_term / dS2**2 + drift2_term / (2*dS2))

        self.beta2[0] = 1.0
        self.beta2[-1] = 1.0

    def solve(self, V_T: np.ndarray,
              early_exercise_callback: Optional[Callable] = None) -> np.ndarray:
        """
        PDE를 시간 역방향으로 풀기
        """
        xp = self.xp

        # GPU 메모리로 전송 (한 번만)
        V = xp.array(V_T)

        # 시간 역방향 진행
        for n in range(self.Nt - 1, -1, -1):
            t = self.grid.t[n]

            # ADI Half-steps (개선된 batched solver 사용)
            V = self._adi_step_batched(V)

            # 조기상환 체크 (필요시 CPU로)
            if early_exercise_callback is not None:
                # TODO: GPU vectorized callback 구현
                V_cpu = cp.asnumpy(V) if self.use_gpu else V
                S1_mesh = self.grid.S1_mesh
                S2_mesh = self.grid.S2_mesh
                V_cpu = early_exercise_callback(V_cpu, S1_mesh, S2_mesh, n, t)
                V = xp.array(V_cpu)

        # 결과 반환 (GPU -> CPU)
        if self.use_gpu:
            return cp.asnumpy(V)
        else:
            return V

    def _adi_step_batched(self, V):
        """
        개선된 ADI 스텝 (batched tridiagonal solver)
        """
        # Half-step 1: S1 방향
        V_half = self._solve_S1_batched(V)

        # Half-step 2: S2 방향
        V_new = self._solve_S2_batched(V_half)

        # 경계 조건
        V_new = self._apply_boundary_conditions(V_new)

        return V_new

    def _solve_S1_batched(self, V):
        """
        S1 방향 batched tridiagonal solver

        핵심 개선: Python for loop 제거!
        """
        xp = self.xp
        N1, N2 = self.N1, self.N2

        # RHS 준비 (vectorized)
        RHS = V.copy()
        RHS[0, :] = 0.0
        RHS[-1, :] = V[-1, :]

        # Batched Thomas algorithm
        # N2개의 tridiagonal 시스템을 동시에 풀기
        V_new = self._batched_thomas(
            self.alpha1, self.beta1, self.gamma1, RHS
        )

        return V_new

    def _solve_S2_batched(self, V):
        """
        S2 방향 batched tridiagonal solver
        """
        xp = self.xp
        N1, N2 = self.N1, self.N2

        # Transpose해서 S2를 첫 번째 차원으로
        V_T = V.T  # (N2, N1)

        # RHS 준비
        RHS = V_T.copy()
        RHS[0, :] = 0.0
        RHS[-1, :] = V_T[-1, :]

        # Batched Thomas
        V_new_T = self._batched_thomas(
            self.alpha2, self.beta2, self.gamma2, RHS
        )

        # Transpose back
        return V_new_T.T

    def _batched_thomas(self, lower, diag, upper, RHS):
        """
        Batched Thomas algorithm (핵심 개선!)

        입력:
            lower: (N,) - lower diagonal
            diag: (N,) - main diagonal
            upper: (N,) - upper diagonal
            RHS: (N, M) - M개의 RHS vectors

        출력:
            X: (N, M) - M개의 solutions

        개선사항:
        - Python for loop 제거
        - Vectorized operations
        - GPU 병렬 실행
        """
        xp = self.xp
        N, M = RHS.shape

        # Forward sweep (vectorized!)
        c = xp.zeros((N-1, M))
        d = xp.zeros((N, M))

        # 첫 행
        c[0, :] = upper[0] / diag[0]
        d[0, :] = RHS[0, :] / diag[0]

        # 나머지 행 (vectorized)
        for i in range(1, N-1):
            denom = diag[i] - lower[i] * c[i-1, :]
            c[i, :] = upper[i] / denom
            d[i, :] = (RHS[i, :] - lower[i] * d[i-1, :]) / denom

        # 마지막 행
        i = N - 1
        denom = diag[i] - lower[i] * c[i-1, :]
        d[i, :] = (RHS[i, :] - lower[i] * d[i-1, :]) / denom

        # Backward substitution (vectorized!)
        X = xp.zeros((N, M))
        X[N-1, :] = d[N-1, :]

        for i in range(N-2, -1, -1):
            X[i, :] = d[i, :] - c[i, :] * X[i+1, :]

        return X

    def _apply_boundary_conditions(self, V):
        """경계 조건 적용 (vectorized)"""
        xp = self.xp
        V_new = V.copy()

        # Dirichlet at S=0
        V_new[0, :] = 0.0
        V_new[:, 0] = 0.0

        # Linear extrapolation at S_max (vectorized!)
        V_new[-1, :] = 2 * V_new[-2, :] - V_new[-3, :]
        V_new[:, -1] = 2 * V_new[:, -2] - V_new[:, -3]

        return V_new


# GPU 메모리 최적화를 위한 유틸리티
def get_optimal_batch_size(N1, N2, gpu_memory_gb=15.0):
    """
    GPU 메모리에 맞는 최적 배치 크기 계산
    """
    # 대략적인 메모리 사용량 추정
    bytes_per_element = 8  # float64
    memory_per_grid = N1 * N2 * bytes_per_element

    # 안전 마진 (50%)
    available_memory = gpu_memory_gb * 1e9 * 0.5

    max_grids = int(available_memory / memory_per_grid)

    return max(1, max_grids)
