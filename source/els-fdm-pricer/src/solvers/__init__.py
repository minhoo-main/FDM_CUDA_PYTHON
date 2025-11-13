"""FDM Solvers"""

from .fdm_solver_base import FDMSolver2D
from .adi_solver import ADISolver

try:
    from .gpu_adi_solver import GPUADISolver, check_gpu_available
    from .gpu_adi_solver_optimized import OptimizedGPUADISolver
    __all__ = ['FDMSolver2D', 'ADISolver', 'GPUADISolver', 'OptimizedGPUADISolver', 'check_gpu_available']
except ImportError:
    # GPU 지원 없음 (CuPy 미설치)
    __all__ = ['FDMSolver2D', 'ADISolver']
