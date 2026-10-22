"""CUDA spectral ocean waves, with an explicit device coupling contract."""
from .model import SpectralGrid, WaveModel
from .solver import create_solver
__all__ = ["SpectralGrid", "WaveModel", "create_solver"]
