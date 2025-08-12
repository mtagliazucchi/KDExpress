"""
KDExpress: Blazing-fast Kernel Density Estimation with FFT and JAX
Author: Matteo Tagliazucchi (https://github.com/mtagliazucchi)
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Matteo Tagliazucchi"
__license__ = "MIT"
__url__ = "https://github.com/mtagliazucchi/KDExpress"

# Check for required dependencies
try:
  import jax
  import jax.numpy as jnp
except ImportError:
  raise ImportError(
    "KDExpress requires JAX. Install it follwoing the instructions at https://github.com/jax-ml/jax"
  )

# Core exports
from .univariate import (
  build_hist_edges,
  silverman_bw1d,
  fft_kde1d,
  binned_kde1d
)
from .multivariate import (
  silverman_bw2d,
  fft_kde2d,
  silverman_bw3d,
  fft_kde3d
)

__all__ = [
  'build_hist_edges',
  'silverman_bw1d',
  'fft_kde1d',
  'binned_kde1d',
  'silverman_bw2d',
  'fft_kde2d',
  'silverman_bw3d',
  'fft_kde3d'
]
