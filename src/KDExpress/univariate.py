import jax
import jax.numpy as jnp
from functools import partial

# Common function

@jax.jit
def build_hist_edges(bin_centers):
  """Compute histogram edges surrounding given bin centers.

  Args:
    bin_centers: Array of equally spaced bin center locations

  Returns:
    Array of edges between bins (length len(bin_centers)+1)
  """
  left_edge = jnp.array([bin_centers[0] - 0.5*(bin_centers[1]-bin_centers[0])])
  right_edge = jnp.array([bin_centers[-1] + 0.5*(bin_centers[-1]-bin_centers[-2])])
  inner_edges = 0.5*(bin_centers[1:]+bin_centers[:-1])
  return jnp.concatenate([left_edge, inner_edges, right_edge])

@jax.jit
def silverman_bw1d(data, alpha=0.9):
  """Silverman's rule of thumb bandwidth estimator for 2D datasets.

  Args:
    data: 1D array of input data points
    alpha: Scaling factor (default 0.9)

  Returns:
    Estimated optimal bandwidth
  """
  ndata = data.shape[0]
  if ndata <= 1:
    return alpha
  var_width = jnp.std(data)
  q25, q75 = jnp.percentile(data, jnp.array([25.0, 75.0]))
  quantile_width = (q75 - q25) / 1.34
  width = jnp.minimum(var_width, quantile_width)
  width = jnp.where(width == 0.0,
          jnp.where(var_width == 0.0, 1.0, var_width),
          width)
  return alpha * width * (ndata ** -0.2)

# ======================
# 1D KDE IMPLEMENTATION
# ======================

@jax.jit
def cf_gaussian_kernel_1d(t, sigma):
  """Characteristic function of Gaussian kernel for FFT convolution.

  Args:
    t: Frequency domain values
    sigma: Bandwidth parameter

  Returns:
    Fourier transform of Gaussian kernel
  """
  return jnp.exp(-0.5*(t*sigma)**2)

@jax.jit
def fft_kde1d(points, data, weights=None, bw=None, bin_edges=None):
  """Compute a 1D Kernel Density Estimation (KDE) using FFT-based Gaussian smoothing.

  Args:
    points: Equally spaced evaluation points (1D array)
    data: Input data points (1D array)
    weights: Optional weights for each data point
    bw: Optional bandwidth (auto-estimated if None)
    bin_edges: Optional precomputed histogram edges

  Returns:
    Smoothed kernel density estimates at evaluation points
  """
  # Check points are equally spaced
  grid_step = points[1] - points[0]
  jax.debug.callback(
    lambda x: print("Warning: Points are not equally spaced") if not x else None,
    jnp.allclose(jnp.diff(points), grid_step, atol=1e-6)
  )

  # Normalize weights
  if weights is None:
    weights = jnp.ones_like(data)
  assert len(weights) == len(data), "Weights must match data lenght."
  weights /= jnp.sum(weights)

  # Build histogram edges if necessary
  if bin_edges is None:
    bin_edges = build_hist_edges(points)

  # Compute weighted histogram
  pdf_at_points, _ = jnp.histogram(data, weights=weights,
                                  bins=bin_edges, density=True)

  # Compute bandwidth if necessary
  if bw is None:
    bw = silverman_bw1d(data)

  # FFT-based convolution smoothing
  freqs = jnp.fft.fftfreq(len(pdf_at_points), d=grid_step)
  fft_kernel = cf_gaussian_kernel_1d(2 * jnp.pi * freqs, bw)
  fft_pdf_at_points = jnp.fft.fft(pdf_at_points)
  fft_kde = fft_pdf_at_points * fft_kernel  # Frequency-domain smoothing -> KDE in frequency domain
  kde = jnp.fft.ifft(fft_kde).real # KDE in "time" domanin

  # Mask negative value and normalize
  kde = jnp.where(kde < 0, 0.0, kde)
  kde /= jnp.sum(kde) * grid_step

  return kde

# ==============
# BINNED 1D KDE
# ==============

@partial(jax.jit, static_argnames=['kernel', 'nbins', 'cut_sigma_data'])
def binned_kde1d(points,
  data,
  weights=None,
  bw=None,
  kernel='gaussian',
  nbins = 200,
  cut_sigma_data = None):
  """Compute a 1D Kernel Density Estimation (KDE) binning fataset to spped up the evaluation.
  Supportg both Epanechnikov and Gaussian kernels.

  Args:
    points: 1D array of evaluation points where KDE is computed
    data: 1D array of input data points
    weights: Optional 1D array of weights for each data point (default: uniform weights)
    bw: Bandwidth value (if None, uses Silverman's rule)
    kernel: Kernel type ('epan' for Epanechnikov, otherwise Gaussian)
    nbins: Number of bins for histogram approximation
    cut_sigma_data: If specified, reduce evaluation points beyond data support by this many standard deviations

  Returns:
    1D array of density estimates at input points
  """
  # Normalize weights
  if weights is None:
    weights = jnp.ones_like(data) / dataset.size
  else:
    weights = weights / jnp.sum(weights)

  # Bw selection
  if bw is None:
    bw = silverman_bw1d(data)

  # Binning
  new_weights, new_data_edges = jnp.histogram(data, weights=weights, bins=nbins)
  new_data = 0.5*(new_data_edges[1:]+new_data_edges[:-1])

  # Compute "effective points" if requested -> useful if points extends much further away data support
  if cut_sigma_data is not None:
    min_data, max_data, std_data = jnp.min(data), jnp.max(data), jnp.std(data)
    eff_points = jnp.linspace(min_data-cut_sigma_data*std_data, max_data+cut_sigma_data*std_data, len(points)//2) # guess it is okay...
  else:
    eff_points = points

  # Choose kernel and compute kernel values
  kernel_fn   = _epan_kernel if kernel == 'epan' else _gaussian_kernel
  kernel_vals = kernel_fn((eff_points[:,None] - new_data) / bw)

  # Calculate KDE
  kde = jnp.sum(new_weights * kernel_vals, axis=-1) / bw
  if cut_sigma_data is not None:
    return jnp.interp(points, eff_points, kde)
  else:
    return kde

@jax.jit
def _epan_kernel(u):
  return jnp.where(jnp.abs(u) <= 1, 3/4 * (1 - u**2), 0)

@jax.jit
def _gaussian_kernel(u):
  return jnp.exp(-0.5 * u**2)  / jnp.sqrt(2 * jnp.pi)
