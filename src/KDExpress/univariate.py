import jax
import jax.numpy as jnp
from functools import partial
from .hist import build_hist_edges, hist1d
from .utils import safe_div, safe_average, safe_sqrt, EPS

# Common function

@jax.jit
def silverman_bw1d(data, alpha=0.9):
  """Silverman's rule of thumb bandwidth estimator for 1D datasets.

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

@jax.jit
def weighted_std(data, weights):
  mean = safe_average(data, weights=weights)
  variance = safe_average((data - mean)**2, weights=weights)
  return safe_sqrt(variance)

@jax.jit
def scott_bw1d(data, weights):
  """Scott's rule of thumb bandwidth estimator for 1D datasets.

  Args:
    data: 1D array of input data points
    weights: 1D array of input weights

  Returns:
    Estimated optimal bandwidth
  """
  weights = safe_div(weights, jnp.sum(weights))
  sum_w2 = jnp.sum(weights**2)
  neff = safe_div(1.0, sum_w2) 
  bw = jnp.power(neff, -1. / (1 + 4))
  bw *= weighted_std(data, weights)
  return bw

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
def cf_epan_kernel_1d(t, sigma):
    """Characteristic function of Epanechnikov kernel for FFT convolution.
    
    Args:
        t: Frequency domain values
        sigma: Bandwidth parameter
    
    Returns:
        Fourier transform of scaled Epanechnikov kernel: K_sigma(u) = 1/sigma * K(u/sigma)
    """
    t_scaled = t * sigma
    t_abs = jnp.abs(t_scaled)
    t2 = t_scaled**2
    
    taylor = 1.0 - t2/10.0 + t2**2/280.0 - t2**3/15120.0 + t2**4/1330560.0
    
    safe_t = jnp.where(t_abs < 1e-10, 1.0, t_scaled)
    exact = 3.0 * (jnp.sin(safe_t) - safe_t * jnp.cos(safe_t)) / (safe_t**3)
    
    alpha = 0.1 
    transition = 0.5 * (1.0 + jnp.tanh((0.1 - t_abs) / alpha))
    
    result = transition * taylor + (1.0 - transition) * exact
    
    cutoff_scale = 20.0
    cutoff = 0.5 * (1.0 + jnp.tanh((cutoff_scale - t_abs) / alpha))
    result = result * cutoff
    
    result = jnp.clip(result, 0.0, 1.0)
    
    return result


@partial(jax.jit, static_argnames=['kernel'])
def fft_kde1d(points, data, weights=None, bw=None, kernel='gaussian', bin_edges=None):
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

  #jax.debug.callback(
  #  lambda x: print("Warning: Points are not equally spaced") if not x else None,
  #  jnp.allclose(jnp.diff(points), grid_step, atol=1e-6)
  #) slow down a lot the gpu usage

  # Normalize weights to sum to 1. Downstream (hist1d's density branch,
  # scott_bw1d) divides by sum(weights); when the raw weights sum to an
  # astronomically small but nonzero number (e.g. an event whose samples
  # sit far in the tail of the population model), the *value* of that
  # division is still well-conditioned (numerator and denominator are
  # comparably tiny), but its gradient carries an implicit 1/sum(weights)**2
  # term that overflows float32 long before the raw sum itself would.
  # Normalizing once here keeps every downstream sum O(1).
  if weights is None:
    weights = jnp.ones_like(data)
  weights = safe_div(weights, jnp.sum(weights))

  # Build histogram edges if necessary
  if bin_edges is None:
    bin_edges = build_hist_edges(points)

  # Compute weighted histogram
  pdf_at_points, _ = hist1d(data, bin_edges, weights=weights, density=True) # check density here

  # Compute bandwidth if necessary
  if bw is None:
    bw = scott_bw1d(data, weights)

  if kernel == 'epan':
    cf = cf_epan_kernel_1d
  else:
    cf = cf_gaussian_kernel_1d

  # FFT-based convolution smoothing
  freqs = jnp.fft.fftfreq(len(pdf_at_points), d=grid_step)
  fft_pdf_at_points = jnp.fft.fft(pdf_at_points)
  fft_kernel = cf(2 * jnp.pi * freqs, bw)
  fft_kde = fft_pdf_at_points * fft_kernel  # Frequency-domain smoothing -> KDE in frequency domain
  kde = jnp.fft.ifft(fft_kde).real # KDE in "time" domanin

  # Mask negative value and normalize
  kde = jnp.where(kde < 0, 0.0, kde)
  kde = safe_div(kde, jnp.sum(kde) * grid_step)

  return kde

# ==============
# BINNED 1D KDE
# ==============

@partial(jax.jit, static_argnames=['kernel', 'nbins'])
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


  # Normalize weights up front (see fft_kde1d for why: binning then
  # dividing by the *raw* bin sum carries a 1/sum(weights)**2 term in the
  # gradient that overflows float32 well before sum(weights) itself would).
  if weights is None:
    weights = jnp.ones_like(data)
  weights = safe_div(weights, jnp.sum(weights))

  # Binning
  new_weights, new_data_edges = hist1d(data, nbins, weights=weights, density=False)
  new_weights = safe_div(new_weights, jnp.sum(new_weights))
  new_data = 0.5*(new_data_edges[1:]+new_data_edges[:-1])

  # Bw selection
  if bw is None:
    bw = scott_bw1d(new_data, new_weights)

  # Compute "effective points" if requested -> useful if points extends much further away data support
  if cut_sigma_data is not None:
    data_min, data_max, data_std = jnp.min(data), jnp.max(data), jnp.std(data)
    lb = jnp.where(data_min-cut_sigma_data*data_std > jnp.min(points), data_min-cut_sigma_data*data_std, jnp.min(points))
    ub = jnp.where(data_max+cut_sigma_data*data_std < jnp.max(points), data_max+cut_sigma_data*data_std, jnp.max(points))
    eff_points = jnp.linspace(lb, ub, len(points)//3) # guess it is okay...
  else:
    eff_points = points

  # Choose kernel and compute kernel values
  kernel_fn   = _epan_kernel if kernel == 'epan' else _gaussian_kernel
  kernel_vals = kernel_fn((eff_points[:,None] - new_data) / bw)

  # Calculate KDE
  kde = jnp.sum(new_weights * kernel_vals, axis=-1) / bw
  return jnp.interp(points, eff_points, kde, left=0., right=0.)

@jax.jit
def _epan_kernel(u):
  return jnp.where(jnp.abs(u) <= 1, 3/4 * (1 - u**2), 0)

@jax.jit
def _gaussian_kernel(u):
  return jnp.exp(-0.5 * u**2)  / jnp.sqrt(2 * jnp.pi)
