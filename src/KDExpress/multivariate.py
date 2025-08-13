import jax
import jax.numpy as jnp
from .univariate import build_hist_edges, silverman_bw1d

# ======================
# 2D KDE IMPLEMENTATION
# ======================

@jax.jit
def silverman_bw2d(data, alpha=0.9):
  """Silverman's rule of thumb bandwidth estimator for 2D datasets.

  Args:
    data: 2D array of input data points
    alpha: Scaling factor (default 0.9)

  Returns:
    Estimated optimal bandwidth
  """
  ndata = data.shape[0]
  if ndata <= 1:
    return jnp.array([alpha]*2)

  # Calculate for each dimension
  def inner_routine(_data):
    width = silverman_bw1d(_data, alpha)
    width *= (ndata ** (-1/6)) / (ndata ** -0.2)  # 2D adjustment
    return width

  return jax.vmap(inner_routine, in_axes = (1,))(data)

@jax.jit
def cf_gaussian_kernel_2d(tx, ty, sigma_x, sigma_y):
  """Characteristic function of 2D Gaussian kernel for FFT convolution.

  Args:
    tx: Frequency domain x-values
    ty: Frequency domain y-values
    sigma_x: Bandwidth parameter for x points
    sigma_y: Bandwidth parameter for y points

  Returns:
    Fourier transform of Gaussian kernel
  """
  return jnp.exp(-0.5 * ((tx*sigma_x)**2 + (ty*sigma_y)**2))

@jax.jit
def fft_kde2d(points_x, points_y, data, weights=None, bw=None, bin_edges=None):
  """Compute a 2D Kernel Density Estimation (KDE) using FFT-based Gaussian smoothing.

  This function implements a binned KDE with Gaussian kernels, optimized using FFT convolution.
  The implementation follows Silverman's rule for bandwidth selection and handles weighted data.

  Args:
  points_x : 1D array of equally-spaced x-coordinates for the evaluation grid
  points_y : 1D array of equally-spaced y-coordinates for the evaluation grid
  data : 2D array of shape (N_samples, 2) containing the input data points
  weights : 1D array of weights for each data point (must sum to 1). If None, uniform weights are used.
  bw : Bandwidth (sigma_x, sigma_y) for the Gaussian kernel. If None, automatically computed using Silverman's rule.
  bin_edges :  Precomputed bin edges for histogramming as (x_edges, y_edges). If None, automatically computed.

  Returns
    2D array containing the KDE evaluated at the grid points (points_x, points_y)
  """
  # Check if points are equally spaced
  grid_steps = []
  for points in [points_x, points_y]:
    grid_step = points[1] - points[0]
    #jax.debug.callback(
    #  lambda x: print("Warning: Points are not equally spaced") if not x else None,
    #  jnp.allclose(jnp.diff(points), grid_step, atol=1e-6)
    # ) # slow down a lot the gpu usage
    grid_steps.append(grid_step)

  assert data.shape[1] == 2, "Data must be a (N_samples, 2) Array"

  # Normalize weights
  if weights is None:
    weights = jnp.ones_like(data)
  # assert len(weights) == len(data), "Weights lenght must match data lenght." # slow down gpu usage
  weights /= jnp.sum(weights)

  # Build histogram edges if necessary
  if bin_edges is None:
    bin_edges = [build_hist_edges(points) for points in [points_x, points_y]]
  else:
    assert len(bin_edges) == 2

  # Compute weighted histogram
  pdf_at_points, _, _ = jnp.histogram2d(
    data[:, 0], data[:, 1],
    weights=weights,
    bins=bin_edges,
    density=True
  )

  # Compute bandwidth if necessary
  if bw is None:
    bw = silverman_bw2d(data)
  else:
    assert len(bin_edges) == 2

  # FFT-based convolution smoothing
  freqs = [jnp.fft.fftfreq(pdf_at_points.shape[i], d=grid_steps[i]) for i in range(2)]
  fx, fy = jnp.meshgrid(*freqs, indexing='ij')
  fft_kernel = cf_gaussian_kernel_2d(2*jnp.pi * fx, 2*jnp.pi * fy, bw[0], bw[1])
  fft_pdf_at_points = jnp.fft.fft2(pdf_at_points)
  fft_kde = fft_pdf_at_points * fft_kernel  # Frequency-domain smoothing -> KDE in frequency domain
  kde = jnp.fft.ifft2(fft_kde).real # KDE in "time" domanin

  # Mask negative value and normalize
  kde = jnp.where(kde < 0, 0.0, kde)
  kde /= jnp.sum(kde) * jnp.prod(jnp.asarray(grid_steps))

  return kde

# ======================
# 3D KDE IMPLEMENTATION
# ======================

@jax.jit
def silverman_bw3d(data, alpha=0.9):
  """Silverman's rule of thumb bandwidth estimator for 3D datasets.

  Args:
      data: 2D array of shape (N_samples, 3)
      alpha: Scaling factor (default 0.9)

  Returns:
      Estimated optimal bandwidth for each dimension (3,)
  """
  ndata = data.shape[0]
  if ndata <= 1:
    return jnp.array([alpha] * 3)

  # Calculate for each dimension with 3D adjustment (n^(-1/7))
  def inner_routine(_data):
    width = silverman_bw1d(_data, alpha)
    width *= (ndata ** (-1/7)) / (ndata ** -0.2)  # 3D adjustment
    return width

  return jax.vmap(inner_routine, in_axes=(1,))(data)

@jax.jit
def cf_gaussian_kernel_3d(tx, ty, tz, sigma_x, sigma_y, sigma_z):
  """Characteristic function of 3D Gaussian kernel for FFT convolution.

  Args:
    tx: Frequency domain x-values
    ty: Frequency domain y-values
    tz: Frequency domain z-values
    sigma_x: Bandwidth for x dimension
    sigma_y: Bandwidth for y dimension
    sigma_z: Bandwidth for z dimension

  Returns:
    Fourier transform of 3D Gaussian kernel
  """
  return jnp.exp(-0.5 * ((tx*sigma_x)**2 + (ty*sigma_y)**2 + (tz*sigma_z)**2))

@jax.jit
def fft_kde3d(points_x, points_y, points_z, data, weights=None, bw=None, bin_edges=None):
  """Compute a 3D Kernel Density Estimation (KDE) using FFT-based Gaussian smoothing.

  Args:
    points_x: 1D array of equally-spaced x-coordinates
    points_y: 1D array of equally-spaced y-coordinates
    points_z: 1D array of equally-spaced z-coordinates
    data: 2D array of shape (N_samples, 3)
    weights: 1D array of weights for each point (None for uniform)
    bw: Bandwidth tuple (sigma_x, sigma_y, sigma_z) (None for auto)
    bin_edges: Precomputed bin edges as (x_edges, y_edges, z_edges)

  Returns:
    3D array of KDE evaluated at grid points
  """
  # Check grid spacing
  grid_steps = []
  for points in (points_x, points_y, points_z):
    grid_step = points[1] - points[0]
    # jax.debug.callback(
    #  lambda x: print("Warning: Points are not equally spaced") if not x else None,
    #  jnp.allclose(jnp.diff(points), grid_step, atol=1e-6)
    # ) # slow down gpu usgae
    grid_steps.append(grid_step)

  assert data.shape[1] == 3, "Data must be shape (N_samples, 3)"

  # Normalize weights
  if weights is None:
    weights = jnp.ones(data.shape[0])
  # assert len(weights) == len(data), "Weights length must match data" # slow down gpu usgae
  weights /= jnp.sum(weights)

  # Build histogram edges
  if bin_edges is None:
    bin_edges = [build_hist_edges(points) for points in (points_x, points_y, points_z)]
  else:
    assert len(bin_edges) == 3

  # Compute weighted histogram
  pdf_at_points, _ = jnp.histogramdd(
    data,
    bins=bin_edges,
    weights=weights,
    density=True
  )

  # Compute bandwidth
  if bw is None:
    bw = silverman_bw3d(data)
  else:
    assert len(bw) == 3

  # FFT-based convolution
  freqs = [jnp.fft.fftfreq(pdf_at_points.shape[i], d=grid_steps[i]) for i in range(3)]
  fx, fy, fz = jnp.meshgrid(*freqs, indexing='ij')
  fft_kernel = cf_gaussian_kernel_3d(
    2*jnp.pi * fx,
    2*jnp.pi * fy,
    2*jnp.pi * fz,
    bw[0], bw[1], bw[2]
  )
  fft_pdf_at_points = jnp.fft.fftn(pdf_at_points)
  fft_kde = fft_pdf_at_points * fft_kernel
  kde = jnp.fft.ifftn(fft_kde).real

  # Clip negatives and normalize
  kde = jnp.where(kde < 0, 0.0, kde)
  kde /= jnp.sum(kde) * jnp.prod(jnp.array(grid_steps))

  return kde
