import jax
import jax.numpy as jnp
from functools import partial
from plum import dispatch
from typing import List, Union
import builtins
import math

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
def searchsorted_linear_spacing(x, y):
  min_val = jnp.min(x)
  max_val = jnp.max(x)
  n = jnp.sum(~jnp.isnan(x))-1 # len(x)-1
  idx = jnp.floor((y - min_val) / (max_val-min_val) * n)
  idx = jnp.clip(idx, 0, n-1).astype(int)
  return idx

@dispatch
@partial(jax.jit, static_argnames=('density'))
def hist1d(data, bins:jnp.ndarray, weights=None, density:bool=False):

  weights = jnp.ones_like(data) if weights is None else weights

  # Compute bin indices for each data point
  bin_indices = searchsorted_linear_spacing(bins, data)

  # Initialize bin counts array and use scatter-add
  bin_counts = jnp.zeros(len(bins)-1)
  bin_counts = bin_counts.at[bin_indices].add(weights)

  # Apply density normalization if requested
  if density:
    bin_widths = bins[1] - bins[0]
    total_weight = jnp.sum(weights)
    bin_counts = bin_counts / (bin_widths * total_weight)

  return bin_counts, bins

@dispatch
@partial(jax.jit, static_argnames=('density', 'bins'))
def hist1d(data, bins:int, weights=None, density:bool=False):
  weights = jnp.ones_like(data) if weights is None else weights

  # Compute bin indices for each data point
  bin_edges = jnp.linspace(jnp.min(data), jnp.max(data), bins+1)
  bin_indices = searchsorted_linear_spacing(bin_edges, data)

  # Initialize bin counts array and use scatter-add
  bin_counts = jnp.zeros(bins)
  bin_counts = bin_counts.at[bin_indices].add(weights)

  # Apply density normalization if requested
  if density:
    bin_widths = bin_edges[1] - bin_edges[0]
    total_weight = jnp.sum(weights)
    bin_counts = bin_counts / (bin_widths * total_weight)

  return bin_counts, bin_edges

###########################
# Histogram N-dimensional #
###########################

# Similar to jnp.histogramdd but do not use searchsorted but uses the fact that bins are linearly spaced.

@dispatch
@partial(jax.jit, static_argnames=('bins', 'density'))
def histnd(data, bins:Union[int, List[int]], weights=None, density=False):

  N, D = jnp.shape(data)
  weights = jnp.ones(N) if weights is None else weights

  if jnp.isscalar(bins):
    bins = D * [bins]
  else:
    if len(bins) != D:
      raise ValueError("should be a bin for each dimension.")
    bins = list(bins)  # type: ignore[arg-type]

  bin_idx_by_dim = []
  bin_edges_by_dim = []

  for i in builtins.range(D):
    bin_edges = jnp.linspace(jnp.min(data[:,i]), jnp.max(data[:,i]), bins[i]+1)
    bin_idx = searchsorted_linear_spacing(bin_edges, data[:,i])+1 # compatibility with jax
    bin_idx_by_dim.append(bin_idx)
    bin_edges_by_dim.append(bin_edges)

  nbins = tuple(len(bin_edges) + 1 for bin_edges in bin_edges_by_dim)

  xy = jnp.ravel_multi_index(tuple(bin_idx_by_dim), nbins, mode='clip')
  hist = jnp.bincount(xy, weights, length=math.prod(nbins))
  hist = jnp.reshape(hist, nbins)
  core = D*(slice(1, -1),)
  hist = hist[core]

  if density:
    dedges = [jnp.diff(bin_edges) for bin_edges in bin_edges_by_dim]
    hist = hist.astype(data.dtype)
    hist /= hist.sum()
    for norm in jnp.ix_(*dedges):
      hist = jnp.where(~jnp.isnan(norm), hist / norm, hist)

  return hist, bin_edges_by_dim


@dispatch
@partial(jax.jit, static_argnames=('density'))
def histnd(data, bins:List[jnp.ndarray], weights=None, density=False):

  N, D = jnp.shape(data)
  weights = jnp.ones(N) if weights is None else weights

  if len(bins) != D:
    raise ValueError("should be a bin for each dimension.")
  bin_edges = list(bins)

  bin_idx_by_dim = []
  bin_edges_by_dim = []

  for i in builtins.range(D):
    bin_idx = searchsorted_linear_spacing(bin_edges[i], data[:,i])+1
    bin_idx_by_dim.append(bin_idx)
    bin_edges_by_dim.append(bin_edges[i])

  nbins = tuple(len(bin_edges) + 1 for bin_edges in bin_edges_by_dim)

  xy = jnp.ravel_multi_index(tuple(bin_idx_by_dim), nbins, mode='clip')
  hist = jnp.bincount(xy, weights, length=math.prod(nbins))
  hist = jnp.reshape(hist, nbins)
  core = D*(slice(1, -1),)
  hist = hist[core]

  if density:
    dedges = [jnp.diff(bin_edges) for bin_edges in bin_edges_by_dim]
    hist = hist.astype(data.dtype)
    hist /= hist.sum()
    for norm in jnp.ix_(*dedges):
      hist = jnp.where(~jnp.isnan(norm), hist / norm, hist)

  return hist, bin_edges_by_dim
