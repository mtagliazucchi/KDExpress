API reference
=============

.. currentmodule:: KDExpress

Kernel density estimation
-------------------------

.. autofunction:: fft_kde1d
.. autofunction:: binned_kde1d
.. autofunction:: fft_kde2d
.. autofunction:: fft_kde3d

Bandwidth selectors
-------------------

.. autofunction:: scott_bw1d
.. autofunction:: silverman_bw1d
.. autofunction:: scott_bw2d
.. autofunction:: silverman_bw2d
.. autofunction:: scott_bw3d
.. autofunction:: silverman_bw3d

Histograms
----------

.. autofunction:: build_hist_edges

These two functions use multiple dispatch on the type of ``bins``, so they are
documented by hand. Bins must be linearly spaced. Both are jitted and fast
(5-7 times faster than the ``jax.numpy`` equivalents).

.. function:: hist1d(data, bins, weights=None, density=False)

   Histogram of 1D data on linearly spaced bins.

   :param data: 1D array of samples.
   :param bins: Either an ``int`` (number of bins, spanning ``min(data)`` to
      ``max(data)``) or an array of linearly spaced bin edges.
   :param weights: Optional 1D array of weights (default: all ones).
   :param bool density: If ``True``, normalise to a probability density.
   :returns: ``(counts, edges)``.

.. function:: histnd(data, bins, weights=None, density=False)

   Histogram of N-D data on linearly spaced bins.

   :param data: Array of shape ``(N, D)``.
   :param bins: Either an ``int`` or a list of ``D`` ints (bins per
      dimension), or a list of ``D`` arrays of linearly spaced bin edges.
   :param weights: Optional 1D array of ``N`` weights (default: all ones).
   :param bool density: If ``True``, normalise to a probability density.
   :returns: ``(counts, edges_by_dimension)``.
