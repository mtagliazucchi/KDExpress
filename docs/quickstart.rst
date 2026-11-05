Quickstart
==========

1D KDE
------

The FFT estimators need an **equally spaced** evaluation grid. Weights and
bandwidth are optional; the bandwidth defaults to Scott's rule.

.. code-block:: python

   import jax.numpy as jnp
   import numpy as np
   import KDExpress as kd

   data = jnp.array(np.random.default_rng(0).normal(size=5000))
   grid = jnp.linspace(-5, 5, 512)

   pdf = kd.fft_kde1d(grid, data)                        # Gaussian kernel
   pdf_epan = kd.fft_kde1d(grid, data, kernel="epan")    # Epanechnikov kernel

Evaluating on a non-regular grid
--------------------------------

Evaluate on a regular grid, then interpolate:

.. code-block:: python

   x = jnp.array([-1.3, 0.2, 2.7])
   pdf_x = jnp.interp(x, grid, pdf)

For N-D grids use :class:`jax.scipy.interpolate.RegularGridInterpolator`.

Binned KDE
----------

:func:`~KDExpress.binned_kde1d` bins the data first and accepts any evaluation
points, so no regular grid is required:

.. code-block:: python

   pdf = kd.binned_kde1d(grid, data, nbins=200)

2D and 3D
---------

.. code-block:: python

   data2 = jnp.array(np.random.default_rng(1).normal(size=(2000, 2)))
   gx = jnp.linspace(-4, 4, 64)
   pdf2 = kd.fft_kde2d(gx, gx, data2)        # shape (64, 64)

   data3 = jnp.array(np.random.default_rng(2).normal(size=(2000, 3)))
   gz = jnp.linspace(-4, 4, 32)
   pdf3 = kd.fft_kde3d(gz, gz, gz, data3)    # shape (32, 32, 32)

Gradients
---------

Everything is differentiable, so KDEs can sit inside a larger JAX
computation, e.g. a likelihood:

.. code-block:: python

   import jax

   grad = jax.grad(lambda d: kd.fft_kde1d(grid, d).sum())(data)

Bandwidths
----------

.. code-block:: python

   bw = kd.silverman_bw1d(data)
   bw = kd.scott_bw1d(data, jnp.ones_like(data))
   pdf = kd.fft_kde1d(grid, data, bw=bw)

More worked examples and benchmarks are in the
`examples folder <https://github.com/mtagliazucchi/KDExpress/tree/main/examples>`_.
