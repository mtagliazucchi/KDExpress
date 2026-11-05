KDExpress
=========

Kernel Density Estimation accelerated with the Fast Fourier Transform and
`JAX <https://github.com/jax-ml/jax>`_: JIT compilation, GPU support and
automatic differentiation.

- **FFT-accelerated KDE** in 1D, 2D and 3D (:func:`~KDExpress.fft_kde1d`,
  :func:`~KDExpress.fft_kde2d`, :func:`~KDExpress.fft_kde3d`), on regular grids.
- **Binned KDE** in 1D with arbitrary evaluation points
  (:func:`~KDExpress.binned_kde1d`).
- **Bandwidth selectors** following Scott's and Silverman's rules.
- **Fast histograms** in 1D and N-D (:func:`~KDExpress.hist1d`,
  :func:`~KDExpress.histnd`).
- Gaussian and Epanechnikov kernels, differentiable end to end.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   examples/index
   api
   changelog
