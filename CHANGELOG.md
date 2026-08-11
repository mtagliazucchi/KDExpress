# Changelog

All notable changes to KDExpress will be documented in this file.

## [0.2.1] - 2026-08-11

### Fixed
- Numerical stability issues with Epanechnikov kernel in FFT-based KDE that were breaking automatic differentiation.

## [0.2.0] - 2025-09-03

### Added
- 1D and ND histogram calculation using linear bin spacing. These are jitted. 5-7 times than `jax.numpy` implementation.
- New `scott_bw1d` for Scott's rule BW estimate

## [0.1.0] - 2025-08-12

### Added
- First release of KDExpress
