
# KDExpress ![Logo](./KDExpress_logo.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-compatible-red)](https://github.com/google/jax)

Fast Kernel Density Estimation (KDE) using a Fast Fourier Transform. Writen in JAX (JIT+GPU support+AD) and  inspired by [KernelDensity.jl](https://github.com/JuliaStats/KernelDensity.jl).

## Features
- **FFT-accelerated KDE**:
  - `fft_kde1d`: 1D KDE with FFT convolution (requires regular grid)
  - `fft_kde2d`: 2D KDE with FFT convolution (requires regular grid)
  - `fft_kde3d`: 3D KDE with FFT convolution (requires regular grid)

  The KDE can be evaluated on non-regular grid by interpolation, such as `jax.numpy.interp` or `jax.scipy.interpolate.RegularGridInterpolator`

- **Binned-accelerated implementation**:
  - `binned_kde1d`: 1D KDE with data binning

- **Bandwidth estimators**:
  - Silverman's rule (`silverman_bw1d/2d/3d`)

## Installation
You must install **JAX** separately following the instructions at: [https://github.com/jax-ml/jax](https://github.com/jax-ml/jax).
To get the `KDExpress` you can simply clone this repo

```bash
git clone https://github.com/mtagliazucchi/KDExpress
```
and then use it as
```python
import sys; sys.path.append("path/to/KDExpress")
from KDExpress import fft_kde1d
```
## Usage and benchmarks
See the [examples](https://github.com/mtagliazucchi/KDExpress/examples) folder.

## License
MIT © [Matteo Tagliazucchi](https://github.com/mtagliazucchi)
