<p align="center">
  <img src="KDExpress_logo.png" alt="KDExpress Logo" style="max-width:100%; width:320px; min-width:200px;" />
</p>

<p align="center">
  Kernel Density Estimation accelerated with Fast Fourier Transform and JAX (JIT+GPU support+AD).
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python"></a>
  <a href="https://github.com/google/jax"><img src="https://img.shields.io/badge/JAX-compatible-red" alt="JAX"></a>
</p>


## Features
- **FFT-accelerated KDE** (inspired by [KernelDensity.jl](https://github.com/JuliaStats/KernelDensity.jl)):
  - `fft_kde1d`: 1D KDE with FFT convolution (requires regular grid)
  - `fft_kde2d`: 2D KDE with FFT convolution (requires regular grid)
  - `fft_kde3d`: 3D KDE with FFT convolution (requires regular grid)

  The KDE can be evaluated on non-regular grid by interpolation, such as `jax.numpy.interp` or `jax.scipy.interpolate.RegularGridInterpolator`

- **Binned-accelerated implementation**:
  - `binned_kde1d`: 1D KDE with data binning

- **Bandwidth estimators**:
  - Silverman's rule (`silverman_bw1d/2d/3d`)

## Installation

1. Install **JAX** following the instructions at: [https://github.com/jax-ml/jax](https://github.com/jax-ml/jax), e.g

    ```bash
    pip install "jax[cpu]" # For CPU-only version
    ```
    or
    ```bash
    pip install "jax[cuda12]" # For GPU support (CUDA)
    ```

2. Clone the `KDExpress` repo

    ```bash
    git clone https://github.com/mtagliazucchi/KDExpress
    ```

3. Install the code

    a. Editable install:
    ```bash
    cd KDExpress
    pip install -e .
    ```

    b. Or use it ad-hoc:
    ```python
    import sys; sys.path.append("/path/to/KDExpress")
    from KDExpress import fft_kde1d
    ```

## Usage and benchmarks
See the [examples](https://github.com/mtagliazucchi/KDExpress/examples) folder.

## License
MIT © [Matteo Tagliazucchi](https://github.com/mtagliazucchi)
