import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm

import KDExpress as kd


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(0)
    return jnp.array(rng.normal(size=5000))


@pytest.fixture(scope="module")
def grid():
    return jnp.linspace(-5, 5, 512)


@pytest.mark.parametrize("kernel", ["gaussian", "epan", "epanechnikov"])
def test_fft_kde1d_normalised_and_accurate(sample, grid, kernel):
    p = kd.fft_kde1d(grid, sample, kernel=kernel)
    assert float(jnp.trapezoid(p, grid)) == pytest.approx(1.0, abs=1e-3)
    assert float(jnp.max(jnp.abs(p - norm.pdf(grid)))) < 0.05


@pytest.mark.parametrize("kernel", ["gaussian", "epan"])
def test_fft_kde1d_gradient_finite(sample, grid, kernel):
    g = jax.grad(lambda x: kd.fft_kde1d(grid, x, kernel=kernel).sum())(sample)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_epanechnikov_alias_matches_epan(sample, grid):
    a = kd.fft_kde1d(grid, sample, kernel="epan")
    b = kd.fft_kde1d(grid, sample, kernel="epanechnikov")
    assert jnp.allclose(a, b)


def test_unknown_kernel_raises(sample, grid):
    with pytest.raises(ValueError):
        kd.fft_kde1d(grid, sample, kernel="triangle")


def test_binned_kde1d_close_to_truth(sample, grid):
    p = kd.binned_kde1d(grid, sample)
    assert float(jnp.max(jnp.abs(p - norm.pdf(grid)))) < 0.05


def test_bandwidths_positive(sample):
    assert float(kd.silverman_bw1d(sample)) > 0
    assert float(kd.scott_bw1d(sample, jnp.ones_like(sample))) > 0


def test_fft_kde2d_default_and_explicit_weights():
    rng = np.random.default_rng(1)
    data = jnp.array(rng.normal(size=(2000, 2)))
    g = jnp.linspace(-4, 4, 64)
    a = kd.fft_kde2d(g, g, data)  # weights=None used to crash
    b = kd.fft_kde2d(g, g, data, weights=jnp.ones(2000))
    assert a.shape == (64, 64)
    assert jnp.allclose(a, b, atol=1e-6)


def test_fft_kde3d_default_weights():
    rng = np.random.default_rng(2)
    data = jnp.array(rng.normal(size=(2000, 3)))
    g = jnp.linspace(-4, 4, 32)
    assert kd.fft_kde3d(g, g, g, data).shape == (32, 32, 32)


def test_star_import():
    ns = {}
    exec("from KDExpress import *", ns)
    for name in kd.__all__:
        assert name in ns
