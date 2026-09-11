import jax
import jax.numpy as jnp

EPS = 1E-100

@jax.jit
def safe_div(x, y, eps=EPS):
    """
    Safe division with robust handling of zero and near-zero denominators.
    """
    y_abs = jnp.abs(y)
    y_safe = jnp.where(y_abs > eps, y, eps * jnp.where(y >= 0, 1.0, -1.0))
    return x / y_safe

@jax.jit
def safe_average(data, weights=None, epsilon=EPS):
    """
    Safe version of jnp.average using safe_div for robust division.
    
    Args:
        data: Array of values
        weights: Optional array of weights (must be same shape as data)
        epsilon: Small value for safe division fallback
    
    Returns:
        Weighted average with safe division handling
    """
    if weights is None:
        # Unweighted average
        sum_data = jnp.sum(data)
        n = data.size
        return safe_div(sum_data, n, eps=epsilon)
    else:
        # Weighted average
        weighted_sum = jnp.sum(data * weights)
        total_weight = jnp.sum(weights)
        return safe_div(weighted_sum, total_weight, eps=epsilon)

@jax.jit
def safe_sqrt(x, eps=EPS):
    """sqrt with a finite gradient everywhere, including at x <= 0."""
    safe_x = jnp.where(x > 0, x, eps)      # never actually sqrt(0) or sqrt(neg) in the traced branch
    return jnp.where(x > 0, jnp.sqrt(safe_x), 0.0)
