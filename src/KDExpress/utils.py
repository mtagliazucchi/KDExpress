import jax
import jax.numpy as jnp

@jax.jit
def safe_average(data, weights=None):
    """
    Weighted (or unweighted) average.

    Args:
        data: Array of values
        weights: Optional array of weights (must be same shape as data)

    Returns:
        Weighted average
    """
    if weights is None:
        # Unweighted average
        sum_data = jnp.sum(data)
        n = data.size
        return sum_data / n
    else:
        # Weighted average
        weighted_sum = jnp.sum(data * weights)
        total_weight = jnp.sum(weights)
        return weighted_sum / total_weight
