import numpy as np
from typing import List, Callable, Tuple

# ============================================================================
# Elementary Quantile Scoring Functions
# ============================================================================


def elementary_score_quantile(
    p, y, theta: float, alpha: float = 0.5
):
    r"""
    Elementary quantile scoring function for alpha-level quantile.

    S_{alpha, theta}(p, y) = (1{y < p} - alpha) * (1{theta < p} - 1{theta < y})

    Supports scalar float input, or p and y as equal-length np.ndarray.

    Parameters
    ----------
    p : float or array-like
        Predicted quantile level(s) (in [0, 1])
    y : float or array-like
        True observation(s) (in [0, 1])
    theta : float
        Threshold parameter (float)
    alpha : float
        Quantile level (0.5 = median)

    Returns
    -------
    score : float or np.ndarray
        Score value(s)
    """
    p_arr = np.asarray(p)
    y_arr = np.asarray(y)
    # Handle same shape or broadcast error if needed
    if p_arr.shape != y_arr.shape:
        raise ValueError("p and y must have the same shape.")
    # Term 1: (1{y < p} - alpha)
    term1 = (y_arr < p_arr).astype(float) - alpha
    # Term 2: (1{theta < p} - 1{theta < y})
    term2 = (theta < p_arr).astype(float) - (theta < y_arr).astype(float)
    
    out = term1 * term2
    if p_arr.shape == y_arr.shape and len(p_arr.shape) == 1 and p_arr.shape[0] > 1:
        return np.mean(out)
    else:
        return out

def pinball_loss(p, y, alpha):
    return (y - p) * (y > p) * alpha + (p - y) * (y <= p) * (1-alpha)


def create_scoring_function_class(
    alpha_list: List[float],
    m: int = None,
    thetas: np.ndarray = None,
) -> Tuple[List[List[Callable]], np.ndarray, np.ndarray]:
    """
    Create the class S_alpha of elementary scoring functions for MULTIPLE alphas.
    
    Modified to handle multiple quantile levels simultaneously.
    
    Parameters
    ----------
    alpha_list : List[float]
        List of quantile levels to predict (e.g., [0.2, 0.4, 0.7])
    m : int
        Number of discretized theta values
    
    Returns
    -------
    scoring_functions : List[List[Callable]]
        2D list: scoring_functions[i_alpha][i_theta] = S_{alpha_i, theta_i}
        Shape: (len(alpha_list), m)
    alpha_array : np.ndarray
        Array of quantile levels, shape (len(alpha_list),)
    thetas : np.ndarray
        Discretized theta values: theta_i = 1/(2m) + i/m
    """
    
    if thetas is None:
        if m is None:
            raise ValueError("either thetas or m must be provided.")
        else:
            # Discretize theta as theta_i = 1/(2m) + i/m
            print(f"Discretizing theta as theta_i = 1/(2m) + i/m")
            thetas = 1.0 / (2 * m) + np.arange(m) / m

    alpha_array = np.asarray(alpha_list, dtype=np.float64)
    
    def make_scoring_function(theta: float, alpha: float):
        """Create scoring function for specific (theta, alpha) pair."""
        return lambda p, y: elementary_score_quantile(p, y, theta, alpha)
    
    # Create 2D array of scoring functions
    scoring_functions = [
        [make_scoring_function(theta, alpha) for theta in thetas]
        for alpha in alpha_array
    ]
    
    return scoring_functions, thetas
