import numpy as np
from typing import List, Callable, Tuple

# ============================================================================
# Elementary Quantile Scoring Functions
# ============================================================================


def elementary_score_quantile(
    p, y, theta: float, alpha: float = 0.5, keepdims: bool = False
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
    p_arr = np.asarray(p, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    p_arr, y_arr = np.broadcast_arrays(p_arr, y_arr)
    # Term 1: (1{y < p} - alpha)
    term1 = (y_arr < p_arr).astype(float) - alpha
    # Term 2: (1{theta < p} - 1{theta < y})
    term2 = (theta < p_arr).astype(float) - (theta < y_arr).astype(float)
    
    out = term1 * term2
    if keepdims:
        return out
    elif p_arr.shape == y_arr.shape and len(p_arr.shape) == 1 and p_arr.shape[0] > 1:
        return np.mean(out)
    else:
        return out

def pinball_loss(p, y, alpha):
    """
    Compute the pinball (quantile) loss for given predictions, targets, and quantile levels.

    Parameters
    ----------
    p : array-like
        Predicted quantile values
    y : array-like
        True target values
    alpha : float or array-like
        Quantile level(s). Scalar or broadcast-compatible with p, y.

    Returns
    -------
    loss : array-like
        Pinball loss values
    """
    p = np.asarray(p)
    y = np.asarray(y)
    alpha = np.asarray(alpha)
    return np.maximum(alpha * (y - p), (1 - alpha) * (p - y))

def ql_loss(p, y, alpha_list):
    """
    Compute the mean pinball (quantile) loss across provided quantile levels.

    Parameters
    ----------
    p : array-like, shape (n_samples, n_quantiles)
        Predicted quantiles for each sample and each quantile level.
    y : array-like, shape (n_samples,)
        True target values.
    alpha_list : array-like, shape (n_quantiles,)
        List of quantile levels.

    Returns
    -------
    mean_loss : float
        Mean pinball loss over all quantile levels, averaged for each sample (returns shape [n_samples]).
    """
    p = np.asarray(p)
    y = np.asarray(y)
    alpha_list = np.asarray(alpha_list)
    # p: (n_samples, n_quantiles)
    # y: (n_samples,)
    # alpha_list: (n_quantiles,)
    # Broadcast y to (n_samples, n_quantiles)
    loss = pinball_loss(p, y[:, None], alpha_list[None, :])
    return np.mean(loss, axis=1)


# NOT CONSISTENT LOSS FUNCTION
# def log_loss_quantile(p, y, alpha_list):
#     """
#     Approximate log loss from predicted quantiles by constructing
#     a piecewise-uniform density between adjacent quantiles.

#     Parameters
#     ----------
#     p : array-like, shape (n_samples, n_quantiles)
#         Predicted quantiles.
#     y : array-like, shape (n_samples,)
#         True target values.
#     alpha_list : array-like, shape (n_quantiles,)
#         Quantile levels (sorted).

#     Returns
#     -------
#     loss : array-like, shape (n_samples,)
#         Log loss for each sample.
#     """
#     p = np.asarray(p)
#     y = np.asarray(y)
#     alpha_list = np.asarray(alpha_list)

#     T, N = p.shape

#     # find interval indices
#     idx = np.sum(y[:, None] >= p, axis=1) - 1
#     idx = np.clip(idx, 0, N - 2)

#     # interval widths
#     q_left = p[np.arange(T), idx]
#     q_right = p[np.arange(T), idx + 1]
#     width = q_right - q_left

#     # probability mass between quantiles
#     prob = alpha_list[idx + 1] - alpha_list[idx]

#     density = np.zeros_like(width)
#     mask = width != 0
#     density[mask] = prob[mask] / width[mask]
#     loss = -np.log(density)

#     return loss




def elementary_scores_grid_N(p_vec_N, y, thetas, alpha_list):
    N = alpha_list.shape[0]
    m = thetas.shape[0]
    alpha_vec = alpha_list[:, None]
    thetas_row = thetas[None, :]
    assert p_vec_N.shape == (N,), f"p_vec_N.shape: {p_vec_N.shape}, N: {N}"
    assert alpha_vec.shape == (N, 1), f"alpha_vec.shape: {alpha_vec.shape}, N: {N}"
    assert thetas_row.shape == (1, m), f"thetas_row.shape: {thetas_row.shape}, m: {m}"
    
    p_vec = p_vec_N[:, None]        # (N, 1)
    theta_lt_y = (thetas_row < y).astype(float)    # (1, m)
    term1 = (y < p_vec).astype(float) - alpha_vec    # (N, 1)
    term2 = (thetas_row < p_vec).astype(float) - theta_lt_y    # (1, m)

    return term1 * term2    # (N, m)


def elementary_scores_grid_N_m(p_grid_N_m, y, thetas, alpha_list):
    N = alpha_list.shape[0]
    m = thetas.shape[0]
    alpha_vec = alpha_list[:, None]
    thetas_row = thetas[None, :]
    assert p_grid_N_m.shape == (N, m)
    assert alpha_vec.shape == (N, 1)
    assert thetas_row.shape == (1, m)
    
    p_grid = p_grid_N_m     # (N, m)    
    theta_lt_y = (thetas_row < y).astype(float)    # (1, m)
    term1 = (y < p_grid).astype(float) - alpha_vec    # (N, m)
    term2 = (thetas_row < p_grid).astype(float) - theta_lt_y    # (1, m)
    
    return term1 * term2    # (N, m)


def elementary_scores_grid_N_F(p_grid_N_F, y, thetas, alpha_list):
    N = alpha_list.shape[0]
    F = p_grid_N_F.shape[1]
    assert p_grid_N_F.shape == (N, F)
    
    p_grid = p_grid_N_F[:, None, :]     # (N, 1, F)
    theta_grid = thetas[None, :, None]      # (1, m, 1)
    alpha_grid = alpha_list[:, None, None]  # (N, 1, 1)

    term1 = (y < p_grid).astype(float) - alpha_grid          # (N, 1, F)
    term2 = (theta_grid < p_grid).astype(float) - (theta_grid < y).astype(float)  # (1, m, F)

    return term1 * term2          # (N, m, F)


def elementary_scores_grid_T_N(p_grid_T_N, y, thetas, alpha_list):
    T = y.shape[0]
    N = alpha_list.shape[0]
    assert p_grid_T_N.shape == (T, N), f"p_grid_T_N.shape: {p_grid_T_N.shape}, T: {T}, N: {N}"
    
    p_grid = p_grid_T_N[:, :, None]     # (T, N, 1)
    y_arr = y[:, None, None]           # (T, 1, 1)
    theta_grid = thetas[None, None, :]      # (1, 1, m)
    alpha_grid = alpha_list[None, :, None]  # (1, N, 1)
    term1 = (y_arr < p_grid).astype(float) - alpha_grid          # (T, N, 1)
    term2 = (theta_grid < p_grid).astype(float) - (theta_grid < y_arr).astype(float)  # (1, 1, m)
    return term1 * term2          # (T, N, m)




def create_scoring_function_class(
    alpha_list: List[float],
    m: int = None,
    thetas: np.ndarray = None,
    keepdims: bool = False
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
    
    def make_scoring_function(theta: float, alpha: float, keepdims: bool = False):
        """Create scoring function for specific (theta, alpha) pair."""
        return lambda p, y: elementary_score_quantile(p, y, theta, alpha, keepdims=keepdims)
    
    # Create 2D array of scoring functions
    scoring_functions = [
        [make_scoring_function(theta, alpha, keepdims=keepdims) for theta in thetas]
        for alpha in alpha_array
    ]
    
    return scoring_functions, thetas
