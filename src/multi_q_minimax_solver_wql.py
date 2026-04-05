# This version runs omniprediction over weighted quantile losses. (Much faster without discretization)

import numpy as np
from typing import List, Tuple

# ============================================================================
# Efficient V_n Computation with Precomputation
# ============================================================================

def multi_q_minmax_solver_wql(
    weights_NF: np.ndarray,  # (N, F)
    forecasts_NF: np.ndarray,  # (N, F)
) -> dict:
    """
    Solve the minmax problem for a single quantile level.
    Contains randomness in the choice of phat.
    """
    N, F = weights_NF.shape
    assert weights_NF.shape == forecasts_NF.shape
    assert np.isclose(np.sum(weights_NF), 1.0)

    # First find V_n and j_n^*
    Vn_dic = solve_weighted_hinge_split_all_n(weights_NF = weights_NF, 
                            forecasts_NF = forecasts_NF, 
                            tol=min(weights_NF.min()/2,1e-10)
                            )
    
    Vn_values = Vn_dic['minimum']
    j_optimal = Vn_dic['theta_interval'][:,0]
    assert Vn_values.shape == j_optimal.shape == (N+1,)
    assert np.all(j_optimal[:-1] <= j_optimal[1:]), f"j_optimal must be sorted (non-decreasing), j_optimal: {j_optimal}"
    #print(f'Old theta stars {j_optimal}')

    # print(f"Vn_values: {Vn_values}")
    # print(f"j_optimal: {j_optimal}")

    weights_N = weights_NF.sum(axis=1)
    weighted_forecasts_N = (forecasts_NF * weights_NF).sum(axis=1)
    
    numerator_N = weighted_forecasts_N + Vn_values[:-1] - Vn_values[1:]
    phat = np.concatenate([[-np.inf], numerator_N / weights_N])   # Just to make the indices aligned with V_n
    # print(f'old numerator: {numerator_N}')
    # print(f'old denominator: {weights_N}')
    
    if not np.all(phat[:-1] <= phat[1:] + 1e-10):
        print(f'phat is not ordered, max violation: {np.max(phat[:-1] - phat[1:])}')
        for i in range(1, N):
            if phat[i] >= phat[i+1] + 1e-10:
                print(f'phat_n: {phat[i]}, phat_n+1: {phat[i+1]}')
                print(f'theta_n-1^*: {j_optimal[i-1]} \n theta_n^*: {j_optimal[i]} \n theta_n+1^*: {j_optimal[i+1]}')
                
                print('Printing V_j')
                def V_func(V_idx, eval_idx):
                    left_term = np.maximum(forecasts_NF[:V_idx,:] - j_optimal[eval_idx], 0.0)
                    right_term = np.maximum(j_optimal[eval_idx] - forecasts_NF[V_idx:,:], 0.0)
                    return (weights_NF[:V_idx,:] * left_term).sum() + (weights_NF[V_idx:,:] * right_term).sum()

                print(f'V(n-1)*: {V_func(i-1, i-1)}')
                print(f'V(n-1)(theta_n^*): {V_func(i-1, i)}')
                print(f'V(n+1)*: {V_func(i+1, i+1)}')
                print(f'V(n+1)(theta_n^*): {V_func(i+1, i)}')
                print(f'weight sum: {weights_NF[(i-1),:].sum()}')
                print(f'min weight: {np.min(weights_NF)}')
                print(f'min_wieght/2: {weights_NF.min()/2}')
                
                print('#'*20)
                print('Rerun with verbose mode')
                solve_weighted_hinge_split_all_n(weights_NF = weights_NF, 
                            forecasts_NF = forecasts_NF, 
                            tol=min(weights_NF.min()/2,1e-10),
                            verbose=True,
                            n_list=[i, i+1]
                            )
                raise ValueError('phat is not ordered')

    return phat[1:], Vn_values



def efficeint_solve_weighted_hinge_split(
    weights_NF: np.ndarray,
    forecasts_NF: np.ndarray,
    tol: float = 1e-12,
    verbose: bool = False,
) -> tuple[float, tuple[float, float]]:
    """
    Efficiently solve the weighted hinge split problem.
    """
    weights = np.asarray(weights_NF, dtype=np.float64)
    forecasts = np.asarray(forecasts_NF, dtype=np.float64)

    if weights.shape != forecasts.shape:
        raise ValueError(f"Shape mismatch: weights {weights.shape}, forecasts {forecasts.shape}")
    if weights.ndim != 2:
        raise ValueError(f"Expected 2D arrays of shape (N, F), got ndim={weights.ndim}")
    if np.any(weights < -tol):
        raise ValueError("weights_NF must be nonnegative")

    N, F = weights.shape

    # Column sum and cumsum
    w_sum_N = np.concatenate([[0.0], np.sum(weights, axis=1)])
    w_cumsum_N_front = np.cumsum(w_sum_N)               # w_cumsum_N_front[i]: quantile level 1 to i's weight sum
    w_cumsum_N_back = w_sum_N.sum() - w_cumsum_N_front  # w_cumsum_N_back[i]:  quantile level i+1 to N's weight sum
    wf_N = np.concatenate([[0.0], np.sum(weights * forecasts, axis=1)])

    weights_total = w_sum_N.sum()
    if not np.isclose(weights_total, 1.0, rtol=0.0, atol=tol):
        # raise ValueError(f"weights_NF must sum to 1.0, got {total_w}")
        print(f"weights_NF must sum to 1.0, got {weights_total}")

    w_1d = weights.reshape(-1)
    f_1d = forecasts.reshape(-1)
    order = np.argsort(f_1d, kind="mergesort")
    f_sorted = f_1d[order]
    w_sorted = w_1d[order]
    
    #########################################################################################
    # Find theta_star for each n_split = 1, ..., N-1, which is one of the values in f_sorted.
    #########################################################################################
    f_uniq_vals: list[float] = []               # unique values of f_sorted, let length be K
    f_uniq_vals_w_cumsum: list[float] = []      # f_uniq_vals_w_cumsum[k]:  \sum_{s,f) w_s^f * 1(f_s^f <= f_uniq_vals[k])
    f_uniq_vals_wf_cumsum: list[float] = []    # f_uniq_vals_wf_cumsum[k]: \sum_{s,f) w_s^f * f_s^f * 1(f_s^f = f_uniq_vals[k])
    M = f_sorted.size
    running_w_sum = 0.0
    running_wf_sum = 0.0
    i = 0
    while i < M:
        v = float(f_sorted[i])
        j = i
        while j < M and f_sorted[j] == v:
            j += 1
        here_w_sum = w_sorted[i:j].sum()
        running_w_sum += here_w_sum
        running_wf_sum += here_w_sum * v
        f_uniq_vals.append(float(v))
        f_uniq_vals_w_cumsum.append(running_w_sum)
        f_uniq_vals_wf_cumsum.append(running_wf_sum)
        i = j
    wf_total = wf_N[1:].sum()
    w_total = w_sum_N[1:].sum()
    assert np.isclose(running_w_sum, 1.0, rtol=0.0, atol=1e-10), f'running_w_sum: {running_w_sum}, 1.0: {1.0}'
    assert np.isclose(running_wf_sum, wf_total, rtol=0.0, atol=1e-10), f'running_wf_sum: {running_wf_sum}, w_total: {wf_total}'

    K = len(f_uniq_vals)
    theta_stars = np.empty(N+1, dtype=float)
    theta_stars[0] = f_uniq_vals[0]
    theta_stars[N] = f_uniq_vals[K-1]
    theta_stars_idx = np.empty(N+1, dtype=int)   # Index of theta_star in f_sorted
    theta_stars_idx[0] = 0
    theta_stars_idx[N] = K-1


    curr_k_idx = 0
    s_idx = np.repeat(np.arange(N, dtype=int), F)
    Vn = np.zeros(N+1, dtype=float)
    for n_split in range(1, N):
        while f_uniq_vals_w_cumsum[curr_k_idx] < w_cumsum_N_front[n_split]:
            curr_k_idx += 1
            if curr_k_idx >= K:
                raise ValueError(f'curr_k_idx reached end of uniq_vals, curr_k_idx: {curr_k_idx}, K: {K}')
        theta_stars[n_split] = f_uniq_vals[curr_k_idx]
        theta_stars_idx[n_split] = curr_k_idx

        # Compute objective value at theta_star for this n_split.
        left_mask = s_idx < n_split
        left_term = np.maximum(f_1d[left_mask] - theta_stars[n_split], 0.0)
        right_term = np.maximum(theta_stars[n_split] - f_1d[~left_mask], 0.0)
        Vn[n_split] = float(np.dot(w_1d[left_mask], left_term) + np.dot(w_1d[~left_mask], right_term))

    Vn_diff_arr = Vn[:-1] - Vn[1:]
    numerator_N = wf_N[1:] + Vn_diff_arr
    phat = numerator_N / (w_sum_N[1:])
    # print(f'new thetas: {theta_stars}')

    # Numerical stability warnings
    if np.abs(np.sum(Vn_diff_arr)) > 1e-10:
        print(f'Warning: V0 - VN = {np.sum(Vn_diff_arr)} exceeds 1e-10')
    if np.min(phat[1:] - phat[:-1]) < -1e-12:
        print(f'Warning: phat is not ordered, min violation: {np.min(phat[1:] - phat[:-1])}')


    return phat, Vn[:-1] - Vn[1:]

    # # #########################################################################################
    # # # Compute V_{n-1} - V_n
    # # #########################################################################################
    # def zero_if_neg_idx(arr, idx):
    #     if idx < 0:
    #         return 0.0
    #     if idx >= len(arr):
    #         return arr[len(arr)-1]
    #     try:
    #         return arr[idx]
    #     except IndexError:
    #         raise IndexError(f'idx: {idx}, arr_length: {len(arr)}')

    # # print(f'K={K}, theta_stars_idx: {theta_stars_idx}')
    # # print(f'fuvwc len: {len(f_uniq_vals_w_cumsum)}, fuvwfc len: {len(f_uniq_vals_wf_cumsum)}')

    # Vn_diff_arr = np.empty(N+1, dtype=float)    # Vn_diff_arr[n]: V_{n-1} - V_n, Vn_diff_arr[0] not used
    # for n in range(1, N + 1):
    #     Vn_diff_arr[n] = zero_if_neg_idx(f_uniq_vals_w_cumsum, theta_stars_idx[n-1]-1) * theta_stars[n-1] + \
    #         (w_total - zero_if_neg_idx(f_uniq_vals_w_cumsum, theta_stars_idx[n]-1)) * theta_stars[n] + \
    #         zero_if_neg_idx(f_uniq_vals_wf_cumsum, theta_stars_idx[n]-1) - zero_if_neg_idx(f_uniq_vals_wf_cumsum, theta_stars_idx[n-1]-1) - \
    #         w_cumsum_N_front[n-1] * theta_stars[n-1] - w_cumsum_N_back[n] * theta_stars[n] - wf_N[n]

    #     # print(f'n: {n}, Vn_diff_arr[n]: {Vn_diff_arr[n]}')
    #     # print(f'term 1: {zero_if_neg_idx(f_uniq_vals_w_cumsum, theta_stars_idx[n-1]-1) * theta_stars[n-1]}')
    #     # print(f'term 2: {(1 - zero_if_neg_idx(f_uniq_vals_w_cumsum, theta_stars_idx[n]-1)) * theta_stars[n]}')
    #     # print(f'term 3: {zero_if_neg_idx(f_uniq_vals_wf_cumsum, theta_stars_idx[n]-1) - zero_if_neg_idx(f_uniq_vals_wf_cumsum, theta_stars_idx[n-1]-1)}')
    #     # print(f'term 4: {- w_cumsum_N_front[n-1] * theta_stars[n-1]}')
    #     # print(f'term 5: {- w_cumsum_N_back[n] * theta_stars[n]}')
    #     # print(f'term 6: {- wf_N[n]}')

    # numerator_N = wf_N[1:] + Vn_diff_arr[1:]
    # phat = numerator_N / (w_sum_N[1:])

    

    return phat, Vn_diff_arr[1:]



def minimax_value_neg(alpha_list: np.ndarray, Vn_values: np.ndarray) -> float:
    """
    Negation of minimax value of negative of the minimax value.
    = - \sum_{n=1}^{N-1} (\tau_{n+1} - \tau_n) V_n
    """
    N = alpha_list.shape[0]
    return np.sum(alpha_list * (Vn_values[0:N] - Vn_values[1:N+1]))


def _solve_weighted_hinge_from_sorted(
    g_sorted: np.ndarray,
    w_sorted: np.ndarray,
    left_sorted: np.ndarray,
    tol: float = 1e-12,
    verbose: bool = False,
) -> tuple[float, tuple[float, float]]:
    """
    Internal helper: given flattened, globally sorted values, weights and
    a left/right split encoded by left_sorted, return (theta_star, interval).
    """
    if g_sorted.size == 0:
        raise ValueError("Empty input to _solve_weighted_hinge_from_sorted")

    slope_minus_inf = -float(w_sorted[left_sorted].sum())

    uniq_vals: list[float] = []
    m_minus: list[float] = []
    m_plus: list[float] = []
    running_slope = slope_minus_inf

    M = g_sorted.size
    i = 0
    while i < M:
        v = g_sorted[i]
        j = i
        jump = 0.0
        while j < M and g_sorted[j] == v:
            jump += w_sorted[j]
            j += 1

        slope_after = running_slope + jump
        uniq_vals.append(float(v))
        m_minus.append(float(running_slope))
        m_plus.append(float(slope_after))
        running_slope = slope_after
        i = j

    theta_lo = np.inf
    theta_hi = -np.inf

    def _include_interval(lo: float, hi: float) -> None:
        nonlocal theta_lo, theta_hi
        theta_lo = min(theta_lo, lo)
        theta_hi = max(theta_hi, hi)

    # Flat segment to the far left.
    if abs(slope_minus_inf) <= tol:
        _include_interval(-np.inf, uniq_vals[0])

    if verbose:
        print(f'tol: {tol}')
    K = len(uniq_vals)
    # for k in range(K):
    #     # Knot minimizer if 0 in subgradient interval [m_minus, m_plus].
    #     if m_minus[k] < tol and m_plus[k] > -tol:
    #         # _include_interval(uniq_vals[k], uniq_vals[k])
    #         theta_lo = uniq_vals[k]
    #         theta_hi = uniq_vals[k]
    #         # if m_minus[k] < 0 and m_plus[k] > 0:
    #         #     break
    #         if verbose:
    #             print(f'k={k}: Knot minimizer: uniq_vals[k]: {uniq_vals[k]}, m_minus[k]: {m_minus[k]}, m_plus[k]: {m_plus[k]}')

    #     # Flat segment between knots.
    #     if k < K - 1 and abs(m_plus[k]) <= tol:
    #         _include_interval(uniq_vals[k], uniq_vals[k + 1])
    #         if verbose:
    #             print(f'k={k}: Flat segment: uniq_vals[k]: {uniq_vals[k]}, uniq_vals[k + 1]: {uniq_vals[k + 1]}, m_plus[k]: {m_plus[k]}')

    for k in range(K):
        if m_plus[k] >=0:
            theta_lo = uniq_vals[k]
            theta_hi = uniq_vals[k]
            break


    # Flat segment to the far right.
    if abs(m_plus[-1]) <= tol:
        _include_interval(uniq_vals[-1], np.inf)
        print(f'Flat segment to the far right: uniq_vals[-1]: {uniq_vals[-1]}, np.inf')

    if theta_lo > theta_hi:
        # Degenerate numerical fallback.
        theta_lo = uniq_vals[0]
        theta_hi = uniq_vals[0]
        print(f'Fallback used: theta_lo: {theta_lo}, theta_hi: {theta_hi}')

    if np.isneginf(theta_lo) and np.isposinf(theta_hi):
        theta_star = 0.0
        print(f'Both theta_lo and theta_hi are -inf and inf, theta_star: {theta_star}')
    elif np.isneginf(theta_lo):
        theta_star = float(theta_hi)
        print(f'theta_lo is -inf, theta_star: {theta_star}')
    elif np.isposinf(theta_hi):
        theta_star = float(theta_lo)
        print(f'theta_hi is inf, theta_star: {theta_star}')
    else:
        theta_star = float(theta_lo)

    return float(theta_star), (float(theta_lo), float(theta_hi))


def solve_weighted_hinge_split_minimization(
    weights_NF: np.ndarray,
    preds_NF: np.ndarray,
    n_split: int,
    tol: float = 1e-12,
) -> dict:
    """
    Solve
        min_theta [
            sum_{s=1}^n   sum_{f=1}^F w_s^f * max(g_f^s - theta, 0)
          + sum_{s=n+1}^N sum_{f=1}^F w_s^f * max(theta - g_f^s, 0)
        ].

    Parameters
    ----------
    weights_NF : np.ndarray
        Shape (N, F), nonnegative weights that sum to 1 (within tolerance).
    preds_NF : np.ndarray
        Shape (N, F), scalar base predictions g_f^s.
    n_split : int
        Number of rows in the first group (1 <= s <= n_split).
        Uses 0-indexed slicing in code:
            left block  = [:n_split, :]
            right block = [n_split:, :]
    tol : float
        Numerical tolerance for weight sum and slope sign checks.

    Returns
    -------
    dict
        {
            "theta_star": float,            # one minimizer
            "theta_interval": (float, float), # minimizer interval [lo, hi]
            "minimum": float,               # objective value at theta_star
        }
    """
    weights = np.asarray(weights_NF, dtype=np.float64)
    preds = np.asarray(preds_NF, dtype=np.float64)

    if weights.shape != preds.shape:
        raise ValueError(f"Shape mismatch: weights {weights.shape}, preds {preds.shape}")
    if weights.ndim != 2:
        raise ValueError(f"Expected 2D arrays of shape (N, F), got ndim={weights.ndim}")
    if np.any(weights < -tol):
        raise ValueError("weights_NF must be nonnegative")

    N, _ = weights.shape
    if not (0 <= n_split <= N):
        raise ValueError(f"n_split must be in [0, N], got n_split={n_split}, N={N}")

    total_w = float(weights.sum())
    if not np.isclose(total_w, 1.0, rtol=0.0, atol=tol):
        raise ValueError(f"weights_NF must sum to 1.0, got {total_w}")

    g = preds.reshape(-1)
    w = weights.reshape(-1)
    if g.size == 0:
        raise ValueError("weights_NF and preds_NF must be non-empty")

    # Global sort of all (s, f) pairs once.
    order = np.argsort(g, kind="mergesort")
    g_sorted = g[order]
    w_sorted = w[order]

    # Build left/right mask for this specific n_split.
    left_mask = np.zeros((N, weights.shape[1]), dtype=bool)
    left_mask[:n_split, :] = True
    is_left = left_mask.reshape(-1)
    left_sorted = is_left[order]

    theta_star, theta_interval = _solve_weighted_hinge_from_sorted(
        g_sorted, w_sorted, left_sorted, tol
    )

    left_term = np.maximum(g[is_left] - theta_star, 0.0)
    right_term = np.maximum(theta_star - g[~is_left], 0.0)
    minimum = float(np.dot(w[is_left], left_term) + np.dot(w[~is_left], right_term))

    return {
        "theta_star": theta_star,
        "theta_interval": theta_interval,
        "minimum": minimum,
    }


def solve_weighted_hinge_split_all_n(
    weights_NF: np.ndarray,
    forecasts_NF: np.ndarray,
    tol: float = 1e-12,
    n_list: List[int] = None,
    verbose: bool = False,
) -> dict:
    """
    Compute optimizers for all n_split = 1, ..., N-1 in one pass.

    This reuses a single global sort of (s, f) pairs, so the cost is
    O((N*F) log(N*F) + N*(N*F)) ~ O(N^2 * F) total, instead of paying
    a log-factor sort for every n_split.

    Returns
    -------
    dict with keys
        "theta_star"      : np.ndarray, shape (N+1,)
        "theta_interval"  : np.ndarray, shape (N+1, 2)
        "minimum"         : np.ndarray, shape (N+1,)
    where entry n corresponds to n_split = n.
    """
    weights = np.asarray(weights_NF, dtype=np.float64)
    forecasts = np.asarray(forecasts_NF, dtype=np.float64)

    if weights.shape != forecasts.shape:
        raise ValueError(f"Shape mismatch: weights {weights.shape}, forecasts {forecasts.shape}")
    if weights.ndim != 2:
        raise ValueError(f"Expected 2D arrays of shape (N, F), got ndim={weights.ndim}")
    if np.any(weights < -tol):
        raise ValueError("weights_NF must be nonnegative")

    N, F = weights.shape
    total_w = float(weights.sum())
    if not np.isclose(total_w, 1.0, rtol=0.0, atol=tol):
        # raise ValueError(f"weights_NF must sum to 1.0, got {total_w}")
        print(f"weights_NF must sum to 1.0, got {total_w}")

    g = forecasts.reshape(-1)
    w = weights.reshape(-1)
    if g.size == 0:
        raise ValueError("weights_NF and preds_NF must be non-empty")

    order = np.argsort(g, kind="mergesort")
    g_sorted = g[order]
    w_sorted = w[order]

    # Row indices for each flattened (s, f) pair.
    s_idx = np.repeat(np.arange(N, dtype=int), F)
    s_sorted = s_idx[order]

    theta_star_all = np.empty(N + 1, dtype=float)
    theta_interval_all = np.empty((N + 1, 2), dtype=float)
    minimum_all = np.empty(N + 1, dtype=float)

    theta_star_all[0] = -np.inf
    theta_star_all[N] = np.inf
    theta_interval_all[0,:] = -np.inf
    theta_interval_all[N,:] = np.inf
    minimum_all[0] = 0
    minimum_all[N] = 0

    for n_split in range(1, N):
        if verbose:
            if n_list is not None:
                if n_split not in n_list:
                    continue
            print(f'n_split: {n_split}')

        left_sorted = s_sorted < n_split
        theta_star, theta_interval = _solve_weighted_hinge_from_sorted(
            g_sorted=g_sorted, 
            w_sorted=w_sorted, 
            left_sorted=left_sorted, 
            tol=tol, 
            verbose=verbose,
        )

        # Compute objective value at theta_star for this n_split.
        left_mask = s_idx < n_split
        left_term = np.maximum(g[left_mask] - theta_star, 0.0)
        right_term = np.maximum(theta_star - g[~left_mask], 0.0)
        minimum = float(np.dot(w[left_mask], left_term) + np.dot(w[~left_mask], right_term))

        theta_star_all[n_split] = theta_star
        theta_interval_all[n_split, 0] = theta_interval[0]
        theta_interval_all[n_split, 1] = theta_interval[1]
        minimum_all[n_split] = minimum

    return {
        "theta_star": theta_star_all,
        "theta_interval": theta_interval_all,
        "minimum": minimum_all,
    }