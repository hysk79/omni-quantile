# This version merges omniprediction weight update and selection of best forecaster for each elementary score

import numpy as np
from typing import List, Tuple

# ============================================================================
# Efficient V_n Computation with Precomputation
# ============================================================================

class VnComputer_v2:
    """
    Efficient computation of V_n with precomputed cumulative sums.
    
    The naive O(N*m^2) approach computes V_n for each n by summing over
    all i and s. We can reduce this to O(m) per time step by precomputing
    cumulative sums.
    """
    
    def __init__(self, weights_F: np.ndarray, thetas: np.ndarray, weighted_indicators_NmF: np.ndarray, tol: float = 1e-10):
        """
        Initialize V_n computer with precomputed arrays.
        
        Parameters
        ----------
        weights : np.ndarray
            Shape (N, m, F) where weights[s, i, f] = w_i^s,f
            s is 0-indexed time (0 to N-1)
            i is 0-indexed quantile (0 to m-1)
        thetas : np.ndarray (m,)
        weighted_indicators_NmF : np.ndarray (N, m, F), indicators[s, i, f] = I(theta_i < f_i^s,f) in {0, 1}
        """

        self.weights_F = np.asarray(weights_F, dtype=np.float64)
        self.weights = self.weights_F.sum(axis=2) # (N, m)
        self.thetas = np.asarray(thetas, dtype=np.float64)
        self.weighted_indicators_NmF = weighted_indicators_NmF
        self.tol = tol
        
        self.N = weights_F.shape[0]
        self.m = weights_F.shape[1]
        self.F = weights_F.shape[2]
        
        # Precompute cumulative sums for efficiency
        self._precompute_cumulative_sums()
    


    def _precompute_cumulative_sums(self):
        """
        Precompute cumulative sums to enable O(m) V_n computation.
        
        For each quantile i:
        - past_sum[n, i] = sum_{s=1}^{n+1} w_i^s * I(theta_i < f_i^s)
        - future_sum[n, i] = sum_{s=n+2}^{N} w_i^s * (1 - I(theta_i < f_i^s))
        """
    
        # Compute from time 0 (index 0 is time 1 in the paper)
        weighted_indicators = (self.weighted_indicators_NmF).sum(axis=2) # (N, m, F) -> (N, m) (v2)
        self.past_cumsum = np.zeros((self.N + 1, self.m), dtype=np.float64)
        # Cumulative sum along time axis: axis=0
        # past_cumsum[n, i] = sum of first n elements (s=1 to n)
        for n in range(self.N + 1):
            if n==0:
                self.past_cumsum[0, :] = 0
            else:
                self.past_cumsum[n, :] = self.past_cumsum[n-1, :] + weighted_indicators[n-1, :] # weighted_indicators[s,i] = w_{i}^{s-1}
        
        # Future cumulative sum: sum_{s=n+1}^{N} w_i^s * (1 - I(theta_i < f_i^s))
        # future_cumsum[n, i] = sum of elements from n+1 to N (s=n+1 to N)
        weighted_indicators_complement = self.weights - weighted_indicators
        
        # Reverse cumulative sum
        self.future_cumsum = np.zeros((self.N + 1, self.m), dtype=np.float64)
        for n in range(self.N, 0, -1):
            if n == self.N:
                self.future_cumsum[self.N, :] = 0
            else:
                self.future_cumsum[n, :] = self.future_cumsum[n+1, :] + weighted_indicators_complement[n, :]    # s=n+1

    
    def compute_Vn(self, n: int, j_opt_min: int = 0) -> Tuple[float, int]:
        """
        Compute V_n for time step n.
        
        V_n = min_{j in [m]} (
            sum_{i=j}^{m-1} past_cost[i] + sum_{i=0}^{j-1} future_cost[i]
        )
        
        Parameters
        ----------
        n : int
        j_opt_min : int
            Minimum j* to consider (preventing crossing between j_n*)
        
        Returns
        -------
        V_n : float
            Minimum cost value
        j_opt : int
            Optimal cut point (0-indexed, j in [0, m])
        """
        if n == 0:
            return 0.0, 0,
        if n == self.N:
            return 0.0, self.m,
        assert 0 < n < self.N
        

        j_min = j_opt_min
        past_row = self.past_cumsum[n]
        future_row = self.future_cumsum[n]

        c0 = past_row[j_min:].sum()
        if j_min > 0:
            c0 += future_row[:j_min].sum()

        if j_min < self.m:
            inc = -past_row[j_min:self.m] + future_row[j_min:self.m]
            tail = c0 + np.cumsum(inc)
            costs_segment = np.empty(1 + tail.shape[0], dtype=np.float64)
            costs_segment[0] = c0
            costs_segment[1:] = tail
        else:
            costs_segment = np.array([c0], dtype=np.float64)

        j_rel = int(np.argmin(costs_segment))
        j_opt = j_min + j_rel
        min_cost = float(costs_segment[j_rel])

        
        # print(f'Vn compute at n={n}, j_opt_min={j_opt_min}, j_opt={j_opt}, min_cost={min_cost}')
        # print(costs)

        return min_cost, j_opt, 

    
    def compute_all_Vn(self) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Compute V_n for all n in [0, N].
        
        Returns
        -------
        V_n_values : np.ndarray
            Shape (N+1,) containing V_n for n=0 to N
        j_optimal : np.ndarray
            Shape (N+1,) containing optimal j* for each n
        """
        Vn_values = np.zeros(self.N + 1, dtype=np.float64)
        j_optimal = np.zeros(self.N + 1, dtype=np.int32)
        
        Vn_values[0] = 0.0
        j_optimal[0] = 0
        
        time2 = 0.0
        time3 = 0.0
        for n in range(1, self.N + 1):
            Vn, j_opt, t2, t3 = self.compute_Vn(n=n, j_opt_min=j_optimal[n-1])
            time2 += t2
            time3 += t3
            Vn_values[n] = Vn
            j_optimal[n] = j_opt
            
        print(f'time2: {time2}, time3: {time3}')

        return Vn_values, j_optimal


def j_opt_converter (j_opt: int, thetas: np.ndarray) -> float:
    thetas_gap = thetas[1] - thetas[0]
    return thetas[0] + (j_opt - 0.5) * thetas_gap

def single_q_minmax_solver2_v2(    # for multi-q case. Should be able to merge with single_q_minmax_solver
    theta_weights_F: np.ndarray,      # (m, F)
    weighted_indicators_m : np.ndarray,    # (m)
    thetas: np.ndarray,             # (m,)
    eq_value: float = 0.0,
    j_opt_pre: int = 0,        # minimum value of k_star, j_{n-1}^*
    j_opt_n: int = np.inf,   # maximum value of k_star, j_n^*
    tol: float = 1e-10,
) -> dict:
    """
    Solve the minmax problem for a single quantile level.
    Contains randomness in the choice of phat.
    """
    m = len(thetas)
    
    j_opt_pre = max(j_opt_pre, 0)
    j_opt_n = min(j_opt_n, m)
    assert j_opt_pre <= j_opt_n
    
    assert theta_weights_F.shape[0] == weighted_indicators_m.shape[0]
    
    theta_weights = theta_weights_F.sum(axis=1)
    # indicators = thetas[:, None] < forecast_values_F
    # weighted_indicators = (theta_weights_F * indicators).sum(axis=1)
    weighted_weights = np.sum(weighted_indicators_m)
    
    Bk_pre = np.concatenate([[-weighted_weights], np.cumsum(theta_weights) - weighted_weights])    # [0 (placeholder), B_1, ..., B_m=0]: Bk_pre[j] = sum_{i=0}^{j-1} w_i - \sum_{i=0}^{m-1} w_i * I(theta_i < f_i^s)
    
    if np.isclose(Bk_pre[j_opt_pre], eq_value, rtol=0, atol=tol):
        return {
            "phat": j_opt_converter(j_opt_pre, thetas),
            "k_star": j_opt_pre,
            "k_star_prob": 1.0,
        }
    # assert Bk_pre[j_opt_pre] < eq_value + tol, f"j_(n-1)*: {j_opt_pre}, Bk_pre[j_opt_pre]: {Bk_pre[j_opt_pre]}, eq_value: {eq_value}"
    # assert eq_value < Bk_pre[j_opt_n] + tol, f"j_n*: {j_opt_n}, Bk_pre[j_opt_n]: {Bk_pre[j_opt_n]}, eq_value: {eq_value}"
    # assert j_opt_pre < j_opt_n

    # Bk_pre[j] = \sum_{i=0}^{j-1} w_i - \sum_{i=0}^{m-1} w_i * I(theta_i < f_i^s). 
    # From theory, there exists k_n* s.t. j_{n-1}* <= k_n* < j_n* and \sum_{i=0}^{k_n*} w_i >= VALUE and \sum_{i=0}^{k_n*-1} w_i < VALUE.
    # So the biggest value of j to check Bk_pre[j] \geq VALUE is j_n*. Since Bk_pre[j_n*] = \sum_{i=0}^{j_n*-1 = <max value of k_n*>} w_i - \sum_{i=0}^{m-1} w_i \indi,
    for j in range(j_opt_pre+1, j_opt_n+1):  
        if Bk_pre[j] >= eq_value - tol:
            # Do we need this?
            # assert not np.isclose(Bk_pre[j-1], Bk_pre[j], rtol=0, atol=tol), f'Bk_pre[j-1]: {Bk_pre[j-1]}, Bk_pre[j]: {Bk_pre[j]}, eq_value: {eq_value}'
            k_star = j-1
            if np.isclose(eq_value, Bk_pre[j-1], rtol=0, atol=tol):
                k_star_prob = 0.0
            elif np.isclose(eq_value, Bk_pre[j], rtol=0, atol=tol):
                k_star_prob = 1.0
            else:
                # Scale numerator and denominator to avoid numerical instability before division
                num = eq_value - Bk_pre[j-1]
                denom = Bk_pre[j] - Bk_pre[j-1]
                scale = max(abs(num), abs(denom), 1.0)
                k_star_prob = (num / scale) / (denom / scale)
            phat = np.random.choice([j_opt_converter(k_star, thetas), j_opt_converter(k_star+1, thetas)], p=[k_star_prob, 1.0-k_star_prob])
            return {
                "phat": phat,
                "k_star": k_star,
                "k_star_prob": k_star_prob,
            }

    assert False, "Single-q search for multi-q optimiation ERROR. Should not reach here." + f'j_opt_pre: {j_opt_pre}, j_opt_n: {j_opt_n}, eq_value: {eq_value}, Bk_pre: {Bk_pre}'


def multi_q_minmax_solver_v2(
    theta_weights_F: np.ndarray,  # (N, m, F)
    thetas: np.ndarray,  # (m,)
    weighted_indicators_NmF: np.ndarray,  # (N, m, F)
    tol: float = 1e-10,
) -> dict:
    """
    Solve the minmax problem for a single quantile level.
    Contains randomness in the choice of phat.
    """
    N = theta_weights_F.shape[0]
    m = len(thetas)
    assert theta_weights_F.shape[1] == m
    assert weighted_indicators_NmF.shape == theta_weights_F.shape
    assert np.isclose(np.sum(theta_weights_F), 1.0)

    # First find V_n and j_n^*
    Vn_computer = VnComputer_v2(weights_F = theta_weights_F, 
                            thetas = thetas, 
                            weighted_indicators_NmF = weighted_indicators_NmF,  # (N, m, F)
                            tol=np.min(theta_weights_F.sum(axis=2))/3, # tol=tol,
                            )
    Vn_values, j_optimal =  Vn_computer.compute_all_Vn()

    # print(f"Vn_values: {Vn_values}")
    # print(f"j_optimal: {j_optimal}")
    phat_dict_list = []

    weighted_indicators_Nm = weighted_indicators_NmF.sum(axis=2) # (N, m, F) -> (N, m)
    for n in range(N):  # n=0 means first quantile (corresponds to n=1 in paper), eq_value = -V1, j_opt_pre = j_0^*
        phat_dict = single_q_minmax_solver2_v2(theta_weights_F=theta_weights_F[n,:,:],      # (m,F)
                            weighted_indicators_m=weighted_indicators_Nm[n,:],    # (m)
                            thetas=thetas,             # (m,)
                            eq_value=Vn_values[n] - Vn_values[n+1],     
                            j_opt_pre=j_optimal[n],
                            j_opt_n=j_optimal[n+1],
                            tol=np.min(theta_weights_F.sum(axis=2))/3, # tol=tol,
                            )
        phat_dict_list.append(phat_dict)

    return phat_dict_list, Vn_values


def minimax_value_neg(alpha_list: np.ndarray, Vn_values: np.ndarray) -> float:
    """
    Negation of minimax value of negative of the minimax value.
    = - \sum_{n=1}^{N-1} (\tau_{n+1} - \tau_n) V_n
    """
    N = alpha_list.shape[0]
    return np.sum(alpha_list * (Vn_values[0:N] - Vn_values[1:N+1]))