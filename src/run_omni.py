# Used from 0315, with covid-trained_ensemble instead of COVIDhub-4_week_ensemble as competitor

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Callable, Tuple
import importlib
import metrics
import multi_q_minmax_solver
importlib.reload(metrics)
importlib.reload(multi_q_minmax_solver)
from metrics import elementary_scores_grid_N, elementary_scores_grid_N_m, elementary_scores_grid_N_F, pinball_loss, create_scoring_function_class
from multi_q_minmax_solver import multi_q_minmax_solver, minimax_value_neg


def omni_error_from_scores(scores: np.ndarray):
    # scores: (T, N, m, ..)
    # return: (T, ..)
    if scores.ndim == 3:
        return np.max(scores.cumsum(axis=0), axis=(1,2)) / (np.arange(scores.shape[0]) + 1)
    elif scores.ndim == 4:
        return np.max(scores.cumsum(axis=0), axis=(1,2)) / (np.arange(scores.shape[0])[:, None] + 1)
    else:
        raise ValueError(f"scores.ndim: {scores.ndim}")



# ============================================================================
# Main Omniprediction Experiment
# ============================================================================

# maybe to add unit size and flexible Y range here. Something like if Y value that is higher than current range of Y, add more \theta_is and put some weights there. (Should prove if we still have same Hedge performance bound)
def omniprediction_multiq_online(Y: pd.Series, forecasts_dict: dict, unit: int = 100,
                                alpha_list: List[float] = [0.5], eta_multiplier: float = 1,
                                eta_f_multiplier: float = 1, seed: int = 42, verbose: bool = False):
    """
    Run omniprediction experiment and compare against base forecasters.
    
    Parameters
    ----------
    T : int
        Time horizon / number of samples
    m : int
        Number of discretized theta values
    F : int
        Number of base forecasters
    alpha_list : List[float]
        List of quantile levels
    eta : float
        Learning rate for weights w_i
    eta_f : float
        Learning rate for forecaster selection v_{i,j}
    seed : int
        Random seed
    
    Returns
    -------
    results : dict
        Dictionary containing all results
    """
    if verbose:
        print("="*70)
        print("OMNIPREDICTION MULTI-QUANTILE EXPERIMENT")
        print("="*70)
    
    np.random.seed(seed)

    
    ###########################
    # NO-X ONLINE PART
    ###########################
    # Import Y and base forecasters
    dates_list = sorted(Y.index)
    T = len(dates_list)


    # Get the min and max of forecasters predictions
    forecast_min = np.inf
    forecast_max = -np.inf
    for f_name, forecast_dic in forecasts_dict.items():
        forecast_min = min(forecast_min, forecast_dic[min(alpha_list)].min())
        forecast_max = max(forecast_max, forecast_dic[max(alpha_list)].max())

    Y_rounded_min = min(int(np.floor(forecast_min / unit)), 0)
    Y_rounded_max = int(np.ceil(forecast_max / unit))
    m = Y_rounded_max - Y_rounded_min
    thetas = np.arange(Y_rounded_min, Y_rounded_max) + 0.5

    F = len(forecasts_dict.keys())
    forecaster_names = list(forecasts_dict.keys())

    alpha_list = np.array(alpha_list)
    assert len(alpha_list.shape) == 1
    N = alpha_list.shape[0]
    
    eta = eta_multiplier * np.sqrt(np.log(m*N)/T)
    eta_f = eta_f_multiplier * np.sqrt(np.log(F)/T) 

    for forecaster in forecaster_names:
        for alpha in alpha_list:
            if alpha not in forecasts_dict[forecaster].keys():
                raise ValueError(f"alpha {alpha} not in forecasts_dict[{forecaster}].keys()")
            assert set(forecasts_dict[forecaster][alpha].index) == set(Y.index), f'forecasts_dict[forecaster][alpha].index: {forecasts_dict[forecaster][alpha].index}, Y.index: {Y.index}'

    Y = (Y/unit).round(decimals=0).copy()
    
    forecasts_dict_rounded = {}
    for forecaster in forecaster_names:
        forecasts_dict_rounded[forecaster] = {}
        for alpha in alpha_list:
            forecasts_dict_rounded[forecaster][alpha] = (forecasts_dict[forecaster][alpha] / unit).round(decimals=0)
            
    all_forecaster_preds_all_dates = np.array([
        [
            [
                forecasts_dict_rounded[forecaster][alpha][date] 
                for forecaster in forecaster_names
            ] 
            for alpha in alpha_list
        ]
        for date in dates_list
    ])   # shape (T, N, F)

    if verbose:
        print(f"\nData:")
        print(f"  Number of dates: {T}")
        print(f"  Y range: [{min(Y.values):.3f}, {max(Y.values):.3f}]")
        print(f"  Unit size: {unit}")
        print(f"  Number of discretized thetas: {m}")
        print(f"  Number of base forecasters: {F}")
        print(f"  Forecaster names: {forecaster_names}")
        print(f"  Quantile level: alpha = {alpha_list} ({len(alpha_list)} levels)")
    
    ###########################


    # Initialize algorithm state
    w = np.ones((N, m)) / (N*m)  # Uniform weights over thetas
    v = np.ones((N, m, F)) / F  # Uniform weights over forecasters for each theta
    f_selected_indices = np.zeros((N, m), dtype=np.int32)  # Initially all use first forecaster
    
    pinball_v = np.ones((N,F)) / F  # each quantilie level separately
    pinball_f_selected_indices = np.zeros(N, dtype=np.int32)
    ql_v = np.ones(F) / F  # all quantile levels together
    ql_f_selected_index = 0
    
    # Storage for regrets over time
    y_arr = np.array(Y.values)
    phat_history = np.zeros((T, N))
    w_history = np.zeros((T, N, m))
    v_history = np.zeros((T, N, m, F))
    minimax_value_history = np.zeros((T,))
    omni_error_history = np.zeros((T, N, m))
    preds_history = np.zeros((T, N, F))     # Predictions of each forecaster
    forecasters_score_history = np.zeros((T, N, m, F))
    forecasters_selection_history = np.zeros((T, N, m))

    pinball_selection_history = np.zeros((T, N))
    pinball_preds_history = np.zeros((T, N))
    ql_selection_history = np.zeros((T))
    ql_preds_history = np.zeros((T, N))
    
    if verbose:
        print(f"\nRunning omniprediction algorithm...")
    
    for t, date in tqdm(enumerate(dates_list)):
        # x_t = X[t]
        y_t = Y[date]
        
        # Step 1: Compute P_t
        all_forecaster_preds = all_forecaster_preds_all_dates[t,:,:]
        
        # Select one forecaster per theta level (currently using index)
        # Use advanced NumPy indexing for compactness and speed
        row_idx = np.arange(N)[:, None]
        col_idx = f_selected_indices
        forecaster_preds = all_forecaster_preds[row_idx, col_idx]
        assert forecaster_preds.shape == (N, m)

        if np.min(w) < 1e-15:
            print(f"Minimum of weight is too small: {np.min(w)}")
        phat_dict_list, Vn_values = multi_q_minmax_solver(
            theta_weights=w,
            thetas=thetas,
            forecast_values=forecaster_preds,
            tol=1e-15
        )
        minimax_value_history[t] = minimax_value_neg(alpha_list=alpha_list, Vn_values=Vn_values)

        phat = np.array([phat_dict["phat"] for phat_dict in phat_dict_list])
        phat_history[t,:] = phat
        k_star = np.array([phat_dict["k_star"] for phat_dict in phat_dict_list])
        k_star_prob = np.array([phat_dict["k_star_prob"] for phat_dict in phat_dict_list])
        
        # Step 2: Compute expected score under P_t (vectorized)
        phat_score = k_star_prob[:, None] * elementary_scores_grid_N((k_star / m), y_t, thetas, alpha_list) + \
            (1 - k_star_prob[:, None]) * elementary_scores_grid_N((k_star + 1) / m, y_t, thetas, alpha_list)

        f_selected_score = elementary_scores_grid_N_m(forecaster_preds, y_t, thetas, alpha_list)
        
        assert phat_score.shape == (N, m), f'phat_score.shape: {phat_score.shape}, N: {N}, m: {m}'
        assert f_selected_score.shape == (N, m), f'f_selected_score.shape: {f_selected_score.shape}, N: {N}, m: {m}'
        
        # Step 3: Update weights w_i
        w_history[t,:] = w
        v_history[t,:,:] = v
        log_w = np.log(w + 1e-10)
        log_w += eta * (phat_score - f_selected_score)
        
        # Normalize in log space
        max_log_w = np.max(log_w)
        log_w -= max_log_w
        w = np.exp(log_w)
        w /= np.sum(w)
        
        # Step 4: Update forecaster selection v_{i,j} (vectorized implementation)
        preds = all_forecaster_preds    # (N, F)
        preds_history[t,:, :] = preds
        scores = elementary_scores_grid_N_F(preds, float(y_t), thetas, alpha_list)
        
        log_v = np.log(v + 1e-10)
        log_v -= eta_f * scores
        
        # Normalize in log-space for numerical stability
        max_log_v = np.max(log_v, axis=2, keepdims=True)
        log_v -= max_log_v
        
        v = np.exp(log_v)
        v /= np.sum(v, axis=2, keepdims=True)
        
        # Step 4-2: Hedge algorithm using pinball loss
        # Store selection history in vectorized manner
        pinball_selection_history[t, :] = pinball_f_selected_indices
        pinball_preds_history[t, :] = preds[np.arange(N), pinball_f_selected_indices]
        
        # Vectorized pinball loss computation and update
        # Compute pinball losses for all (N, F) at once
        pinball_losses = pinball_loss(preds, y_t, alpha_list[:, None])  # expects broadcasting: preds (N, F), alpha_list (N,1) -> (N,F)
        assert pinball_losses.shape == (N, F)

        # Vectorized update of pinball_v (N,F)
        pinball_v = np.log(pinball_v + 1e-10)
        pinball_v -= eta_f * pinball_losses / m
        pinball_v -= np.max(pinball_v, axis=1, keepdims=True)
        pinball_v = np.exp(pinball_v)
        pinball_v /= np.sum(pinball_v, axis=1, keepdims=True)

        # Vectorized forecaster selection for all n at once
        # Use cumulative sums for np.random.choice efficiency
        cum_v = np.cumsum(pinball_v, axis=1)
        r = np.random.rand(N, 1)
        pinball_f_selected_indices = (cum_v > r).argmax(axis=1)


        # Step 4-3: Hedge algorithm using QL loss
        ql_selection_history[t] = ql_f_selected_index
        ql_preds_history[t,:] = preds[:, ql_f_selected_index]
        ql_loss = np.mean(pinball_losses, axis=0)
        ql_v = np.log(ql_v + 1e-10)
        ql_v -= eta_f * ql_loss / m
        ql_v -= np.max(ql_v)
        ql_v = np.exp(ql_v)
        ql_v /= np.sum(ql_v)
        ql_f_selected_index = np.random.choice(F, p=ql_v)


        # Step 5: Sample new forecasters
        # for n in range(N):
        #     for i in range(m):
        #         j_ni = np.random.choice(F, p=v[n,i,:])
        #         f_selected_indices[n,i] = j_ni
        cum_v = np.cumsum(v, axis=2)          # shape (N, m, F)
        r = np.random.rand(N, m, 1)           # shape (N, m, 1)
        f_selected_indices = (cum_v > r).argmax(axis=2)  # shape (N, m)
    
        # Compute regret at time t
        forecasters_selection_history[t,:,:] = f_selected_indices
        forecasters_score_history[t,:,:,:] = scores
        # omni_error_history[t,:,:] = np.stack([[S(phat[n], y_t) for S in scoring_functions[n]] for n in range(N)])
        omni_error_history[t,:,:] = elementary_scores_grid_N(phat, y_t, thetas, alpha_list)


    # print(f"\n" + "="*70)
    # print("RESULTS")
    # print("="*70)

    # print(f"\nOmniprediction error (from learned P):")
    # print(f"  sup_{{S in S_alpha}} E[S(P(X), Y)] = {omni_error:.6f}")
    
    # # Compute regret of each base forecaster
    # print(f"\nRegret of each base forecaster:")
    # print(f"  sup_{{S in S_alpha}} E[S(f_j(X), Y)]")

    omni_score_trace = omni_error_from_scores(omni_error_history)
    assert omni_score_trace.shape == (T,)

    forecasters_score_trace = omni_error_from_scores(forecasters_score_history)
    assert forecasters_score_trace.shape == (T, F)

    best_forecaster_score_trace = forecasters_score_trace.min(axis=1)
    
    # Theoretical bound
    # theoretical_bound = np.sqrt(np.log(m * F) / T)
    # print(f"\nTheoretical omniprediction guarantee:")
    # print(f"  O(sqrt(log(m*F)/T)) = O(sqrt({np.log(m*F):.2f}/{T}))")
    # print(f"                      = O({theoretical_bound:.6f})")
    
    # Return results
    return {
        'phat_history': phat_history,
        # 'w_history': w_history,
        # 'v_history': v_history,
        'minimax_value_history': minimax_value_history, # (T,)
        # 'omni_error_history': omni_error_history,   # (T, N, m) 
        'forecasters_preds_history': preds_history, # (T, N, F)
        # 'forecasters_score_history': forecasters_score_history, # (T, N, m, F)
        # 'forecasters_selection_history': forecasters_selection_history, # (T, N, m)
        'omni_score_trace': omni_score_trace,   # (T,)
        'forecasters_score_trace': forecasters_score_trace, # (T, F)    
        'best_forecaster_score_trace': best_forecaster_score_trace, # (T,)
        'thetas': thetas,
    
        'pinball_selection_history': pinball_selection_history,     # (T, N)
        'pinball_preds_history': pinball_preds_history,             # (T, N)
        'ql_selection_history': ql_selection_history,               # (T,)
        'ql_preds_history': ql_preds_history,                       # (T, N)
        
        'Y': Y,
        'y_arr': y_arr,
        'dates_list': dates_list,
        'T': T,
        'unit': unit,
        'm': m,
        'F': F,
        'eta_multiplier': eta_multiplier,
        'eta_f_multiplier': eta_f_multiplier,
        'seed': seed,
        'alpha_list': alpha_list,
        'forecaster_names': forecaster_names,
        # 'theoretical_bound': theoretical_bound,
    }


class OmniResult():
    def __init__(self, results: dict):
        import importlib
        importlib.reload(metrics)
        from metrics import elementary_scores_grid_T_N
        self.results = results
        self.alpha_list = results['alpha_list']
        self.Y_arr = results['y_arr']
        self.m = results['m']
        self.F = results['F']
        self.T = results['T']
        self.N = self.alpha_list.shape[0]
        
        self.alpha_list = results['alpha_list']
        self.unit = results['unit']
        self.thetas = results['thetas']

        ens_preds = results['forecasters_preds_history'][:,:,-1] 
        assert results['forecaster_names'][-1] == 'COVIDhub-trained_ensemble'

        pinball_scores = elementary_scores_grid_T_N(self.results['pinball_preds_history'], self.Y_arr, self.thetas, self.alpha_list)
        ql_scores = elementary_scores_grid_T_N(self.results['ql_preds_history'], self.Y_arr, self.thetas, self.alpha_list)
        ensemble_scores = elementary_scores_grid_T_N(ens_preds, self.Y_arr, self.thetas, self.alpha_list)
        
        self.pinball_omni_score_trace = np.max(np.cumsum(pinball_scores, axis=0), axis=(1,2)) / np.arange(1, self.T+1)
        self.ql_omni_score_trace = np.max(np.cumsum(ql_scores, axis=0), axis=(1,2)) / np.arange(1, self.T+1)
        self.ensemble_omni_score_trace = np.max(np.cumsum(ensemble_scores, axis=0), axis=(1,2)) / np.arange(1, self.T+1)

        # Best Val and Best Test
        forecasters_preds_history = results['forecasters_preds_history']    # (T, N, F)
        forecasters_score_trace = results['forecasters_score_trace']   # (T, F)
        best_val_forecaster = np.concatenate([[0], np.argmin(forecasters_score_trace, axis=1)[:-1]])   # [0] as padding to look one step before
        best_test_forecaster = np.argmin(forecasters_score_trace, axis=1)

        self.best_val_forecaster_preds = forecasters_preds_history[np.arange(self.T), :, best_val_forecaster]
        self.best_test_forecaster_preds = forecasters_preds_history[np.arange(self.T), :, best_test_forecaster]   # (T, N)
        # best_val_forecaster_error = forecasters_score_trace[np.arange(T), best_val_forecaster]
        # best_test_forecaster_error = forecasters_score_trace[np.arange(T), best_test_forecaster]     # (T, )
        

    def simple_plot(self, ax=None):
        """Plot omniprediction experiment results."""
        assert self.results['forecaster_names'][-1] == 'COVIDhub-trained_ensemble'
        dates_list = pd.to_datetime(self.results['dates_list'])

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))

        ax.plot(dates_list, (self.results['omni_score_trace'] - self.results['best_forecaster_score_trace']), label='Omniprediction')
        # ax.plot(dates_list, results['best_forecaster_score_trace'], label='Best forecaster')
        ax.plot(dates_list, (self.pinball_omni_score_trace - self.results['best_forecaster_score_trace']), label='Pinball-opt')
        ax.plot(dates_list, (self.ql_omni_score_trace - self.results['best_forecaster_score_trace']), label='QL-opt')
        ax.plot(dates_list, (self.ensemble_omni_score_trace - self.results['best_forecaster_score_trace']), label='COVIDhub-ensemble')
        #ax.plot(dates_list, [np.minimum(np.sqrt(np.log(self.m * self.F) / (t+1)), 1.0) for t in np.arange(self.T)], '--', label='Theoretical bound')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Time t')
        ax.tick_params(axis='x', labelrotation=45)
        # ax.legend()

        return ax

    def minimax_plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        
        dates_list = pd.to_datetime(self.results['dates_list'])
        ax.plot(dates_list, self.results['minimax_value_history'], label='Minimax value')
        ax.set_xlabel('Time t')
        ax.tick_params(axis='x', labelrotation=45)
        ax.legend()

        return ax


    def single_q_pred_df (self, ia):
        return pd.DataFrame(
                np.concatenate(
                    [
                        self.results['phat_history'][:, ia][:, None],
                        self.results['forecasters_preds_history'][:, ia, :],
                        self.best_val_forecaster_preds[:, ia][:, None], 
                        self.best_test_forecaster_preds[:, ia][:, None],
                        self.results['pinball_preds_history'][:, ia][:, None], 
                        self.results['ql_preds_history'][:, ia][:, None], 
                        self.Y_arr[:, None]
                    ], axis=1
                ),
                columns = (['phat'] +
                            [f'pred_{i}' for i in range(1, self.results['forecasters_preds_history'].shape[2] + 1)] +
                            ['best_val_pred', 'best_test_pred', 'pinball_pred', 'ql_pred', 'Y']
                            )
            )


    def plot_prediction_panel(self, alpha, relative, ax=None):
        dates_list = pd.to_datetime(self.results['dates_list'])
        Y_arr = self.Y_arr
        unit = self.unit
        forecasters_preds_history = self.results['forecasters_preds_history']
        forecaster_names = self.results['forecaster_names']

        if isinstance(alpha, int):
            df = self.single_q_pred_df(alpha)
        else:
            ia = np.where(self.alpha_list == alpha)[0][0]
            df = self.single_q_pred_df(ia)

        if ax is None:
            fig, ax = plt.subplots(figsize=(18, 10))
        if not relative:
            assert forecaster_names[-1] == 'COVIDhub-trained_ensemble'

            for f in range(self.F):
                if f < self.F-1:
                    ax.plot(dates_list, unit*forecasters_preds_history[:, ia, f], color='gray', linewidth=1, linestyle='--', label=f'Base{forecaster_names[f]}')
                else:
                    ax.plot(dates_list, unit*forecasters_preds_history[:, ia, f], color='red', linewidth=1, label='COVIDhub-ens')

            ax.plot(dates_list, unit*Y_arr, color='black', linewidth=2, label='True Y')
            # ax.plot(dates_list, unit*df[f'best_test_pred'], color='red', linewidth=1, label='Best Base Forecaster (hindsight)')
            ax.plot(dates_list, unit*df['phat'], color='blue', linewidth=2, label='MultiQ Omniprediction')
            ax.plot(dates_list, unit*df[f'ql_pred'], color='green', linewidth=1, linestyle='--', label='(Unweighted) Quantile loss optimization')
            ax.set_ylabel('Predicted Y')
        else:
            ax.plot(dates_list, unit*(df[f'ql_pred'] - Y_arr), color='red', linewidth=2, label='MultiQ QL')
            ax.plot(dates_list, unit*(df[f'best_val_pred'] - Y_arr), color='green', linewidth=1, label='Best VAL')
            ax.plot(dates_list, unit*(df['phat'] - Y_arr), color='blue', linewidth=2, label='MultiQ Omniprediction')
            ax.hlines(0, dates_list[0], dates_list[-1], color='black', linewidth=0.5, linestyle='--')
            ax.set_ylabel('Predicted Y - True Y')
        ax.set_xlabel('Dates')
        ax.set_title(f'alpha = {alpha}')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


    def plot_prediction_panel_all(self, relative=False):
    
        fig, ax = plt.subplots(self.N, 1, figsize=(15, 15))
        for ia, alpha in enumerate(self.alpha_list):
            self.plot_prediction_panel(
                alpha=alpha, relative=relative, ax=ax[ia]
            )
        fig.tight_layout()
        plt.show()


    def quantile_plot(self, q_preds=None, f_name=None, color='tab:blue', alpha_list=None, ax=None, q_preds_to_add=None):
        if ax is None:
            ax = plt.gca()
        if alpha_list is None:
            alpha_list = self.results['alpha_list']
        if q_preds is None and f_name is None:
            raise ValueError("Either q_preds or f_name must be provided")
        if f_name:
            if q_preds is not None:
                print("Warning: q_preds is ignored because f_name is provided")
            if f_name == 'omni':
                q_preds = self.results['phat_history']
            elif f_name == 'pinball':
                q_preds = self.results['pinball_preds_history']
            elif f_name == 'ql':
                q_preds = self.results['ql_preds_history']
            elif f_name == 'ens':
                q_preds = self.results['forecasters_preds_history'][:,:,np.where(np.array(self.results['forecaster_names']) == 'COVIDhub-trained_ensemble')[0][0]]
            elif f_name == '4week_ens':
                q_preds = self.results['forecasters_preds_history'][:,:,np.where(np.array(self.results['forecaster_names']) == 'COVIDhub-4_week_ensemble')[0][0]]
            else:
                q_preds = self.results['forecasters_preds_history'][:,:,np.where(np.array(self.results['forecaster_names']) == f_name)[0][0]]
        
        for i in range((len(alpha_list)+1) // 2):
            if alpha_list[i] + alpha_list[len(alpha_list)-1-i] != 1:
                raise ValueError("alpha_list is not symmetric")

        for ia, alpha_q in enumerate(alpha_list): 
            if alpha_q > 0.5:
                break
            alpha = 0.2 + alpha_q*6/5
            ax.fill_between(pd.to_datetime(self.results['dates_list']), 
                            y1=q_preds[:,ia] * self.results['unit'], 
                            y2=q_preds[:,len(alpha_list)-1-ia] * self.results['unit'], 
                            color=color, alpha=alpha, linewidth=0
                            )
        ax.tick_params(axis='x', rotation=45)
        ax.plot(pd.to_datetime(self.results['dates_list']), self.results['Y']*self.results['unit'], color='black', linewidth=2, label='True Y')

        if q_preds_to_add is not None:
            return ax, q_preds_to_add+q_preds
        else:
            return ax, q_preds
            