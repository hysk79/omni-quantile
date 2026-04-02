# This version runs omniprediction over weighted quantile losses. (Much faster without discretization)

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Callable, Tuple
import importlib
import metrics
import multi_q_minimax_solver_wql
importlib.reload(metrics)
importlib.reload(multi_q_minimax_solver_wql)
from metrics import pinball_loss
from multi_q_minimax_solver_wql import multi_q_minmax_solver_wql, minimax_value_neg

# ============================================================================
# Main Omniprediction Experiment
# ============================================================================

def omni_error_from_pb_loss(pb_loss: np.ndarray):
    # scores: (T, N, ..)
    # return: (T, ..)
    if pb_loss.ndim == 2:
        return np.max(pb_loss.cumsum(axis=0), axis=1) / (np.arange(pb_loss.shape[0]) + 1)
    elif pb_loss.ndim == 3:
        return np.max(pb_loss.cumsum(axis=0), axis=1) / (np.arange(pb_loss.shape[0])[:, None] + 1)
    else:
        raise ValueError(f"scores.ndim: {pb_loss.ndim}")


# maybe to add unit size and flexible Y range here. Something like if Y value that is higher than current range of Y, add more \theta_is and put some weights there. (Should prove if we still have same Hedge performance bound)
def omniprediction_multiq_wql(Y: pd.Series, forecasts_dict: dict, unit: int = 100,
                                alpha_list: List[float] = [0.5], eta_multiplier: float = 1,
                                seed_list: int = 42, verbose: bool = False):

    dates_list = sorted(Y.index)
    alpha_list = np.array(alpha_list)
    T = len(dates_list)
    N = alpha_list.shape[0]
    F = len(forecasts_dict.keys())
    eta = eta_multiplier * np.sqrt(np.log(N*F)/T)
    forecaster_names = list(forecasts_dict.keys())
    assert len(alpha_list.shape) == 1
    
    # Setting up the range of Y and discretized thetas
    forecast_min = np.inf
    forecast_max = -np.inf
    for f_name, forecast_dic in forecasts_dict.items():
        forecast_min = min(forecast_min, forecast_dic[min(alpha_list)].min())
        forecast_max = max(forecast_max, forecast_dic[max(alpha_list)].max())

    Y_rounded_min = min(int(np.floor(forecast_min / unit)), 0)
    Y_rounded_max = int(np.ceil(forecast_max / unit))
    m = Y_rounded_max - Y_rounded_min

    # Scaling the forecasts and Y
    Y = (Y/unit).copy()
    all_forecaster_preds_all_dates = np.array([
        [
            [
                forecasts_dict[forecaster][alpha][date] / unit
                for forecaster in forecaster_names
            ] 
            for alpha in alpha_list
        ]
        for date in dates_list
    ])   # shape (T, N, F)
    assert all_forecaster_preds_all_dates.shape == (T, N, F)


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

    forecasters_pb_loss_history = np.zeros((T, N, F)) 
    for t, date in enumerate(dates_list):
        forecasters_pb_loss_history[t,:,:] = pinball_loss(p=all_forecaster_preds_all_dates[t,:,:], y=Y[date], alpha=alpha_list[:, None]) / m
    forecasters_score_trace = omni_error_from_pb_loss(forecasters_pb_loss_history)
    best_forecaster_score_trace = forecasters_score_trace.min(axis=1)
    assert forecasters_score_trace.shape == (T, F)


    results = {}
    for seed in tqdm(seed_list, desc=f"Eta: {eta_multiplier}, Seeds"):
        np.random.seed(seed)
        
        # Initialize algorithm state
        w = np.ones((N, F)) / (N*F)  # Uniform weights over thetas and forecasters
        
        # Storage for regrets over time
        y_arr = np.array(Y.values)
        phat_history = np.zeros((T, N))
        w_history = np.zeros((T, N, F))
        minimax_value_history = np.zeros((T,))
        omni_pb_loss_history = np.zeros((T, N)) 

        if verbose:
            print(f"\nRunning omniprediction algorithm...")
        
        for t, date in enumerate(dates_list):
            y_t = Y[date]
            
            # Step 1: Compute P_t
            forecaster_preds = all_forecaster_preds_all_dates[t,:,:]    # (N, F)
            assert forecaster_preds.shape == (N, F)

            forecaster_preds = np.asarray(forecaster_preds, dtype=np.float64)
            
            if np.min(w) < 1e-15:
                print(f"Minimum of weight is too small: {np.min(w)}")
            phat, Vn_values = multi_q_minmax_solver_wql(
                weights_NF=w,
                forecasts_NF=forecaster_preds,
            )
            
            phat_history[t,:] = phat
            minimax_value_history[t] = minimax_value_neg(alpha_list=alpha_list, Vn_values=Vn_values)
        
            # Step 2: Compute expected score under P_t (vectorized) - QL grid
            
            f_scores = pinball_loss(p=forecaster_preds, y=y_t, alpha=alpha_list[:, None]) / m  # Consistent scaling
            assert f_scores.shape == (N, F), f'f_scores.shape: {f_scores.shape}, N: {N}, F: {F}'
        
            omni_pb_loss_history[t,:] = pinball_loss(p=phat, y=y_t, alpha=alpha_list) / m
            
            # Step 3: Update weights w_i
            w_history[t,:] = w
            log_w = np.log(w + 1e-12)
            log_w += eta * (phat[:,None] - f_scores)

            # Normalize in log space, then ensure min(w) = 1e-12 and sum(w) = 1
            max_log_w = np.max(log_w)
            log_w -= max_log_w
            w = np.exp(log_w)
            
            # Guarantee minimum of w is at least 1e-12
            w = np.maximum(w, 1e-12)
            w /= np.sum(w)


        omni_score_trace = omni_error_from_pb_loss(omni_pb_loss_history)
        assert omni_score_trace.shape == (T,)
        

        results[seed] = {
            'phat_history': phat_history,   # (T, N)
            'w_history': w_history,        # (T, N, F)
            'minimax_value_history': minimax_value_history, # (T,)
            
            'omni_pb_loss_history': omni_pb_loss_history,   # (T, N)
            'forecasters_pb_loss_history': forecasters_pb_loss_history, # (T, N, F)    

            'omni_score_trace': omni_score_trace,   # (T,)
            'omni_score_trace_rel': omni_score_trace - best_forecaster_score_trace[:,None],   # (T, F)
            'forecasters_score_trace': forecasters_score_trace, # (T, F)    
            'forecasters_score_trace_rel': forecasters_score_trace - best_forecaster_score_trace[:,None], # (T, F)    
        }
    settings = {
        'eta_multiplier': eta_multiplier,
        'seed_list': seed_list,
        'forecasters_preds_history': all_forecaster_preds_all_dates, # (T, N, F)
        'best_forecaster_score_trace': best_forecaster_score_trace, # (T,)
        'Y': Y,
        'y_arr': y_arr,
        'dates_list': dates_list,
        'T': T,
        'unit': unit,
        'm': m,
        'F': F,
        'alpha_list': alpha_list,
        'forecaster_names': forecaster_names,
    }

    return results, settings


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
            