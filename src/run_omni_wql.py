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
from metrics import pinball_loss, omni_error_from_pb_loss, omni_error_from_pb_loss_multiH
from multi_q_minimax_solver_wql import multi_q_minmax_solver_wql, minimax_value_neg, efficeint_solve_weighted_hinge_split, efficeint_solve_weighted_hinge_split_multiH

import time

# ============================================================================
# Main Omniprediction Experiment
# ============================================================================


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

    time_old = 0.0
    time_new = 0.0
    results = {}
    
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
        # SLOWER minimax solver
        # time1 = time.time()
        # phat, Vn_values = multi_q_minmax_solver_wql(
        #     weights_NF=w,
        #     forecasts_NF=forecaster_preds,
        # )
        # time_old += time.time() - time1
        # phat_history[t,:] = phat
        # minimax_value_history[t] = minimax_value_neg(alpha_list=alpha_list, Vn_values=Vn_values)
        
        # FASTER minimax solver 
        time1 = time.time()
        phat, Vn_diff_new = efficeint_solve_weighted_hinge_split(weights_NF=w, 
        forecasts_NF=forecaster_preds,
        )
        time_new += time.time() - time1
        phat_history[t,:] = phat
        minimax_value_history[t] = np.sum(alpha_list * Vn_diff_new)

        # Comparing two different solvers.
        # print(f'OLD Vn_diff: {Vn_values[:-1] - Vn_values[1:]}')
        # print(f'NEW Vn_diff: {Vn_diff_new}')
        # print(f'Vn diff diff: {Vn_values[:-1] - Vn_values[1:] - Vn_diff_new}')
        # assert np.allclose(np.sum(np.abs(Vn_values[:-1] - Vn_values[1:] - Vn_diff_new)), 0.0, rtol=0, atol=min(1e-11, w.min()/3)), f'abs sum: {np.sum(np.abs(Vn_values[:-1] - Vn_values[1:] - Vn_diff_new))}'
        
        # if np.max(np.abs(phat - phat_nw)) > 1e-10:
        #     print(f'Warning: phat and phat_nw differ by more than 1e-10, max diff: {np.max(np.abs(phat - phat_nw))}')
        
        
    
        # Step 2: Compute expected score under P_t (vectorized) - QL grid
        p_scores = pinball_loss(p=phat, y=y_t, alpha=alpha_list) / m
        f_scores = pinball_loss(p=forecaster_preds, y=y_t, alpha=alpha_list[:, None]) / m  # Consistent scaling
        assert p_scores.shape == (N,), f'p_scores.shape: {p_scores.shape}, N: {N}'
        assert f_scores.shape == (N, F), f'f_scores.shape: {f_scores.shape}, N: {N}, F: {F}'
        omni_pb_loss_history[t,:] = p_scores
        
        # Step 3: Update weights w_i
        w_history[t,:] = w
        log_w = np.log(w + 1e-12)
        log_w += eta * (p_scores[:,None] - f_scores)           
        

        # Normalize in log space, then ensure min(w) = 1e-12 and sum(w) = 1
        max_log_w = np.max(log_w)
        log_w -= max_log_w
        w = np.exp(log_w)
        
        # Guarantee minimum of w is at least 1e-12
        w = np.maximum(w, 1e-12)
        w /= np.sum(w)

    omni_score_trace = omni_error_from_pb_loss(omni_pb_loss_history)
    assert omni_score_trace.shape == (T,)
    

    results = {
        'phat_history': phat_history,   # (T, N)
        'forecasters_preds_history': all_forecaster_preds_all_dates, # (T, N, F)

        'w_history': w_history,        # (T, N, F)
        'minimax_value_history': minimax_value_history, # (T,)
        
        'omni_pb_loss_history': omni_pb_loss_history,   # (T, N)
        'forecasters_pb_loss_history': forecasters_pb_loss_history, # (T, N, F)    

        'best_forecaster_score_trace': best_forecaster_score_trace, # (T,)
        'omni_score_trace': omni_score_trace,   # (T,)
        'omni_score_trace_rel': omni_score_trace - best_forecaster_score_trace,   # (T,)
        'forecasters_score_trace': forecasters_score_trace, # (T, F)    
        'forecasters_score_trace_rel': forecasters_score_trace - best_forecaster_score_trace[:,None], # (T, F)    

        'eta_multiplier': eta_multiplier,
        
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

    return results




def omniprediction_multiq_wql_multiH(Y: pd.Series, H: int, forecasts_dict: dict, unit: int = 100,
                                alpha_list: List[float] = [0.5], eta_multiplier: float = 1,
                                verbose: bool = False, ):

    dates_list = sorted(Y.index)
    datas_list_arr = np.array([dates_list[i:i+H] for i in range(0, len(dates_list), H)])
    alpha_list = np.array(alpha_list)
    T = len(datas_list_arr)
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
    ])   # shape (T*H, N, F)
    assert all_forecaster_preds_all_dates.shape == (T*H, N, F)

    ###########################
    forecasters_pb_loss_history = np.zeros((T*H, N, F)) 
    for t, date in enumerate(dates_list):
        forecasters_pb_loss_history[t,:,:] = pinball_loss(p=all_forecaster_preds_all_dates[t,:,:], y=Y[date], alpha=alpha_list[:, None]) / m
    forecasters_pb_loss_history = forecasters_pb_loss_history.reshape(T, H, N, F)

    forecasters_score_trace = omni_error_from_pb_loss_multiH(forecasters_pb_loss_history)
    best_forecaster_score_trace = forecasters_score_trace.min(axis=1)
    assert forecasters_score_trace.shape == (T, F)

    results = {}
    # Initialize algorithm state
    w = np.ones((H, N, F)) / (H*N*F)  # Uniform weights over thetas and forecasters
    
    # Storage for regrets over time
    y_arr = np.array(Y.values)
    phat_history = np.zeros((T, H, N))
    w_history = np.zeros((T, H, N, F))
    minimax_value_history = np.zeros((T,))
    omni_pb_loss_history = np.zeros((T, H, N)) 

    if verbose:
        print(f"\nRunning omniprediction algorithm...")
    
    for t, dates_t in enumerate(datas_list_arr):
        y_t = Y[dates_t].values
        assert y_t.shape == (H,), f'y_t.shape: {y_t.shape}, H: {H}'
        
        # Step 1: Compute P_t
        forecaster_preds = all_forecaster_preds_all_dates[t*H:(t+1)*H,:,:]    # (H, N, F)
        assert forecaster_preds.shape == (H, N, F)
        forecaster_preds = np.asarray(forecaster_preds, dtype=np.float64)
        
        if np.min(w) < 1e-15:
            print(f"Minimum of weight is too small: {np.min(w)}")
        phat_all, Vn_diff_all = efficeint_solve_weighted_hinge_split_multiH(
            weights_HNF=w, 
            forecasts_HNF=forecaster_preds,
        )

        phat_history[t,:,:] = phat_all
        minimax_value_history[t] = np.sum(alpha_list[None,:] * Vn_diff_all)
    
        # Step 2: Compute expected score under P_t (vectorized) - QL grid
        p_scores = pinball_loss(p=phat_all, y=y_t[:,None], alpha=alpha_list[None,:]) / m
        f_scores = pinball_loss(p=forecaster_preds, y=y_t[:,None, None], alpha=alpha_list[None,:, None]) / m  # Consistent scaling
        assert p_scores.shape == (H, N), f'p_scores.shape: {p_scores.shape}, H: {H}, N: {N}'
        assert f_scores.shape == (H, N, F), f'f_scores.shape: {f_scores.shape}, H: {H}, N: {N}, F: {F}'
        omni_pb_loss_history[t,:,:] = p_scores
        
        # Step 3: Update weights w_i
        w_history[t,:,:,:] = w
        log_w = np.log(w + 1e-12)
        log_w += eta * (p_scores[:,:,None] - f_scores)           
        

        # Normalize in log space, then ensure min(w) = 1e-12 and sum(w) = 1
        max_log_w = np.max(log_w)
        log_w -= max_log_w
        w = np.exp(log_w)
        
        # Guarantee minimum of w is at least 1e-12
        w = np.maximum(w, 1e-12)
        w /= np.sum(w)

    omni_score_trace = omni_error_from_pb_loss_multiH(omni_pb_loss_history)
    assert omni_score_trace.shape == (T,)
    

    results = {
        'phat_history': phat_history,   # (T, H, N)
        'forecasters_preds_history': all_forecaster_preds_all_dates, # (T*H, N, F)

        'w_history': w_history,        # (T, H, N, F)
        'minimax_value_history': minimax_value_history, # (T,)
        
        'omni_pb_loss_history': omni_pb_loss_history,   # (T, H, N)
        'forecasters_pb_loss_history': forecasters_pb_loss_history, # (T*H, N, F)    

        'best_forecaster_score_trace': best_forecaster_score_trace, # (T,)
        'omni_score_trace': omni_score_trace,   # (T,)
        'omni_score_trace_rel': omni_score_trace - best_forecaster_score_trace,   # (T,)
        'forecasters_score_trace': forecasters_score_trace, # (T, F)    
        'forecasters_score_trace_rel': forecasters_score_trace - best_forecaster_score_trace[:,None], # (T, F)    

        'eta_multiplier': eta_multiplier,
        
        'Y': Y,
        'y_arr': y_arr,
        'dates_list': dates_list,
        'T': T,
        'unit': unit,
        'm': m,
        'F': F,
        'H': H,
        'N': N,
        'alpha_list': alpha_list,
        'forecaster_names': forecaster_names,
    }

    return results