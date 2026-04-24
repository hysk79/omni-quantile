# Runs QL-opt and Pinball-opt

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats.distributions import f_gen
from tqdm import tqdm
from typing import List, Callable, Tuple
import importlib
import metrics
importlib.reload(metrics)
from metrics import pinball_loss, elementary_scores_grid_T_N, elementary_scores_grid_N_F, ql_error_from_pb_loss


def omni_error_from_scores(scores: np.ndarray):
    # scores: (T, N, m, ..)
    # return: (T, ..)
    if scores.ndim == 3:
        return np.max(scores.cumsum(axis=0), axis=(1,2)) / (np.arange(scores.shape[0]) + 1)
    elif scores.ndim == 4:
        return np.max(scores.cumsum(axis=0), axis=(1,2)) / (np.arange(scores.shape[0])[:, None] + 1)
    else:
        raise ValueError(f"scores.ndim: {scores.ndim}")



def ql_pb_opt(Y: pd.Series, forecasts_dict: dict, unit: int = 100,
                                alpha_list: List[float] = [0.5], eta_multiplier: float = 1,
                             round_Y_F: bool = True):

    dates_list = sorted(Y.index)
    T = len(dates_list)

    # Setting up the range of Y and discretized thetas
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

    forecasts_dict_unit = {}
    for forecaster in forecaster_names:
        forecasts_dict_unit[forecaster] = {}
        for alpha in alpha_list:
            if round_Y_F:
                forecasts_dict_unit[forecaster][alpha] = (forecasts_dict[forecaster][alpha] / unit).round(decimals=0)
            else:
                forecasts_dict_unit[forecaster][alpha] = forecasts_dict[forecaster][alpha] / unit

    if round_Y_F:
        Y = (Y/unit).round(decimals=0).copy()
    else:
        Y = (Y/unit).copy()

    alpha_list = np.array(alpha_list)
    assert len(alpha_list.shape) == 1
    N = alpha_list.shape[0]
    eta = eta_multiplier * np.sqrt(np.log(F)/T)

    for forecaster in forecaster_names:
        for alpha in alpha_list:
            if alpha not in forecasts_dict[forecaster].keys():
                raise ValueError(f"alpha {alpha} not in forecasts_dict[{forecaster}].keys()")
            assert set(forecasts_dict[forecaster][alpha].index) == set(Y.index), f'forecasts_dict[forecaster][alpha].index: {forecasts_dict[forecaster][alpha].index}, Y.index: {Y.index}'

    all_forecaster_preds_all_dates = np.array([
        [
            [
                forecasts_dict_unit[forecaster][alpha][date]
                for forecaster in forecaster_names
            ] 
            for alpha in alpha_list
        ]
        for date in dates_list
    ])   # shape (T, N, F)
    assert all_forecaster_preds_all_dates.shape == (T, N, F)


    forecasters_escore_history = np.zeros((T, N, m, F))
    forecasters_pb_loss_history = np.zeros((T, N, F))
    for t, date in enumerate(dates_list):
        y_t = Y[date]
        forecasters_escore_history[t,:,:,:] = elementary_scores_grid_N_F(all_forecaster_preds_all_dates[t,:,:], y_t, thetas, alpha_list)
        forecasters_pb_loss_history[t,:,:] = pinball_loss(all_forecaster_preds_all_dates[t,:,:], y_t, alpha_list[:, None]) / m
    forecasters_score_trace = omni_error_from_scores(forecasters_escore_history)    # cumsum inside
    assert forecasters_score_trace.shape == (T, F)
    best_forecaster_score_trace = forecasters_score_trace.min(axis=1)


    results = {}
    # Initialize algorithm state
    pinball_v = np.ones((N,F)) / F  # each quantilie level separately
    ql_v = np.ones(F) / F  # all quantile levels together
    ql_v_history = np.zeros((T, F))
    
    # Storage for regrets over time
    y_arr = np.array(Y.values)
    forecasters_preds_history = np.zeros((T, N, F))     # Predictions of each forecaster
    pinball_preds_history = np.zeros((T, N))
    ql_preds_history = np.zeros((T, N))

    pinball_pb_loss_history = np.zeros((T, N))
    ql_pb_loss_history = np.zeros((T, N))
    
    for t, date in enumerate(dates_list):
        y_t = Y[date]
        
        forecaster_preds = all_forecaster_preds_all_dates[t,:,:]    # (N, F)
        forecasters_preds_history[t,:,:] = forecaster_preds
        assert forecaster_preds.shape == (N, F)        
        
        #########################################################
        # Pinball-opt update
        #########################################################
        pinball_preds_t = (forecaster_preds * pinball_v).sum(axis=1)
        pinball_preds_history[t, :] = pinball_preds_t
        pinball_pb_loss_history[t,:] = pinball_loss(pinball_preds_t, y_t, alpha_list) / m

        f_pb_losses = forecasters_pb_loss_history[t,:,:] # expects broadcasting: preds (N, F), alpha_list (N,1) -> (N,F)
        assert f_pb_losses.shape == (N, F)

        # Vectorized update of pinball_v (N,F)
        pinball_v = np.log(pinball_v + 1e-10)
        pinball_v -= eta * f_pb_losses
        pinball_v -= np.max(pinball_v, axis=1, keepdims=True)
        pinball_v = np.exp(pinball_v)
        pinball_v /= np.sum(pinball_v, axis=1, keepdims=True)

        #########################################################
        # QL-opt update
        #########################################################
        ql_preds_t = (forecaster_preds * ql_v[None, :]).sum(axis=1)
        ql_preds_history[t,:] = ql_preds_t
        ql_pb_loss_history[t,:] = pinball_loss(ql_preds_t, y_t, alpha_list) / m
        ql_v_history[t,:] = ql_v
        
        f_ql_loss = np.mean(f_pb_losses, axis=0)
        ql_v = np.log(ql_v + 1e-10)
        ql_v -= eta * f_ql_loss
        ql_v -= np.max(ql_v)
        ql_v = np.exp(ql_v)
        ql_v /= np.sum(ql_v)

    
    pinball_elementary_scores = elementary_scores_grid_T_N(pinball_preds_history, y_arr, thetas, alpha_list)
    ql_elementary_scores = elementary_scores_grid_T_N(ql_preds_history, y_arr, thetas, alpha_list)

    pinball_omni_score_trace = np.max(np.cumsum(pinball_elementary_scores, axis=0), axis=(1,2)) / np.arange(1, T+1)
    ql_omni_score_trace = np.max(np.cumsum(ql_elementary_scores, axis=0), axis=(1,2)) / np.arange(1, T+1)
    



    results = {
        'pinball_preds_history': pinball_preds_history,             # (T, N)
        'ql_preds_history': ql_preds_history,                       # (T, N)
        'ql_v_history': ql_v_history,                               # (T, F)

        'pinball_omni_history': np.max(pinball_elementary_scores, axis=(1,2)),
        'ql_omni_history': np.max(ql_elementary_scores, axis=(1,2)),

        'pinball_omni_score_trace': pinball_omni_score_trace,
        'ql_omni_score_trace': ql_omni_score_trace,
        'pinball_omni_score_trace_rel': pinball_omni_score_trace - best_forecaster_score_trace,
        'ql_omni_score_trace_rel': ql_omni_score_trace - best_forecaster_score_trace,
    
        'pinball_pb_loss_history': pinball_pb_loss_history, # (T, N)
        'ql_pb_loss_history': ql_pb_loss_history, # (T, N)
        # 'forecasters_pb_loss_history': forecasters_pb_loss_history, # (T, N, F)
        #'best_f_pb_loss_trace': ql_error_from_pb_loss(forecasters_pb_loss_history).min(axis=1), # (T,)

        'eta': eta,
        # 'forecasters_preds_history': forecasters_preds_history, # (T, N, F)   
        # 'best_forecaster_score_trace': best_forecaster_score_trace,
        'Y': Y,
        'round_Y_F': round_Y_F,
        'dates_list': dates_list,
        'T': T,
        'unit': unit,
        'm': m,
        'F': F,
        'alpha_list': alpha_list,
        'forecaster_names': forecaster_names,
    }

    return results