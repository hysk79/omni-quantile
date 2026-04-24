import matplotlib.pyplot as plt
import numpy as np

def color_func(total_n, idx):
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / (1.25*(total_n-1))) for i in range(2*(total_n-1))]
    return colors[idx]

def decimal_to_str(eta):
    return str(np.round(eta, 4)).replace('.', ',')
    
def exp_name_string_v2(w, eta, suffix=''):
    return f"wk{w}_eta{decimal_to_str(eta)}{suffix}"

def exp_name_string_ql_pb(w, eta, round_Y_F, suffix=''):
    return f"wk{w}_eta{decimal_to_str(eta)}{'_orig' if not round_Y_F else ''}{suffix}"

def exp_name_string_state_v2(w, geo, eta, suffix=''):
    return f"wk{w}_{geo}_eta{decimal_to_str(eta)}{suffix}"

def exp_name_string_state_ql_pb(w, geo, eta, round_Y_F, suffix=''):
    return f"wk{w}_{geo}_eta{decimal_to_str(eta)}{'_orig' if not round_Y_F else ''}{suffix}"