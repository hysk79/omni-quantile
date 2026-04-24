import numpy as np 
import pandas as pd

###############################
save_folder = 'hosp0420_weekly'
SAVE_DIR_OMNI = f'../results/{save_folder}'
SAVE_DIR_QL_PB = f'../results/hosp0420_weekly_ql_pb'

eta_list_omni = np.round(np.power(10, np.concatenate([[-3.0, -2.0], np.arange(-1.5, 1.6, 0.25)])), 4)
eta_list_ql = np.sort(np.round(np.power(10, np.concatenate([[-3.0, -2.0], np.arange(-1, 4.1, 0.5), np.arange(-0.25, 2.1, 0.5)])), 4))
###############################

ens_model_names = ['COVIDhub-4_week_ensemble', 'COVIDhub-trained_ensemble', 'JHUAPL-SLPHospEns']

geo_full_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'HI', 
'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 
'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 
'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'US']

geo_big = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI', 'NJ', 'VA', 'WA']
geo_med = ['AZ', 'TN', 'MA', 'IN', 'MO', 'MD', 'CO', 'WI', 'MN', 'SC', 'AL', 'LA', 'KY', 'OR', 'OK', 'CT']
geo_small =  ['UT', 'NV', 'IA', 'AR', 'KS', 'MS', 'NM', 'ID', 'NE', 'WV', 'HI', 'NH', 'ME', 'MT', 'RI', 'DE', 'SD', 'ND', 'AK', 'DC', 'VT', 'WY']
geo_list = geo_big + geo_med + geo_small + ['US']

alpha_list = np.array([0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99])

dates_list_pd = pd.date_range(start='2020-12-29', end='2023-05-30', freq='7D')
dates_list = dates_list_pd.strftime('%Y-%m-%d')
T = 127
F = 18
N = 23
