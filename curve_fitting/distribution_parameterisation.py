# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:38:21 2019

@author: Makhan.Gill
"""

#%% Package Imports

import numpy as np
from scipy.stats import gamma
from scipy.stats import lognorm
from scipy.stats import pareto, chi
import matplotlib.pyplot as plt


#%% Points for fitting

mean = 0.29
lejp = 0.375
uejp = 0.45


#%% Error functions

def gamma_square_error(a, loc, scale):

    distribution = gamma(a=a, loc=loc, scale=scale)

    square_errors = [np.power(mean-distribution.mean(), 2.0),
                     np.power(lejp-distribution.ppf(0.9), 2.0),
                     np.power(uejp-distribution.ppf(0.975), 2.0)]

    return square_errors


def lognorm_square_error(s, loc, scale):

    

    distribution = lognorm(s=s, loc=loc, scale=scale)

    square_errors = [np.power(mean-distribution.mean(), 2.0),
                     np.power(lejp-distribution.ppf(0.9), 2.0),
                     np.power(uejp-distribution.ppf(0.975), 2.0)]

    return square_errors


def pareto_square_error(b, loc, scale):
    
    distribution = pareto(b=b, loc=loc, scale=scale)
    
    square_errors = [np.power(mean-distribution.mean(), 2.0),
                     np.power(lejp-distribution.ppf(0.9), 2.0),
                     np.power(uejp-distribution.ppf(0.975), 2.0)]

    return square_errors


def chi_square_error(df, loc, scale):
    
    distribution = chi(df=df, loc=loc, scale=scale)
    
    square_errors = [np.power(mean-distribution.mean(), 2.0),
                     np.power(lejp-distribution.ppf(0.9), 2.0),
                     np.power(uejp-distribution.ppf(0.975), 2.0)]

    return square_errors


#%% Solver function for shape, loc, scale

def brute_force_solve(error_function, params):

    best_params = (1, 0, 1)
    best_sse = np.inf

    shape = np.linspace(params_log['shape_min'], params_log['shape_max'], num=50)
    loc = np.linspace(params_log['loc_min'], params_log['loc_max'], num=50)
    scale = np.linspace(params_log['scale_min'], params_log['scale_max'], num=50)

    for i in shape:
        for j in loc:
            for k in scale:

                sse = np.sum(error_function(i, j, k))

                params = (i, j, k)

                if best_sse > sse > 0:
                    best_params = params
                    best_sse = sse

    # Higher resolution solver

    # Previous step size

    shape_step = shape[1]-shape[0]
    loc_step = loc[1]-loc[0]
    scale_step = scale[1]-scale[0]

    shape_2 = np.linspace(best_params[0]-shape_step, best_params[0]+shape_step, num=50)
    loc_2 = np.linspace(best_params[1]-loc_step, best_params[1]+loc_step, num=50)
    scale_2 = np.linspace(best_params[2]-scale_step, best_params[2]+scale_step, num=50)

    for i in shape_2:
        for j in loc_2:
            for k in scale_2:

                sse = np.sum(error_function(i, j, k))

                params = (i, j, k)

                if best_sse > sse > 0:
                    best_params = params
                    best_sse = sse

    # Final level

    shape2_step = shape_2[1]-shape_2[0]
    loc2_step = loc_2[1]-loc_2[0]
    scale2_step = scale_2[1]-scale_2[0]

    shape_3 = np.linspace(best_params[0]-shape2_step, best_params[0]+shape2_step, num=50)
    loc_3 = np.linspace(best_params[1]-loc2_step, best_params[1]+loc2_step, num=50)
    scale_3 = np.linspace(best_params[2]-scale2_step, best_params[2]+scale2_step, num=50)

    for i in shape_3:
        for j in loc_3:
            for k in scale_3:

                sse = np.sum(error_function(i, j, k))

                params = (i, j, k)

                if best_sse > sse > 0:
                    best_params = params
                    best_sse = sse

    return best_params, best_sse


#%% Set Parameters, need some more of these

params_log = {'shape_min':0,
              'shape_max':10,
              'loc_min':0,
              'loc_max':5,
              'scale_min':0,
              'scale_max':2}


#%% Runs

best_params_log, best_gse_log = brute_force_solve(lognorm_square_error, params_log)

best_params_gamma, best_gse_gamma = brute_force_solve(gamma_square_error, params_log)

best_params_pareto, best_gse_pareto = brute_force_solve(pareto_square_error, params_log)

best_params_chi, best_gse_chi = brute_force_solve(chi_square_error, params_log)


#%% Plots

dist_gamma = gamma(a=best_params_gamma[0], loc=best_params_gamma[1], scale=best_params_gamma[2])

dist_lognorm = lognorm(s=best_params_log[0], loc=best_params_log[1], scale=best_params_log[2])

dist_pareto = pareto(b=best_params_pareto[0], loc=best_params_pareto[1], scale=best_params_pareto[2])

dist_chi = chi(df=best_params_chi[0], loc=best_params_chi[1], scale=best_params_chi[2])

lognorm_mu = np.log(best_params_log[2])
lognorm_sigma = best_params_log[0]

gamma_alpha = best_params_gamma[0]
gamma_beta = 1/best_params_gamma[2]

pareto_b = best_params_pareto[0]

chi_df = best_params_chi[0]

x = np.linspace(0, 1, num = 500)

plt.plot(x, dist_gamma.ppf(x), color ='red', label='gamma')
plt.plot(x, dist_lognorm.ppf(x), color ='blue', label='lognormal')
plt.plot(x, dist_pareto.ppf(x), color ='brown', label='pareto')
plt.plot(x, dist_chi.ppf(x), color ='purple', label='chi')

plt.scatter([0.9,0.975], [0.375, 0.45], color='green', label='judgements')
plt.hlines(mean, 0, 1, colors='black', linestyles='dashed', label='mean')
plt.legend()
plt.show()


#%% Discrete Functions - Large Loss Frequency

from scipy.stats import poisson

def poisson_square_error(mu, loc):
    
    distribution = poisson(mu=mu, loc=loc)
    
    square_errors = [np.power(mean-distribution.mean(), 2.0),
                     np.power(lejp-distribution.ppf(0.9), 2.0),
                     np.power(uejp-distribution.ppf(0.975), 2.0)]

    return square_errors


#%%



