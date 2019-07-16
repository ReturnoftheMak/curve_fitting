# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:15:31 2019

@author: Makhan.Gill
"""

import pandas as pd
from scipy import optimize
import numpy as np 
from scipy import stats
from scipy.stats import lognorm

x_test = 2 * np.random.randn(10000) + 7.0 # normally  distributed values
y_test = np.exp(x_test) # these values have lognormal distribution

def fit_curve(x_array, y_array, func):
    """ Fits a curve to the arrays based on a second order polynomial

    Args:
        arg1 x_array (pandas.core.series): dependent variable
        arg2 y_array (pandas.core.series): independent variable

    Returns:
        parameters
    """

    params, param_cov = optimize.curve_fit(func, x_array, y_array)

    return params

mean = 29.0

lejp = (1.0-1.0/10.0, 37.5)
hejp = (1.0-1.0/40.0, 45.0)

x = pd.Series(np.array([lejp[0], hejp[0]]))
y = pd.Series(np.array([lejp[1], hejp[1]]))

x = pd.Series([0.9, 0.55, 0.975])
y = pd.Series([37.5, 29.0, 45.0])

def lognormal_dist(x, s, loc, scale):
    
    return lognorm.pdf(x, s, loc, scale)

params = fit_curve(x, y, lognormal_dist)

def least_squares(x,y):
    
    params = fit_curve(x, y, lognormal_dist)
    
    ((y - lognormal_dist(x, params[0], params[1], params[2]))**2).sum()
    

def mean_error(params, mean):
    
    return abs(mean - params)


def minimuse_function(x,y):
    
    params = fit_curve(x, y, lognormal_dist)
    
    return least_squares(x,y) + mean_error(params, mean)


#%%

from matplotlib import pyplot as plt


x = np.linspace(0, 1, num = 500)

y = 3.45 * np.sin(1.334 * x) + np.random.normal(size = 40)

def test(x, a, b):
    return a * np.sin(b * x)
  

param, param_cov = optimize.curve_fit(test, x, y)
  

ans = (param[0]*(np.sin(param[1]*x)))



plt.plot(x, y, 'o', color ='red', label ="data")
plt.plot(x, ans, '--', color ='blue', label ="optimized data")
plt.legend()
plt.show()


#%%

from scipy.stats import gamma

mean = 0.29

logn = lognorm(s=0.239, loc=0, scale=np.exp(-1.266))

rv = gamma(a=15.76, loc=0, scale=0.0184)

plt.plot(x, rv.pdf(x), color ='red', label='gamma')
plt.plot(x, logn.pdf(x), color ='blue', label='lognormal')
plt.legend()
plt.show()


#%% Need an error function to minimise for given arguments of loc, scale, and optional other arguments

x = np.array([0.9, 0.975])

y = np.array([0.375, 0.45])

mean = 0.29
lejp = 0.375
uejp = 0.45

mean_square_error = np.sum(
                           [np.power(mean-rv.mean(), 2.0),
                            np.power(lejp-rv.ppf(0.9), 2.0),
                            np.power(uejp-rv.ppf(0.975), 2.0)]
                           )

square_errors = [np.power(mean-rv.mean(), 2.0),
                 np.power(lejp-rv.ppf(0.9), 2.0),
                 np.power(uejp-rv.ppf(0.975), 2.0)]


#%%

#from scipy.optimize import minimize
#
#def gamma_square_error(x, shape, loc, scale):
#    
#    distribution = gamma(a=shape, loc=loc, scale=scale)
#    
#    square_errors = [np.power(mean-distribution.mean(), 2.0),
#                     np.power(lejp-distribution.ppf(0.9), 2.0),
#                     np.power(uejp-distribution.ppf(0.975), 2.0)]
#    
#    return square_errors
#
#initial_guess = [10,0,1]
#
#result = minimize(gamma_square_error, initial_guess, args=('shape','loc','scale'))


#%%

def gamma_square_error(a, loc, scale):

    from scipy.stats import gamma

    distribution = gamma(a=a, loc=loc, scale=scale)

    square_errors = [np.power(mean-distribution.mean(), 2.0),
                     np.power(lejp-distribution.ppf(0.9), 2.0),
                     np.power(uejp-distribution.ppf(0.975), 2.0)]

    return square_errors

def lognorm_square_error(s, loc, scale):

    from scipy.stats import lognorm

    distribution = lognorm(s=s, loc=loc, scale=scale)

    square_errors = [np.power(mean-distribution.mean(), 2.0),
                     np.power(lejp-distribution.ppf(0.9), 2.0),
                     np.power(uejp-distribution.ppf(0.975), 2.0)]

    return square_errors


def brute_force_solve(error_function):

    best_params = (1, 0, 1)
    best_gse = np.inf

    shape = np.linspace(1, 16, num=50)
    loc = np.linspace(0, 5, num=50)
    scale = np.linspace(0, 2, num=50)

    for i in shape:
        for j in loc:
            for k in scale:

                gse = np.sum(error_function(i, j, k))

                params = (i, j, k)

                if best_gse > gse > 0:
                    best_params = params
                    best_gse = gse

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

                gse = np.sum(error_function(i, j, k))

                params = (i, j, k)

                if best_gse > gse > 0:
                    best_params = params
                    best_gse = gse

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

                gse = np.sum(error_function(i, j, k))

                params = (i, j, k)

                if best_gse > gse > 0:
                    best_params = params
                    best_gse = gse

    return best_params, best_gse


#%%

best_params_log, best_gse_log = brute_force_solve(lognorm_square_error)

best_params_gamma, best_gse_gamma = brute_force_solve(gamma_square_error)

rv = gamma(a=best_params_gamma[0], loc=best_params_gamma[1], scale=best_params_gamma[2])

ln = lognorm(s=best_params_log[0], loc=best_params_log[1], scale=best_params_log[2])

plt.plot(x, rv.ppf(x), color ='red', label='gamma')
plt.plot(x, ln.ppf(x), color ='blue', label='lognormal')
plt.legend()
plt.show()
