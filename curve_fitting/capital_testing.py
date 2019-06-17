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
    
    ((y - lognormal_dist(x, params[0], params[1], params[2]))**2).sum
    

def mean_error(params, mean):
    
    return abs(mean - params)


def minimuse_function(x,y):
    
    params = fit_curve(x, y, lognormal_dist)
    
    return least_squares(x,y) + mean_error(params, mean)









