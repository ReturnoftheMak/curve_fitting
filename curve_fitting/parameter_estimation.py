# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:10:40 2019

@author: Makhan.Gill
"""

# Initial Guesses for Distributions

#%% Import Packages

import numpy as np
import math


#%% Judgement Points

mean = 0.058
lejp = 0.10
uejp = 0.15
percentile_lower = 0.9
percentile_upper = 0.98
mean_error_weight = 7.0
lejp_error_weight = 3.0
uejp_error_weight = 2.0


#%% Initial Estimate

def variance_guess(mean, lejp, uejp):
    
    from scipy.stats import norm
    
    guess1 = (lejp - mean) / norm.ppf(percentile_lower)
    
    guess2 = (uejp - mean) / norm.ppf(percentile_upper)
    
    return (guess1+guess2)/2


#%% Parameter Estimation

# Beta Distribution - 2 params

def return_parameters_beta(mean, variance):
    
    alpha = (((1 - mean)/variance) - (1/mean)) * mean**2
    
    _beta = alpha * ((1/mean) - 1)
    
    return alpha, _beta

# Chi Squared - 1 param

# Not sure how to do this one as there are 2 equations, so an overspecified system

def return_parameter_chi2(mean, variance):
    
    k = mean
    
    return k

# Exponential - 1 param

def return_parameter_exp(mean, variance):
    
    _lambda = (1/mean)
    
    return _lambda

# Pareto - 1 param

def return_parameter_pareto(mean, variance):
    
    alpha = mean / (mean-1)
    
    return alpha

# Generalised Pareto - 1 parameter

def return_parameter_genpareto(mean, variance):
    
    shape = mean / (mean-1)
    
    return shape

# Gumbel - 2 Parameters

def return_parameters_gumbel(mean, variance):
    
    beta = np.sqrt((6/(math.pi**2)) * variance)
    
    mu = mean - beta * 0.5772
    
    return mu, beta

# Laplace - 2 Parameters

def return_parameters_laplace(mean, variance):
    
    mu = mean
    
    b = np.sqrt(0.5*variance)
    
    return mu, b

# Lognormal - 2 Parameters

def return_parameters_lognorm(mean, variance):
    
    mu = mean
    
    sigma = np.sqrt(variance)
    
    return mu, sigma

# Normal - 2 Parameters

def return_parameters_normal(mean, variance):
    
    mu = mean
    
    sigma_2 = variance
    
    return mu, sigma_2

# Rayleigh - 1 Parameter

def return_parameter_rayleigh(mean, variance):
    
    sigma = mean / (np.sqrt(math.pi/2))
    
    return sigma


# Gamma - 2 params
    
def return_parameters_gamma(mean, variance):
    
    theta = variance / mean
    
    alpha = mean**2 / variance
    
    return alpha, theta


#%% Error Functions - the ones I can work with Scipy and the specified cases of fixing loc and scale

def square_error_beta(alpha, _beta):
    
    from scipy.stats import beta

    distribution = beta(a=alpha, b=_beta)

    square_errors = [np.power(mean-distribution.mean(), 2.0)*mean_error_weight,
                     np.power(lejp-distribution.ppf(percentile_lower), 2.0)*lejp_error_weight,
                     np.power(uejp-distribution.ppf(percentile_upper), 2.0)*uejp_error_weight]

    return square_errors


def square_error_chi2(k):
    
    from scipy.stats import chi2

    distribution = chi2(df=k)

    square_errors = [np.power(mean-distribution.mean(), 2.0)*mean_error_weight,
                     np.power(lejp-distribution.ppf(percentile_lower), 2.0)*lejp_error_weight,
                     np.power(uejp-distribution.ppf(percentile_upper), 2.0)*uejp_error_weight]

    return square_errors


def square_error_exp(_lambda):
    
    from scipy.stats import expon

    distribution = expon(scale=1/_lambda)

    square_errors = [np.power(mean-distribution.mean(), 2.0)*mean_error_weight,
                     np.power(lejp-distribution.ppf(percentile_lower), 2.0)*lejp_error_weight,
                     np.power(uejp-distribution.ppf(percentile_upper), 2.0)*uejp_error_weight]

    return square_errors


def square_error_pareto(alpha):
    
    from scipy.stats import pareto

    distribution = pareto(b=alpha)

    square_errors = [np.power(mean-distribution.mean(), 2.0)*mean_error_weight,
                     np.power(lejp-distribution.ppf(percentile_lower), 2.0)*lejp_error_weight,
                     np.power(uejp-distribution.ppf(percentile_upper), 2.0)*uejp_error_weight]

    return square_errors


def square_error_genpareto(shape):
    
    from scipy.stats import genpareto

    distribution = genpareto(c=shape)

    square_errors = [np.power(mean-distribution.mean(), 2.0)*mean_error_weight,
                     np.power(lejp-distribution.ppf(percentile_lower), 2.0)*lejp_error_weight,
                     np.power(uejp-distribution.ppf(percentile_upper), 2.0)*uejp_error_weight]

    return square_errors


def square_error_lognorm(mu, sigma):
    
    from scipy.stats import lognorm

    distribution = lognorm(s=sigma, scale=np.exp(mu))

    square_errors = [np.power(mean-distribution.mean(), 2.0)*mean_error_weight,
                     np.power(lejp-distribution.ppf(percentile_lower), 2.0)*lejp_error_weight,
                     np.power(uejp-distribution.ppf(percentile_upper), 2.0)*uejp_error_weight]

    return square_errors


def square_error_gamma(alpha, theta):
    
    from scipy.stats import gamma

    distribution = gamma(a=alpha, scale=theta)

    square_errors = [np.power(mean-distribution.mean(), 2.0)*mean_error_weight,
                     np.power(lejp-distribution.ppf(percentile_lower), 2.0)*lejp_error_weight,
                     np.power(uejp-distribution.ppf(percentile_upper), 2.0)*uejp_error_weight]

    return square_errors


#%% Solver

def parameter_solver(mean, variance, estimator_func, error_func):
    
    if type(estimator_func(mean, variance)) == float:
        
        param_estimate = estimator_func(mean, variance)
        
        best_params = (param_estimate)
        best_sse = np.inf
        
        param_range = np.linspace(param_estimate-10, param_estimate+10, num=100)
        
        for i in param_range:
            
            sse = np.sum(error_func(i))
                    
            params = (i)
                    
            if best_sse > sse > 0:
                best_params = params
                best_sse = sse
            
        
        param_step = param_range[1]-param_range[0]
        
        param_range2 = np.linspace(best_params-param_step, best_params+param_step, num=400)
        
        for i in param_range2:
            
            sse = np.sum(error_func(i))
                    
            params = (i)
                    
            if best_sse > sse > 0:
                best_params = params
                best_sse = sse


    elif len(estimator_func(mean, variance)) == 2:
        
        param1_estimate, param2_estimate = estimator_func(mean, variance)
        
        best_params = (param1_estimate, param2_estimate)
        best_sse = np.inf
        
        param1_range = np.linspace(param1_estimate-10, param1_estimate+10, num=100)
        param2_range = np.linspace(param2_estimate-10, param2_estimate+10, num=100)
        
        for i in param1_range:
            for j in param2_range:

                    
                sse = np.sum(error_func(i, j))
                    
                params = (i, j)
                    
                if best_sse > sse > 0:
                    best_params = params
                    best_sse = sse
                
        
        param1_step = param1_range[1]-param1_range[0]
        param2_step = param2_range[1]-param2_range[0]
        
        param1_range2 = np.linspace(best_params[0]-param1_step, best_params[0]+param1_step, num=400)        
        param2_range2 = np.linspace(best_params[0]-param2_step, best_params[0]+param2_step, num=400)
        
        for i in param1_range2:
            for j in param2_range2:

                sse = np.sum(error_func(i, j))
                    
                params = (i, j)
                    
                if best_sse > sse > 0:
                    best_params = params
                    best_sse = sse
        
    return best_params, best_sse


#%% Run Solver for a given estimator func and error func

variance = variance_guess(mean, lejp, uejp)

beta_params, beta_sse = parameter_solver(mean, variance, return_parameters_beta, square_error_beta)
chi2_params, chi2_sse = parameter_solver(mean, variance, return_parameter_chi2, square_error_chi2)
exp_params, exp_sse = parameter_solver(mean, variance, return_parameter_exp, square_error_exp)
pareto_params, pareto_sse = parameter_solver(mean, variance, return_parameter_pareto, square_error_pareto)
genpareto_params, genpareto_sse = parameter_solver(mean, variance, return_parameter_genpareto, square_error_genpareto)
lognorm_params, lognorm_sse = parameter_solver(mean, variance, return_parameters_lognorm, square_error_lognorm)
gamma_params, gamma_sse = parameter_solver(mean, variance, return_parameters_gamma, square_error_gamma)


#%% Summary and plots

from scipy.stats import gamma

dist_gamma = gamma(a=gamma_params[0], scale=gamma_params[0])

from scipy.stats import lognorm

dist_lognorm = lognorm(s=lognorm_params[1], scale=np.exp(lognorm_params[0]))

from scipy.stats import pareto

dist_pareto = pareto(b=pareto_params)

from scipy.stats import chi2

dist_chi2 = chi2(df=chi2_params)

from scipy.stats import genpareto

dist_genpareto = genpareto(c=genpareto_params)

from scipy.stats import expon

dist_expon = expon(scale=1/exp_params)

x = np.linspace(0, 1, num = 500)

import matplotlib.pyplot as plt

# Use # to uncomment the non-relevant lines and rerun this portion only to get a better view of the fit

#plt.plot(x, dist_gamma.ppf(x), color ='red', label='gamma')
#plt.plot(x, dist_pareto.ppf(x), color ='brown', label='pareto')
#plt.plot(x, dist_chi2.ppf(x), color ='purple', label='chi')
#plt.plot(x, dist_genpareto.ppf(x), color ='purple', label='chi')
plt.plot(x, dist_lognorm.ppf(x), color ='blue', label='lognormal')
plt.plot(x, dist_expon.ppf(x), color ='red', label='exp')

plt.scatter([percentile_lower,percentile_upper], [lejp, uejp], color='green', label='judgements')
plt.hlines(mean, 0, 1, colors='black', linestyles='dashed', label='mean')
plt.legend()
plt.show()

square_error_results = {'beta':beta_sse,
                        'chi_squared':chi2_sse,
                        'exponential':exp_sse,
                        'gamma':gamma_sse,
                        'general pareto':genpareto_sse,
                        'lognormal':lognorm_sse,
                        'pareto':pareto_sse}


sorted_results = [(key, square_error_results[key]) for key in sorted(square_error_results, key=square_error_results.__getitem__)]

print(sorted_results)

print(exp_params)
print(lognorm_params)
