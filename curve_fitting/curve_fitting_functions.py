# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:38:00 2019

@author: Makhan.Gill
"""

# Curve Fitting for Aviation Finance
# help here https://www.geeksforgeeks.org/scipy-curve-fitting/

#%% Import packages

import pandas as pd
from scipy import optimize
#from matplotlib import pyplot as plt


#%% Read in Table

def import_rates(table_name, sqlcon, schema):
    """ Import the data for fitting

    Args:
        arg1 Tablename (str): SQL table name
        arg2 sqlcon: connection to database
        arg3 Schema: table schema

    Returns:
        dataframe
    """

    df = pd.read_sql_table(table_name, sqlcon, schema=schema)

    return df


#%% Creating separate dfs

# Add more loops for keys such as aircraft and term

def arrays_to_fit(df):
    """ Provides arrays for curve fitting from dataframe

    Args:
        arg1 df (pandas.core.frame.DataFrame): from import_rates

    Returns:
        Two dictionaries with keys of credit rating, values of arrays
    """

    LTV = {}
    ELC = {}

    # Loop through each credit rating to create curves for each
    for rating in df['Selected_Credit_Rating'].unique():

        df_c = df[df['Selected_Credit_Rating'] == rating].reset_index()

        x = df_c['Loan_To_Value_Ratio']
        y = df_c['Expected_Loss_Cost']

        LTV[rating] = x
        ELC[rating] = y

    return LTV, ELC


#%% Second Degree Polynomial

def second_degree_polynomial(x, a, b, c):
    """ Simple form of order 2 polynomial

    Args:
        arg1 x: dependent variable
        arg2 a: 2nd order coefficient
        arg3 b: 1st order coefficient
        arg4 c: constant

    Returns:
        y for y = ax^2 + bx + c
    """

    return a*(x**2) + b*x + c


#%% Return Parameters for each array

def fit_curve(x_array, y_array):
    """ Fits a curve to the arrays based on a second order polynomial

    Args:
        arg1 x_array (pandas.core.series): dependent variable
        arg2 y_array (pandas.core.series): independent variable

    Returns:
        parameters
    """

    params, param_cov = optimize.curve_fit(second_degree_polynomial, x_array, y_array)

    return params


#%% Return parameter set for all arrays

def parameter_set(LTV, ELC):
    """ Returns a dictionary with all the curve parameters

    Args:
        arg1 LTV (dict): dictionary of credit rating keys and LTV arrays
        arg2 ELC (dict): dictionary of credit rating keys and ELC arrays

    Returns:
        Dictionary with key of ratings, parameter sets as values
    """

    param_set = {}

    for key in LTV:

        x_array = LTV[key]
        y_array = ELC[key]

        params = fit_curve(x_array, y_array)

        param_set[key] = params

    return param_set


#%% Dataframe manipulation for output to SQL

# With aircraft and term added into the key, we'll need to split out the column in here and do some more renaming

def param_set_manipulation(param_set):
    """ Converts dictionary to dataframe

    Args:
        arg1 param_set (dict): parameter dictionary

    Returns:
        Parameter Dataframe
    """

    params = pd.DataFrame(param_set).transpose().reset_index()

    params = params.rename(columns={'index':'Credit_Rating', 0:'Coefficient_A', 1:'Coefficient_B', 2:'Coefficient_C'})

    return params


#%% Export to SQL

def export_params(params, sqlcon, export_table_name, export_schema):
    """ Export to sql, overwriting previous table

    Args:
        arg1 params: Dataframe of parameters
        arg2 sqlcon: connection to database
        arg3 export_table_name (str): table name to use
        arg4 export_schema (str): schema to use

    Returns:
        none
    """

    params.to_sql(export_table_name, sqlcon, schema=export_schema, if_exists='replace', index=False)
