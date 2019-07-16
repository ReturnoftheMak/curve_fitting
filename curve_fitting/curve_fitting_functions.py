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
    """Import the data for fitting

    Args:
        Tablename (str): SQL table name
        sqlcon: connection to database
        Schema: table schema

    Returns:
        dataframe
    """

    df_rates = pd.read_sql_table(table_name, sqlcon, schema=schema)

    return df_rates


#%% Creating separate dfs

# Add more loops for keys such as aircraft and term

def arrays_to_fit(df_rates):
    """Provides arrays for curve fitting from dataframe

    Args:
        df (pandas.core.frame.DataFrame): from import_rates

    Returns:
        Two dictionaries with keys of credit rating, values of arrays
    """

    loan_to_value = {}
    expected_loss_cost = {}

    # Loop through each credit rating to create curves for each
    for rating in df_rates['Selected_Credit_Rating'].unique():

        for body_type in df_rates['Category'].unique():

            df_c = df_rates[(df_rates['Selected_Credit_Rating'] == rating) & (df_rates['Category'] == body_type)].reset_index()

            x_array = df_c['Loan_To_Value_Ratio']
            y_array = df_c['Expected_Loss_Cost']

            loan_to_value[rating+"_"+body_type] = x_array
            expected_loss_cost[rating+"_"+body_type] = y_array

    return loan_to_value, expected_loss_cost


#%% Second Degree Polynomial

def second_degree_polynomial(x_array, a_coef, b_coef, c_coef):
    """Simple form of order 2 polynomial

    Args:
        x: dependent variable
        a: 2nd order coefficient
        b: 1st order coefficient
        c: constant

    Returns:
        y for y = ax^2 + bx + c
    """

    return a_coef*(x_array**2) + b_coef*x_array + c_coef


#%% Return Parameters for each array

def fit_curve(x_array, y_array, func):
    """Fits a curve to the arrays based on a second order polynomial

    Args:
        x_array (pandas.core.series): dependent variable
        y_array (pandas.core.series): independent variable

    Returns:
        parameters
    """

    params, param_cov = optimize.curve_fit(func, x_array, y_array)

    return params


#%% Return parameter set for all arrays

def parameter_set(loan_to_value, expected_loss_cost, curve_func):
    """Returns a dictionary with all the curve parameters

    Args:
        LTV (dict): dictionary of credit rating keys and LTV arrays
        ELC (dict): dictionary of credit rating keys and ELC arrays

    Returns:
        Dictionary with key of ratings, parameter sets as values
    """

    param_set = {}

    for key in loan_to_value:

        x_array = loan_to_value[key]
        y_array = expected_loss_cost[key]

        params = fit_curve(x_array, y_array, curve_func)

        param_set[key] = params

    return param_set


#%% Dataframe manipulation for output to SQL

# With aircraft and term added into the key, we'll need to split out the column in here and do some more renaming

def param_set_manipulation(param_set):
    """Converts dictionary to dataframe

    Args:
        param_set (dict): parameter dictionary

    Returns:
        Parameter Dataframe
    """

    params = pd.DataFrame(param_set).transpose().reset_index()

    params = params.rename(columns={'index':'Credit_Rating', 0:'Coefficient_A', 1:'Coefficient_B', 2:'Coefficient_C'})

    return params


#%% Export to SQL

def export_params(params, sqlcon, export_table_name, export_schema):
    """Export to sql, overwriting previous table

    Args:
        params: Dataframe of parameters
        sqlcon: connection to database
        export_table_name (str): table name to use
        export_schema (str): schema to use

    Returns:
        none
    """

    params.to_sql(export_table_name, sqlcon, schema=export_schema, if_exists='replace', index=False)
