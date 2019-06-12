# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:38:00 2019

@author: Makhan.Gill
"""

# Curve Fitting for Aviation Finance


#%% Set Variables
import_server_name = 'tcspmSMDB01'
import_db_name = 'AviationFinanceResults'
table_name = 'AviationM'
import_schema = 'avR'
export_table_name = ''
export_schema = ''


#%% Define overall function

def fit_curves():
    """ Imports rates from SQL, fits curves, and exports the polynomial coefficients back to sql

    Args:
        None

    Returns:
        None
    """

    # Imports
    from sql_connection import sql_connection
    from curve_fitting_functions import import_rates, arrays_to_fit, parameter_set, param_set_manipulation, export_params

    sqlcon = sql_connection(import_server_name, import_db_name)

    df = import_rates(table_name, sqlcon, import_schema)

    LTV, ELC = arrays_to_fit(df)

    param_set = parameter_set(LTV, ELC)

    params = param_set_manipulation(param_set)

    export_params(params, sqlcon, export_table_name, export_schema)


if __name__ == "__main__":
    fit_curves()
