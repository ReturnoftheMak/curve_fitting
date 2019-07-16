# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:38:00 2019

@author: Makhan.Gill
"""

# Curve Fitting for Aviation Finance


#%% Set Variables
IMPORT_SERVER_NAME = 'tcspmSMDB01'
IMPORT_DB_NAME = 'AviationFinanceResults'
TABLE_NAME = 'AviationM'
IMPORT_SCHEMA = 'avR'
EXPORT_TABLE_NAME = ''
EXPORT_SCHEMA = ''


#%% Define overall function

def fit_curves(import_server_name, import_db_name, table_name, import_schema, export_table_name, export_schema):
    """ Imports rates from SQL, fits curves, and exports the polynomial coefficients back to sql

    Args:
        import_server_name:
        import_db_name:
        table_name
        import_schema
        export_table_name
        export_schema

    Returns:
        None
    """

    # Imports
    from sql_connection import sql_connection
    from curve_fitting_functions import import_rates, arrays_to_fit, parameter_set, param_set_manipulation, export_params, second_degree_polynomial

    sqlcon = sql_connection(import_server_name, import_db_name)

    df_rates = import_rates(table_name, sqlcon, import_schema)

    loan_to_value, expected_loss_cost = arrays_to_fit(df_rates)

    param_set = parameter_set(loan_to_value, expected_loss_cost, second_degree_polynomial)

    params = param_set_manipulation(param_set)

    export_params(params, sqlcon, export_table_name, export_schema)


if __name__ == "__main__":
    fit_curves(IMPORT_SERVER_NAME, IMPORT_DB_NAME, TABLE_NAME, IMPORT_SCHEMA, EXPORT_TABLE_NAME, EXPORT_SCHEMA)
