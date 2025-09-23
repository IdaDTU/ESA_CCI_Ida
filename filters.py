# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 14:01:38 2025

@author: user
"""

import pandas as pd

def filter_df(df):
    """
    Filters a DataFrame by applying data quality checks on multiple 'tb_measured' columns.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'tb_measured' columns.

    Returns:
        pd.DataFrame: A new DataFrame with rows filtered based on specified criteria.
    """
    # Create a list of the 'tb_measured' column names
    tb_measured_cols = [col for col in df.columns if col.startswith('tb_measured_')]

    # Filter by a reasonable brightness temperature range (100 K to 300 K)
    for col in tb_measured_cols:
        df = df[(df[col] >= 100) & (df[col] <= 300)]
    
    # Remove rows with any NaN values across all columns
    df = df.dropna()

    # Apply outlier removal using the IQR method for each 'tb_measured' column
    for col in tb_measured_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df