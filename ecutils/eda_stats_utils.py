# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs-dev/1_01_eda_stats_utils.ipynb.

# %% ../nbs-dev/1_01_eda_stats_utils.ipynb 3
from __future__ import annotations
from pathlib import Path
from IPython.display import Image, display
from pprint import pprint

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil

# %% auto 0
__all__ = ['pandas_all_cols_and_rows', 'display_full_df', 'ecdf']

# %% ../nbs-dev/1_01_eda_stats_utils.ipynb 4
def pandas_all_cols_and_rows(f):
    """decorator function to force display of all the columns in a DataFrame, only for one time"""
    def wrapper(*args, **kwargs):
        max_rows = pd.options.display.max_rows
        max_cols = pd.options.display.max_columns
        pd.options.display.max_rows = None
        pd.options.display.max_columns = None
        f(*args, **kwargs)
        pd.options.display.max_rows = max_rows
        pd.options.display.max_columns = max_cols
    
    return wrapper

# %% ../nbs-dev/1_01_eda_stats_utils.ipynb 5
@pandas_all_cols_and_rows
def display_full_df(df:pd.DataFrame  # DataFrame to display
                   ):
    """Display a dataframe and shows all columns"""
    if not isinstance(df, pd.DataFrame): raise TypeError('df must me a pandas DataFrame')
    display(df)

# %% ../nbs-dev/1_01_eda_stats_utils.ipynb 8
def ecdf(
    data:pd.Series|np.array,            # data to analyse 
    threshold:int|None = None,          # cummulative frequency used as threshold. Must be between 0 and 1
    figsize:tuple(int,int)|None = None  # figure size (width, height)
)-> tuple(np.array, np.array, int):     # sorted data (ascending), cumulative frequencies, last index
    """Compute Empirical Cumulative Distribution Function (ECDF), plot it and returns values."""

    n = len(data)
    if threshold is None or int(threshold) == 1:
        last_idx = n -1
    elif 0 < threshold < 1:
        last_idx = min(int(np.floor(n * threshold)), n) - 1
    else:
        print(f"threshold = {threshold}. Must be between 0 and 1.\nUsing threshold =  1 instead")
        last_idx = n - 1

    # Data to plot on x-axis and y-axis
    x = np.sort(data)
    y = np.arange(1, n + 1) / n

    x_to_plot, y_to_plot = x[0:last_idx+1], y[0:last_idx+1]

    # Plot the ECDF
    figsize = (8, 4) if figsize is None else figsize
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_to_plot, y_to_plot, marker='.', linestyle='none')
    ax.plot([x[0], x_to_plot[-1]], [0, threshold], color='Gainsboro', linestyle='-', linewidth=1)
    ax.set_title(f"Empirical Cumulative Distribution Function \n(with threshold {threshold})")
    ax.xaxis.set_major_locator(plt.MaxNLocator(min_n_ticks=10))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.grid(visible=True, which='both', axis='both', color='Gainsboro', linestyle='--', linewidth=1)
    plt.show()

    return x, y, last_idx
