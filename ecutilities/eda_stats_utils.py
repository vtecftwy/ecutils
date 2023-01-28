# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs-dev/1_01_eda_stats_utils.ipynb.

# %% ../nbs-dev/1_01_eda_stats_utils.ipynb 3
from __future__ import annotations
from pathlib import Path
from IPython.display import Image, display
from pprint import pprint
from scipy import stats
from scipy.cluster import hierarchy as hc
from typing import Optional

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil

# %% auto 0
__all__ = ['ecdf', 'cluster_columns']

# %% ../nbs-dev/1_01_eda_stats_utils.ipynb 5
def ecdf(
    data:pd.Series|np.ndarray,            # data to analyse 
    threshold:Optional[int] = None,     # cummulative frequency used as threshold. Must be between 0 and 1
    figsize:Optional[tuple[int,int]] = None  # figure size (width, height)
)-> tuple[np.array, np.array, int]:     # sorted data (ascending), cumulative frequencies, last index
    """Compute **Empirical Cumulative Distribution Function** (ECDF), plot it and returns values."""

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

# %% ../nbs-dev/1_01_eda_stats_utils.ipynb 18
def cluster_columns(df:pd.DataFrame,    # Multi-feature dataset with column names
                    figsize:tuple(int, int) = (10,6), # Size of the plotted figure
                    font_size:int = 12    # Font size for the chart on plot
                   ):
    """Plot dendogram based on Dataframe's columns' spearman correlation coefficients"""
    corr = np.round(stats.spearmanr(df).correlation, 4)
    corr_condensed = hc.distance.squareform(1-corr)
    z = hc.linkage(corr_condensed, method='average')
    fig = plt.figure(figsize=figsize)
    hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=font_size)
    plt.show()
