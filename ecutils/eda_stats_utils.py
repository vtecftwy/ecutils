"""
Utility Functions that can be used for exploratory data analysis and statistics

Includes all stable utility functions for eda and statistics.
"""

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil

from pathlib import Path
from IPython.display import Image, display
from pprint import pprint


def ecdf(data, threshold=None, figsize=None):
    """Compute Empirical Cumulative Distribution Function (ECDF).
    
    Plots the ECDF, for values cumulative frequencies from 0 to threshold.
    Returns the values of x, y and last_idx, i.e. the index of x and y corresponding to the threshold

    Parameters:
      data: series or 1 dim array     data to analyze
      threshold: 0 < threshold < 1    cumulative frequency used as threshold
    """
    n = len(data)
    if threshold is None or int(threshold) == 1:
        last_idx = n
    elif 0 < threshold < 1:
        last_idx = min(int(np.floor(n * threshold)), n)
    else:
        print(f"threshold = {threshold}. Must be between 0 and 1.\nUsing threshold =  1 instead")
        last_idx = n

    # Data to plot on x-axis and y-axis
    x = np.sort(data)
    y = np.arange(1, n + 1) / n

    x_to_plot, y_to_plot = x[0:last_idx], y[0:last_idx]

    # Plot the ECDF
    figsize = (8, 4) if figsize is None else figsize
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_to_plot, y_to_plot, marker='.', linestyle='none')
    ax.plot([x[0], x_to_plot[-1]], [0, threshold], color='Gainsboro', linestyle='-', linewidth=1)
    ax.set_title(f"Empirical Cumulative Distribution Function \n(with threshold {threshold})")
    ax.xaxis.set_major_locator(plt.MaxNLocator(min_n_ticks=10))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.grid(b=True, which='both', axis='both', color='Gainsboro', linestyle='--', linewidth=1)
    plt.show()

    return x, y, last_idx

if __name__ == '__main__':
    df = pd.DataFrame(data={'a': np.random.random(100) * 100,'b': np.random.random(100) * 50,'c': np.random.random(100)})
    ecdf(data=df.a, threshold=1, figsize=(10, 10))
    ecdf(data=df.a, threshold=0.5, figsize=(10, 10))