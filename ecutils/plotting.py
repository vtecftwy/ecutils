# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs-dev/0_02_plotting.ipynb.

# %% ../nbs-dev/0_02_plotting.ipynb 3
from __future__ import annotations
from collections import OrderedDict
from itertools import combinations

import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# %% auto 0
__all__ = ['cmaps', 'plot_cmap_collections', 'plot_color_bar', 'get_color_mapper', 'plot_feature_scatter']

# %% ../nbs-dev/0_02_plotting.ipynb 4
cmaps = OrderedDict()

cmaps['Perceptually Uniform Sequential'] = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis']
cmaps['Sequential'] = [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
cmaps['Diverging'] = [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
cmaps['Sequential (2)'] = [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']
cmaps['Cyclic'] = ['twilight', 'twilight_shifted', 'hsv']
cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']
cmaps['Miscellaneous'] = [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']

# %% ../nbs-dev/0_02_plotting.ipynb 5
def plot_cmap_collections(
    cmap_collections: str|list(str)=None  # list of color map collections to display (from cmaps.keys())
):
    """Plot all color maps in the collections passed as cmap_collections"""
    if cmap_collections is None: cmap_collections = cmaps.keys()
    cmap_lists = [cmap_list for cmap_cat, cmap_list in cmaps.items() if cmap_cat in cmap_collections]
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    def plot_color_gradients(cmap_category, cmap_list, nrows):
        fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(8, nrows * 0.3))
        fig.subplots_adjust(top=0.75, bottom=0.01, left=0.2, right=0.99)
        axes[0].set_title(cmap_category + ' colormaps', fontsize=14)

        for ax, name in zip(axes, cmap_list):
            ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
            pos = list(ax.get_position().bounds)
            x_text = pos[0] - 0.01
            y_text = pos[1] + pos[3]/2.
            fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

        # Turn off *all* ticks & spines, not just the ones with colormaps.
        for ax in axes:
            ax.set_axis_off()

    for cmap_category, cmap_list in zip(cmap_collections, cmap_lists):
        n_color_bars = len(cmap_list)
        plot_color_gradients(cmap_category, cmap_list, n_color_bars)

    plt.show()

# %% ../nbs-dev/0_02_plotting.ipynb 16
def plot_color_bar(
    cmap:str,                        # string name of one of the cmaps 
    series:list(int|float) = None    # series of numerical values to show for each color
):
    """Plots a color bar with value overlay"""
    if series is None: series = range(10)
    n_elements = len(series)
    gradient = np.linspace(0, 1, n_elements)
    gradient = np.vstack((gradient, gradient))
    _, ax = plt.subplots(figsize=(min(n_elements, 18),1))
    ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(cmap))
    for i, val in enumerate(series):
        ax.text(i, .7, str(val), color='#ffffff', fontsize='medium', fontweight=500, horizontalalignment='center')

    ax.set_xticks(series)
    ax.set_axis_off()
    plt.show()

# %% ../nbs-dev/0_02_plotting.ipynb 20
def get_color_mapper(
    series:list(int|float),    # series of values to map to colors  
    cmap:str = 'tab10'         # name of the cmap to use
):
    """Return color mapper based on a color map and a series of values"""
    minimum, maximum = min(series), max(series)
    norm = colors.Normalize(vmin=minimum, vmax=maximum, clip=True)
    return cm.ScalarMappable(norm=norm, cmap=cm.get_cmap(cmap))

# %% ../nbs-dev/0_02_plotting.ipynb 24
def plot_feature_scatter(
    X:np.array,              # data used to create plots
    y:np.array|None = None,  # target values
    n_plots:int = 2,         # number of scatter plots to create
    axes_per_row:int = 3,    # number of nbr of scatter plots per row
    axes_size:int = 5        # size of one scatter plot (square)
    ):
    """Plots n_plots scatter plots of random combinations of two features out of X"""

    if y is None: y = np.ones(shape=(X.shape[0],))
    pairs = np.array(list(combinations(range(X.shape[1]), 2)))
    idxs = np.random.randint(pairs.shape[0], size=n_plots)
    
    ncols = axes_per_row
    nrows = n_plots // axes_per_row + (1 if n_plots % axes_per_row > 0 else 0)
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*axes_size, nrows*axes_size))
    for (d1, d2), ax in zip(pairs[idxs, :], axs.reshape(ncols * nrows)):
        ax.scatter(X[:, d1], X[:, d2], c=y)
        ax.set_title(f"X_{min(d1, d2)} and X_{max(d1, d2)}")
        ax.set_xlabel(f"X_{d1}")
        ax.set_ylabel(f"X_{d2}")
    
    plt.show()
