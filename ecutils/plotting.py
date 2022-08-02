"""
Utility Functions that can be used to manage colors and other plotting tools

Includes all stable utility functions.
"""

import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from collections import OrderedDict
from itertools import combinations


__all__ = [
    'plot_cmap_collections', 'plot_color_bar', 'get_color_mapper', 'plot_feature_scatter'
          ]


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


def plot_cmap_collections(cmap_collections=None):
    """Plot all color maps in the collections """
    if cmap_collections is None: cmap_collections = cmaps.keys()
    cmap_lists = [cmap_list for cmap_cat, cmap_list in cmaps.items() if cmap_cat in cmap_collections]
    nrows = max(len(cmap_list) for cmap_cat, cmap_list in cmaps.items() if cmap_cat in cmap_collections)
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    def plot_color_gradients(cmap_category, cmap_list, nrows):
        fig, axes = plt.subplots(nrows=nrows)
        fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
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
        plot_color_gradients(cmap_category, cmap_list, nrows)

    plt.show()


def plot_color_bar(cmap, figsize=15, series=None):
    if series is None: series = range(10)
    gradient = np.linspace(0, 1, len(series))
    gradient = np.vstack((gradient, gradient))
    _, ax = plt.subplots(figsize=(figsize,1))
    ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(cmap))
    # ax.scatter([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], c='#ffffff')
    for i, val in enumerate(series):
        ax.text(i, .7, str(val), color='#ffffff', fontsize='medium', fontweight=500, horizontalalignment='center')

    ax.set_xticks(series)
    ax.set_axis_off()
    plt.show()


def get_color_mapper(series, cmap='tab10'):
    """Return color mapper based on a color map and a series of values

    Used to assure coherent colors for different plots.
    Usage:
        clr_mapper = get_color_mapper([1, 2, 3, 4], cmap='Paired)
        clr_mapper.to_rgba(2) returns the appropriated value

    Example: Getting the same color as the one used for a clustering model and use to plot other values
            clustering = DBSCAN()
            clusters = clustering.fit_predict(embeds)
            cluster_ids = np.unique(clusters)
            cmap='Paired'
            plt.scatter(embeds[:, 0], embeds[:, 1], c=clusters, cmap=cmap)
            plt.colorbar()

            clr_mapper = get_color_mapper(cluster_ids, cmap=cmap)

            #   Plot another feature broken down as per the clustering
            featname = 'name of the feature to plot'
            for c in c_list:
                mask = df.cluster == c
                df[f"{featname}_{c}"] = df.loc[:, featname]
                df.loc[~mask, f"{featname}_{c}"] = np.nan
                ax.plot(df[f"{featname}_{c}"], label=str(c), c=clr_mapper.to_rgba(c))
            ax.set_title(f'{featname} w/ clusters highligthed')
            ax.legend()
            plt.show()
    """
    minimum, maximum = min(series), max(series)
    norm = colors.Normalize(vmin=minimum, vmax=maximum, clip=True)
    return cm.ScalarMappable(norm=norm, cmap=cm.get_cmap(cmap))


def plot_feature_scatter(X, y, n_plots, axes_per_row=3, axes_size=5):
    """Plots n_plots scatter plots of random combinations of two features out of X
    
    Randomly selects `n_plots` pairs out of all possible combinations of features pairs for the dataset X.T

    X, y:           the dataset. X.shape[1] is used to set the total number of features
    n_plots:        the number of feature pairs to plot as a scatter plot
    axes_per_row:   the number of axes per row to plot. number of rows will be calculated accordingly
                    default value is 3 axes per row
    axes_size:      the size of one axes. figsize will be (ncols * axes_size, nrows * axes_size)
                    default value is 5
    """

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