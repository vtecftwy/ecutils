import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

cmaps = OrderedDict()

cmaps['Perceptually Uniform Sequential'] = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis']
cmaps['Sequential'] = [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
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

# cmaps['Favs'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c']


def plot_cmap_collections(cmap_collections=None):
    """Plot all color maps in the collections """
    if cmap_collections is None: cmap_categories = cmaps.keys()
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