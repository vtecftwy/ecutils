# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs-dev/0_01_ipython.ipynb.

# %% ../nbs-dev/0_01_ipython.ipynb 2
from __future__ import annotations
from IPython.core.getipython import get_ipython
from IPython.display import display, Markdown, display_markdown
from pathlib import Path

import numpy as np
import pandas as pd
import sys

# %% auto 0
__all__ = ['nb_setup', 'colab_install_project_code', 'display_mds', 'display_dfs']

# %% ../nbs-dev/0_01_ipython.ipynb 4
def nb_setup(autoreload:bool = True,   # True to set autoreload in this notebook
             paths:list(Path) = None   # Paths to add to the path environment variable
            ):
    """Use in first cell of notebook to set autoreload, and paths"""
#   Add paths. Default is 'src' if it exists
    if paths is None:
        p = Path('../src').resolve().absolute()
        if p.is_dir():
            paths = [str(p)]
        else:
            paths=[]
    if paths:
        for p in paths:
            sys.path.insert(1, str(p))
        print(f"Added following paths: {','.join(paths)}")

#   Setup auto reload
    if autoreload:
        ipshell = get_ipython()
        ipshell.run_line_magic('load_ext',  'autoreload')
        ipshell.run_line_magic('autoreload', '2')
        print('Set autoreload mode')

# %% ../nbs-dev/0_01_ipython.ipynb 6
def colab_install_project_code(
    package_name:str # project package name, e.g. git+https://github.com/vtecftwy/metagentools.git@main
):
    """When nb is running on colab, pip install the project code package"""
    try:
        from google.colab import drive
        ON_COLAB = True
        print('The notebook is running on colab')
        print('Installing project code')
        cmd = f"pip install -U {package_name}"
        run(cmd)

    except ModuleNotFoundError:
        ON_COLAB = False
        print('The notebook is running locally, will not automatically install project code')

    return ON_COLAB

# %% ../nbs-dev/0_01_ipython.ipynb 12
def display_mds(
    *strings:str # any number of strings with text in markdown format
):
    """Display one or several strings formatted in markdown format"""
    for string in strings:
        display_markdown(Markdown(data=string))

# %% ../nbs-dev/0_01_ipython.ipynb 15
def display_dfs(*dfs:pd.DataFrame       # any number of Pandas DataFrames
               ):
    """Display one or several DataFrame in a single cell output"""
    for df in dfs:
        display(df)
