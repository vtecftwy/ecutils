# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs-dev/0_01_ipython.ipynb.

# %% ../nbs-dev/0_01_ipython.ipynb 2
from __future__ import annotations
from IPython.core.getipython import get_ipython
from IPython.display import display, Markdown, display_markdown
from pathlib import Path
from typing import Any, List

import configparser
import numpy as np
import pandas as pd
import subprocess
import sys

# %% auto 0
__all__ = ['nb_setup', 'colab_install_project_code', 'files_in_tree', 'display_mds', 'display_dfs', 'run_cli', 'get_config_value']

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

# %% ../nbs-dev/0_01_ipython.ipynb 9
def files_in_tree(
    path: str|Path,               # path to the directory to scan  
    pattern: str|None = None      # pattern (glob style) to match in file name to filter the content
):
    """List files in directory and its subdiretories, print tree starting from parent directory"""
    pattern = '*' if pattern is None else f"*{pattern}*"
    parents = [p.name for p in path.parents]
    paths = []
    pad = ' ' * 2
    idx = 0
    print(f"{parents[0]}")
    print(f"{pad}|--{path.name}")
    for f in [p for p in path.glob(pattern) if p.is_file()]:
        paths.append(f)
        print(f"{pad}|{pad*2}|--{f.name} ({idx})")
        idx += 1
    for d in [p for p in path.iterdir() if p.is_dir()]:
        print(f"{pad}|{pad*2}|--{d.name}")
        for f in [p for p in d.glob(pattern) if p.is_file()]:
            paths.append(f)
            print(f"{pad}|{pad*2}|{pad*2}|--{f.name} ({idx})")
            idx += 1
    return paths

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

# %% ../nbs-dev/0_01_ipython.ipynb 17
def run_cli(cmd:str = 'ls -l'   # command to execute in the cli
           ):
    """Runs a cli command from jupyter notebook and print the shell output message
    
    Uses subprocess.run with passed command to run the cli command"""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
    print(str(p.stdout, 'utf-8'))

# %% ../nbs-dev/0_01_ipython.ipynb 19
def get_config_value(section:str,                        # section in the configparser cfg file
                     key:str,                            # key in the selected section
                     path_to_config_file:Path|str=None   # path to the cfg file
                    )-> Any :                            # the value corresponding to section>key>value 
    """Returns the value corresponding to the key-value pair in the configuration file (configparser format)"""
    # validate path_to_config_file
    if path_to_config_file is None:
        path_to_config_file = Path('/content/gdrive/MyDrive/private-across-accounts/config-api-keys.cfg')
    if isinstance(path_to_config_file, str): 
        path_to_config_file = Path(path_to_config_file)
    if not path_to_config_file.is_file():
        raise ValueError(f"No file at {path_to_config_file.absolute()}. Check the path")

    configuration = configparser.ConfigParser()
    configuration.read(path_to_config_file)
    return configuration[section][key]
