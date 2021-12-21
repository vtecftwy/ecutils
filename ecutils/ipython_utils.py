"""
This file includes a set of utility function to be used in Jupyter Notebook and Jupyter Lab notebooks

{License_info}
"""

from IPython.core.getipython import get_ipython
from IPython.display import display, Markdown, display_markdown
from pathlib import Path

import sys

__all__ = ['display_mds', 'display_dfs', 'nb_setup']


def display_mds(*strings):
    """
    Utility function to display several strings formatted in markdown format

    :param strings: any number of strings with text in markdown format
    :return: None
    """
    for string in strings:
        display_markdown(Markdown(data=string))


def display_dfs(*dfs):
    """
    Utility function to display several DataFrame with one operations
    :param dfs: any number of Pandas DataFrames
    :return: None
    """
    for df in dfs:
        display(df)


def nb_setup(autoreload=True, paths=None):
    """Use in first cell of nb for setting up autoreload, paths, ...

    By default, nb_setup() loads and set autoreload and adds a path to a directory named 'src'
    at the same level as where the notebook directory is located.

    By default, nb_setup assumes the following file structure:

    project_directory
          | --- notebooks
          |        | --- current_nb.ipynb
          |        | --- ...
          |
          |--- src
          |     | --- scripts_to_import.py
          |     | --- ...
          |
          |--- data
          |     |
          |     | ...
    """
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


if __name__ == '__main__':

    pass
