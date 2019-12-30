"""
This file includes a set of utility function to be used in Jupyter Notebook and Jupyter Lab notebooks

{License_info}
"""

from IPython.display import display, Markdown, display_markdown


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


if __name__ == '__main__':

    pass
