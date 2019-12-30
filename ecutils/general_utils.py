import configparser
import logging
import os
import pkg_resources
import time

from datetime import datetime
from pathlib import Path


def pprint_dict(obj=None, topic=None, depth=None):
    """
    Prints a pretty clean tree representation of a dictionary
    Recursive function

    obj:    Object to be printed. If object is not a dictionary, it will simply be printed as 'print(obj)'
    topic:  (Optional) used in the deeper levels of the recursion to pass information from one level to another one
    i:      (Optional) used in the deeper levels of the recursion to pass the depths
    """
    if obj is None: obj = 'No object provided for printing. Cannot print anything !!!!'
    if depth is None: depth = 0
    sep = ' '
    mark = '-'
    factor = 2
    if isinstance(obj, dict):
        space = depth * factor * sep
        depth += 1
        for key, value in obj.items():
            if isinstance(value, dict): print(f"{space}{sep*factor}{mark}{key}")
            pprint_dict(value, topic=key, depth=depth)
    else:
        space = depth * factor * sep
        print(f"{space}{mark}{'' if topic is None else f'{topic}: '}{obj}")


if __name__ == '__main__':
    from historical_price_handling import ticker_dict
    pprint_dict(ticker_dict)