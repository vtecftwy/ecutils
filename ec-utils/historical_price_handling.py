"""
Master Historical Prices Utility Functions

Includes all stable utility functions for handling historical prices.
Link to this scrip by using a Symlink:
 - mklink /D "D:\PyProjects\fx-bt\src\symlink_name" "D:\PyProjects\ec-utils"
 More info here: https://www.maketecheasier.com/create-symbolic-links-windows10/
"""

import configparser
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pkg_resources
import time
import urllib.request

from datetime import datetime
from pathlib import Path
from matplotlib import style
from bokeh.plotting import figure, show, output_file
from IPython.display import display

# --------------------------------------------------------------------------------------------------------------------
# --- Set general parameter for current file
verbose = True
test_mode = True
forced_file_error = False

# --------------------------------------------------------------------------------------------------------------------
# --- Set and load configuration and default values dictionaries
#
# Location of raw and dataset files
NAS_raw_data = Path('R:\\Financial Data\\Historical Prices Raw Data\\alphavantage')
NAS_datasets = Path('R:\\Financial Data\\Historical Prices Datasets\\alphavantage')

sources_dict = {'alphavantage': {'name': 'alphavantage',
                                 'directory': 'alphavantage',
                                 'timeframe': {'M': 'FX_MONTHLY',
                                               'W': 'FX_WEEKLY',
                                               'D': 'FX_DAILY',
                                               'H1': 'FX_INTRADAY_60min',
                                               'M30': 'FX_INTRADAY_30min',
                                               'M15': 'FX_INTRADAY_15min',
                                               'M5': 'FX_INTRADAY_5min',
                                               'M1': 'FX_INTRADAY_1min'},
                                 'format': 'alphavantage'},
                'axitrader': {'name': 'axitrader',
                              'directory': 'axitrader-mt4',
                              'timeframe': {'D': '1440',
                                            'H4': '240',
                                            'H1': '60'},
                              'format': 'mt4'},
                'yahoo': {'name': 'yahoo',
                          'directory': 'yahoo',
                          'format': 'yahoo'},
                'xm-com': {'name': 'xm-com',
                           'directory': 'xm-com-mt4',
                           'format': 'mt4'},
                'metatrader': {'name': 'metatrader',
                               'directory': 'metatrader-mt4',
                               'format': 'mt4'},
                'mixed-mt4': {'name': 'mixed-mt4',
                              'directory': 'mixed-mt4',
                              'format': 'mt4'},
                'wsj': {'name': 'wsj',
                        'directory': 'wsj',
                        'format': 'wsj'},
                }

# Load ticker dictionary from json file (use pkg_resources to point to package directory)
json_file = Path(pkg_resources.resource_filename(__name__, 'historical_prices_tickers.json'))
if json_file.is_file():
    with open(json_file, 'r') as f:
        ticker_dict = json.load(f)
else:
    raise ValueError(f"No json file with name <{json_file.name}> at {json_file.absolute()}")

# Create default lists for alphavantage
source = 'alphavantage'
forex_alphavantage = [subdict['tickers'][source] for sec, subdict in ticker_dict.items()
                      if source in subdict['tickers'].keys() and subdict['type'] == 'forex']

# Create default lists for axitrader-mt4
source = 'axitrader'
forex_axitrader = [subdict['tickers'][source] for sec, subdict in ticker_dict.items()
                   if source in subdict['tickers'].keys() and subdict['type'] == 'forex']
indices_cash_axitrader = [subdict['tickers'][source] for sec, subdict in ticker_dict.items()
                          if source in subdict['tickers'].keys() and subdict['type'] == 'index']
indices_future_axitrader = [subdict['tickers'][source] for sec, subdict in ticker_dict.items()
                            if source in subdict['tickers'].keys() and subdict['type'] == 'index future']
commodities_cash_axitrader = [subdict['tickers'][source] for sec, subdict in ticker_dict.items()
                              if source in subdict['tickers'].keys() and subdict['type'] == 'commodity']
commodities_future_axitrader = [subdict['tickers'][source] for sec, subdict in ticker_dict.items()
                                if source in subdict['tickers'].keys() and subdict['type'] == 'commodity future']
crypto_axitrader = [subdict['tickers'][source] for sec, subdict in ticker_dict.items()
                    if source in subdict['tickers'].keys() and subdict['type'] == 'crypto']

# --------------------------------------------------------------------------------------------------------------------
# --- Load project config files and save in config dictionary (https://docs.python.org/3.4/library/configparser.html)
# --- required keys: 'module_root_directory', 'alphavantage'


def load_config(cwdir):
    """Walk up folder tree till config.cfg file is found, load any *.cfg file and return config dictionary

    :param: Path() object to the the directory where to check for cfg files
    return: config_dict: configuration dictionary key:value dictionary
    """
    # When method is called from 'ec-utils' directly, instead of another package, use config from fx-bt.
    if cwdir.name == 'ec-utils': cwdir = Path('D:\\PyProjects\\fx-bt')

    cwd_is_config_file_dir = 'config.cfg' in list(f.name for f in cwdir.iterdir() if f.is_file())
    if cwd_is_config_file_dir:
        # Read all *.cfg files in the current working directory
        configuration = configparser.ConfigParser()
        for cfg_file in list(f for f in cwdir.iterdir() if '.cfg' in str(f.name)):
            configuration.read(cfg_file)
        config_dict = {}
        for section in configuration:
            for key in configuration[section]:
                config_dict[key] = configuration[section][key]
        return config_dict
    else:
        cwdir = (cwdir / '..').resolve()
        if cwdir.name == '':
            error_string = 'Current working directory (CWD) is not within module root directory' \
                           + f'\n   - CWD: {str(Path.cwd().absolute())}'
            raise ValueError(error_string)

        return load_config(cwdir)


config = load_config(Path().cwd())


# --------------------------------------------------------------------------------------------------------------------
# --- Module utility functions
# ------------------------------


def print_log(text2print=None, verbose=False):
    """
    Legacy function to pass text2print for logging with logging.info()

    Parameters:
    ----------
    text2print :    String to print out
    verbose :       Default: False. Print log only when True

    Return:
    ------
    Does not return anything
    """
    logging.info(text2print)
    # print(text2print)


def get_module_root_path(module_root=None):
    """
    Return a Path object giving the root directory of the module.

    The function starts from the current working directory (CWD) and climbs up the path up to the module root.
    If the CWD is not in the module path, it will go up to the top of the path and fail with error message.

    Parameter:
    ---------
    module_root :   Optional. Default is the module directory as defined in config.cfg file

    Return:
    ------
    current_directory : Path object pointing to the root directory of the folder with name 'module_root'

    """
    if module_root is None:
        module_root = config['module_root_directory']

    current_directory = Path.cwd()
    while True:
        if current_directory.name == module_root:
            return current_directory
        current_directory = current_directory / '..'
        current_directory = current_directory.resolve()
        if current_directory.name == '':
            error_string = 'Current working directory (CWD) is not within module root directory' \
                           + f'\n   - CWD: {str(Path.cwd().absolute())}' \
                           + f'\n   - Module root directory name: {module_root}'
            raise ValueError(error_string)


def remove_multiple_wildcards(s):
    """returns a str where any occurrence of multiple * is replace by a single *."""
    return s.replace('***', '*').replace('**', '*')


def fill_nan(df):
    """Fill DataFrame nan in two passes: forward fill first, then back fill nan at start of column"""
    df_filled = df.fillna(method='ffill', inplace=False)
    df_filled.fillna(method='bfill', inplace=True)
    return df_filled


def display_df(df, mrows=None, show_info=False):
    """Displays a DataFrame with a custom number of rows for review in Jupyter notebook

    Parameters:
    ----------
    df :        (DataFrame) the DataFrame to display
    mrows :     (int)       the maximum number of rows to display in the notebook

    Return:
    ------
    Does not return anything

    """
    default_max_rows = pd.get_option('display.max_rows')
    if show_info:
        display(df.info())  # only used with jupyter where display() is defined
    if mrows is None:
        mrows = default_max_rows
    elif mrows <= 3:
        mrows = 3
    pd.set_option('display.max_rows', mrows)
    display(df)  # only used with jupyter where display() is defined
    pd.set_option('display.max_rows', default_max_rows)


def str_date(d):
    return f'{d.year}-{d.month}-{d.day}'


def safe_sampling(df, first=None, last=None):
    """Returns tuple(first date, last date) in str format, for passed df"""
    dmin, dmax = df.index[0], df.index[-1]
    earliest, latest = f'{dmin.year}-{dmin.month}-{dmin.day}', f'{dmax.year}-{dmax.month}-{dmax.day}'
    if first is None:
        first = earliest
    if last is None:
        last = latest
    sample = df.loc[max(first, earliest):min(last, latest), :].copy()
    return sample


# --------------------------------------------------------------------------------------------------------------------
# ---- Functions to handle historical price files.
# ------------------------------------------------
# ToDo: refactor these functions to only have one function or class and use parameters for different sources

def date_latest_alphavantage_download():
    target_directory = Path('D:\\PyProjects\\fx-bt') / 'data/raw-data/alphavantage'
    list_of_files_names = list(p.name for p in target_directory.glob('EURUSD_FX_DAILY_*'))
    # EURUSD_FX_DAILY_YYYY-MM-DD.csv 0123456789012345 YYYY 0 MM 3 DD 4
    list_of_dates = [datetime(year=int(s[16:20]),
                              month=int(s[21:23]),
                              day=int(s[24:26])) for s in list_of_files_names]
    return max(list_of_dates)


def get_price_dict_from_alphavantage(ticker='*', timeframe='*', timefilter='', target_directory=None, verbose=False):
    """
    Returns Dict w/ all price information (alphavantage) from 'target_directory', filtered by 'ticker' and 'timeframe'.

    Read price data from file for ALL files in 'target_directory' matching passed 'ticker' and 'timeframe'.
    Wildcard * is permitted for both 'ticker' and 'timeframe'.
    Price data follows alphavantage format

    Store prices from each file into a DataFrame
    Clean the data (take out nan)
    Store price DataFrames into a dictionary with key = source price file name

    Parameters:
    -----------
    ticker :            (str) Name of the ticker for which to load prices, or wilcard *
    timeframe :         (str) Name of the timeframe considered: FX_INTRADAY_60min/30min/15min/5min/1min
                              FX_DAILY, FX_WEEKLY, FX_MONTHLY or wildcard *
    timefilter :        (str) pattern to select a subset of the files based on date: '', 'YYYY', 'YYYY-MM'
    target_directory :  (Path) (Optional) path to the directory where the price files are stored
                        Default is 'data/raw-data/alphavantage'
    verbose :           (boolean) Set True to print more log comment. Default is False

    Return:
    ------
    price_dict :        (dict) dictionary with all uploaded price DataFrame
    """
    if target_directory is None:
        target_directory = get_module_root_path() / 'data/raw-data/alphavantage'

    selection_pattern = remove_multiple_wildcards(f'{ticker}_{timeframe}_{timefilter}*.csv')
    target_files_list = list(target_directory.glob(selection_pattern))
    price_dict = {}
    for file in target_files_list:
        file_absolute_path_str = str(file.absolute())
        if verbose:
            print(file_absolute_path_str)
        prices_df = pd.read_csv(file_absolute_path_str,
                                sep=',',
                                index_col=0,
                                header=None,
                                skiprows=1,
                                parse_dates={'Bar': [0]},
                                infer_datetime_format=True,
                                names=['timestamp', 'Open', 'High', 'Low', 'Close'],
                                na_values=['nan', 'null', 'NULL']
                                )
        prices_df = fill_nan(prices_df)
        price_dict[str(file.name)] = prices_df.sort_index(axis='rows', ascending=True)
    return price_dict


def get_price_dict_from_mt4(ticker='*', timeframe='*', target_directory=None, verbose=False):
    """
    Get price data (MT4 export files) from folder into a Dict of DataFrame. Return the Dict.
    
    Read price data from MT4 format file for ALL the files in 'price files' directory matching 
    passed 'ticker' and 'timeframe' (wildcard * is permitted for both 'ticker' and 'timeframe').
    Store price data into a DataFrame for each selected file
    Clean the data (take out nan)
    Store price DataFrame into a dictionary with key = source price file name

    Parameters:
    -----------
    ticker :            (str) Name of the ticker for which to load prices, or wilcard *
    timeframe :         (str) Name of the timeframe considered: 60, 240, 1440 or wildcard *
    timefilter :        (str) pattern to select a subset of the files based on date: '', 'YYYY', 'YYYY-MM'
    target_directory :  (Path) (Optional) path to the directory where the price files are stored
                        Default is 'data/raw-data/data/metatrader-mt4'
    verbose :           (boolean) Set True to print more log comment. Default is False

    Return:
    ------
    price_dict :        (dict) dictionary with all uploaded price DataFrame
    :return: price_dict: (dict) dictionary with all uploaded price DataFrame
    """
    if target_directory is None:
        target_directory = get_module_root_path() / 'data' / 'raw-data' / 'axitrader-mt4'

    selection_pattern = remove_multiple_wildcards('*' + ticker + timeframe + '*.csv')

    target_files_list = list(target_directory.glob(selection_pattern))
    price_dict = {}
    for file in target_files_list:
        file_absolute_path_str = str(file.absolute())
        if verbose:
            print(file_absolute_path_str)
        prices_df = pd.read_csv(file_absolute_path_str,
                                sep=',',
                                header=None,
                                skiprows=1,
                                index_col=0,
                                parse_dates={'Bar': [0, 1]},
                                infer_datetime_format=True,
                                names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'],
                                na_values=['nan', 'null', 'NULL']
                                )
        prices_df = fill_nan(prices_df)
        price_dict[str(file.name)] = prices_df
    return price_dict


def get_price_dict_from_wsj(ticker='*', timeframe='*', target_directory=None, verbose=False):
    """
    Get price data (wsj download) from folder into a Dict of DataFrame. Return the Dict.

    Read price data from yahoo format file for ALL the files in 'price files' directory matching
    passed 'ticker' and 'timeframe' (wildcard * is permitted for both 'ticker' and 'timeframe').
    Store price data into a DataFrame for each selected file
    Clean the data (take out nan)
    Store price DataFrame into a dictionay with key = source price file name

    Parameters:
    -----------
    ticker :            (str) Name of the ticker for which to load prices, or wilcard *
    timeframe :         (str) Name of the timeframe considered: 60, 240, 1440, w, m or wildcard *
    target_directory :  (Path) (Optional) path to the directory where the price files are stored
                        Default is 'data/raw-data/alphavantage'
    verbose :           (boolean) Set True to print more log comment. Default is False

    Return:
    ------
    price_dict :        (dict) dictionary with all uploaded price DataFrame
    """
    if target_directory is None:
        target_directory = get_module_root_path() / 'data' / 'raw-data' / 'wsj'

    selection_pattern = remove_multiple_wildcards(ticker + timeframe + '*.csv')
    target_files_list = list(target_directory.glob(selection_pattern))
    price_dict = {}
    for file in target_files_list:
        file_absolute_path_str = str(file.absolute())
        print_log(file_absolute_path_str, verbose)
        prices_df = pd.read_csv(file_absolute_path_str,
                                sep=',',
                                header=None,
                                skiprows=1,
                                index_col=0,
                                parse_dates={'Bar': [0]},
                                infer_datetime_format=True,
                                # usecols=['Date', 'Open', 'High', 'Low', 'Close'],
                                names=['Date', 'Open', 'High', 'Low', 'Close'],
                                na_values=['nan', 'null', 'NULL']
                                )
        prices_df = prices_df.sort_index()
        prices_df = fill_nan(prices_df)
        price_dict[str(file.name)] = prices_df
        # price_dict[str(file.name)] = prices_df
    return price_dict


def get_price_dict_from_yahoo(ticker='*', timeframe='*', target_directory=None, verbose=False):
    """
    Get price data (yahoo download) from folder into a Dict of DataFrame. Return the Dict.
    
    Read price data from yahoo format file for ALL the files in 'price files' directory matching 
    passed 'ticker' and 'timeframe' (wildcard * is permitted for both 'ticker' and 'timeframe').
    Store price data into a DataFrame for each selected file
    Clean the data (take out nan)
    Store price DataFrame into a dictionay with key = source price file name

    Parameters:
    -----------
    ticker :            (str) Name of the ticker for which to load prices, or wilcard *
    timeframe :         (str) Name of the timeframe considered: 60, 240, 1440, w, m or wildcard *
    target_directory :  (Path) (Optional) path to the directory where the price files are stored
                        Default is 'data/raw-data/alphavantage'
    verbose :           (boolean) Set True to print more log comment. Default is False

    Return:
    ------
    price_dict :        (dict) dictionary with all uploaded price DataFrame
    """
    if target_directory is None:
        target_directory = get_module_root_path() / 'data' / 'raw-data' / 'yahoo'

    selection_pattern = remove_multiple_wildcards(ticker + timeframe + '*.csv')
    target_files_list = list(target_directory.glob(selection_pattern))
    price_dict = {}
    for file in target_files_list:
        file_absolute_path_str = str(file.absolute())
        print_log(file_absolute_path_str, verbose)
        prices_df = pd.read_csv(file_absolute_path_str,
                                sep=',',
                                header=0,
                                index_col=0,
                                parse_dates={'Bar': [0]},
                                infer_datetime_format=True,
                                usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'],
                                # names=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'],
                                na_values=['nan', 'null', 'NULL']
                                )
        prices_df = fill_nan(prices_df)
        price_dict[str(file.name)] = prices_df
        price_dict[str(file.name)] = prices_df
    return price_dict


def compare_df(df_to_compare, display_diff=False, display_info=True):
    """compare two DataFrames passed as a tuple. returns True is no conflict, False otherwise"""

    start_bar1, end_bar_1 = df_to_compare[0].index[0], df_to_compare[0].index[-1]
    start_bar2, end_bar_2 = df_to_compare[1].index[0], df_to_compare[1].index[-1]
    bar_range_to_compare = (max(start_bar1, start_bar2), min(end_bar_1, end_bar_2))

    if bar_range_to_compare[0] >= bar_range_to_compare[1]:
        if display_info:
            print(f'No common range between both files: ')
        return True

    sdf1, sdf2 = df_to_compare
    sdf1 = sdf1[bar_range_to_compare[0]:bar_range_to_compare[1]]
    sdf2 = sdf2[bar_range_to_compare[0]:bar_range_to_compare[1]]
    diff = (sdf1 - sdf2).loc[(sdf1 - sdf2).apply(
        lambda x: True if any(x[col] > 0.0001 for col in ['Open', 'High', 'Low', 'Close']) else False,
        axis='columns')]

    if diff.shape[0] == 0:
        no_bar_conflict = True
        if display_info:
            print(f'{sdf1.shape[0]} common bars are all identical !')
    else:
        no_bar_conflict = False
        if display_info:
            print(f'++++++++++++++++++++++++++\n{diff.shape[0]} bars with different prices')
            print(f'  First different: {diff.index[0]}.')
            print(f'  Last different: {diff.index[-1]}.')
            print(f'{sdf1.shape[0]} common bars')
            print(f'Common range between both files: ')
            print(f' from {bar_range_to_compare[0]}')
            print(f' to   {bar_range_to_compare[1]}\n')
            print(sdf1.loc[diff.index, :])
            print(sdf2.loc[diff.index, :])
    if display_diff:
        display(diff)  # only used with jupyter where display() is defined

    return no_bar_conflict


def compare_prices_in_files_mt4(ticker='EURUSD', timeframe='1440', target_directory=None, type='mt4', verbose=False):
    """ """

    price_dict = get_price_dict_from_mt4(ticker=ticker,
                                         timeframe=timeframe,
                                         target_directory=target_directory)
    list_of_files = list(price_dict.keys())

    print(f'{len(list_of_files)} files for {ticker} {timeframe}:\n')

    for i, file_name in enumerate(list_of_files):
        print(f'  {i:02d} | {file_name} | {price_dict[file_name].shape[0]} rows')
        df_index = price_dict[file_name].index
        print(f'       First Bar: {df_index[0]}. Weekday: {df_index[0].isoweekday()}.')
        print(f'       Last Bar : {df_index[-1]}. Weekday: {df_index[-1].isoweekday()}.')

    for base_file in range(0, len(list_of_files)):
        for i in list(x for x in range(0, len(list_of_files)) if x != base_file):
            print('<------------------------------------------------------------------------------------>')
            print(f'Comparing: {base_file}:{list_of_files[base_file]} and\n           {i}:{list_of_files[i]}')
            files_to_compare = (base_file, i)

            df_to_compare = (price_dict[list_of_files[files_to_compare[0]]],
                             price_dict[list_of_files[files_to_compare[1]]])
            diff = compare_df(df_to_compare)


def consolidate_price_files_mt4(ticker='EURUSD', timeframe='1440',
                                data_source=None, drive=None,
                                type='mt4', verbose=False):
    """ """

    if data_source is None:
        data_source = 'axitrader'
    if drive is None:
        drive = 'project'

    if drive == 'NAS':
        prices_source_dir = Path('R:\\Financial Data\\Historical Prices Raw Data') / sources_dict[data_source][
            'directory']
    elif drive == 'project':
        prices_source_dir = get_module_root_path() / 'data' / 'raw-data' / sources_dict[data_source]['directory']
    else:
        msg = f'value for drive: {drive} is not recognized'
        raise ValueError(msg)

    price_dict = get_price_dict_from_mt4(ticker=ticker, timeframe=timeframe, target_directory=prices_source_dir)

    list_of_files = list(price_dict.keys())

    all_prices = pd.DataFrame(columns=price_dict[list_of_files[0]].columns)
    for i, file_name in enumerate(list_of_files):
        all_prices = pd.concat([all_prices, price_dict[file_name]],
                               axis='index', join='outer').drop_duplicates(keep='first')
    all_prices = all_prices.sort_index()

    sample = all_prices
    sample_index = sample.index

    print(f'Ticker: {ticker} for {timeframe}')
    print(f'Number of days in sample: {len(sample_index)}')
    print(f'First Period: {sample_index[0]}. Weekday: {sample_index[0].isoweekday()}. Close: {sample.iloc[0, 3]}')
    print(f'Last Period : {sample_index[-1]}. Weekday: {sample_index[-1].isoweekday()}. Close: {sample.iloc[-1, 3]}')
    print('\n')
    print('>>> Rows including N/A value: >>>>>>>>>>>')
    display(sample[sample.isna().any(axis='columns')])
    print('\n')

    return all_prices


def update_price_datasets_alphavantage(ticker_list=None, timeframe_list=None, timefilter=None,
                                       data_source=None, drive=None, type='alphavantage', verbose=False):
    if timeframe_list is None:
        timeframe_list = ['FX_DAILY', 'FX_INTRADAY_60min', 'FX_INTRADAY_30min']
    if ticker_list is None:
        ticker_list = forex_alphavantage
    if timefilter is None:
        timefilter = ''
    if data_source is None:
        data_source = 'alphavantage'

    if drive is None:
        drive = 'project'
    if drive == 'NAS':
        prices_source_dir = Path('R:\\Financial Data\\Historical Prices Raw Data') / sources_dict[data_source][
            'directory']
        datasets_dir = Path('R:\Financial Data\Historical Prices Datasets') / sources_dict[data_source]['directory']
    elif drive == 'project':
        project_root = get_module_root_path()
        prices_source_dir = project_root / 'data' / 'raw-data' / sources_dict[data_source]['directory']
        datasets_dir = project_root / 'data' / 'datasets' / sources_dict[data_source]['directory']
    else:
        msg = f'value for drive: {drive} is not recognized'
        raise ValueError(msg)

    def exclude_idx_in_dataset(dataset, df):
        ds_idx = dataset.index
        df_idx = df.index
        filtered_idx = list(set(df_idx).difference(set(ds_idx)))
        return df.loc[filtered_idx, :]

    for ticker in ticker_list:
        for timeframe in timeframe_list:
            print_log(f'Handling {ticker} for {timeframe}')

            dataset_file_name = f'{data_source}-{ticker}-{timeframe}.csv'
            path2dataset = datasets_dir / dataset_file_name

            alphadict = get_price_dict_from_alphavantage(ticker=ticker, timeframe=timeframe, timefilter=timefilter,
                                                         target_directory=prices_source_dir)
            list_of_files = list(alphadict)

            if path2dataset.exists():
                print_log(f' - Loading existing dataset from {dataset_file_name}')
                alpha_full_ds = pd.read_csv(path2dataset,
                                            sep=',',
                                            index_col=0,
                                            header=None,
                                            skiprows=1,
                                            parse_dates=[0],
                                            infer_datetime_format=True,
                                            names=['Date', 'Open', 'High', 'Low', 'Close'],
                                            na_values=['nan', 'null', 'NULL']
                                            )
            else:
                print_log(f' - Creating a new dataset for {dataset_file_name}')
                alpha_full_ds = pd.DataFrame(columns=alphadict[list_of_files[0]].columns)

            for file_name, df in alphadict.items():
                filtered_df = exclude_idx_in_dataset(alpha_full_ds, df.iloc[:-1, :])
                print_log(f' - Updating with {file_name}')
                print_log(
                    f'   Dataset: {alpha_full_ds.shape[0]}. File: {df.shape[0]}. Added {filtered_df.shape[0]} rows.')
                alpha_full_ds = alpha_full_ds.append(filtered_df, verify_integrity=True, sort=False)
                alpha_full_ds.sort_index(ascending=True, inplace=True)

            assert all(alpha_full_ds.isna()) is True, f'Some values in the dataset (alpha_full_ds) are NaN.'

            print_log(f' - Saving updated dataset into {dataset_file_name}')
            alpha_full_ds.to_csv(path_or_buf=datasets_dir / dataset_file_name, index_label='timestamp')


def update_price_datasets_mt4(ticker_list=None, timeframe_list=None,
                              data_source=None, drive=None, type='mt4', verbose=False):
    PROCESSED_FILE_FLAG = '_processed_into_ds'

    # Define standard lists for MT4
    tickers_forex = ['AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDUSD', 'CADCHF', 'CADJPY', 'CHFJPY', 'EURAUD', 'EURCHF',
                     'EURGBP', 'EURJPY', 'EURSGD', 'EURUSD', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPUSD',
                     'NZDUSD', 'SGDJPY', 'USDCAD', 'USDCHF', 'USDCNH', 'USDHKD', 'USDILS', 'USDJPY', 'USDPLN',
                     'USDSGD', 'USDTHB']

    tickers_indices_cash = ['CN50', 'EU50', 'FRA40', 'GER30', 'UK100', 'US30', 'US500', 'US2000', 'USTECH']
    tickers_indices_future = ['CAC40.fs', 'CHINA50.fs', 'DAX30.fs', 'DJ30.fs', 'EUSTX50.fs', 'FT100.fs',
                              'NAS100.fs', 'S&P.fs']

    tickers_commodities_cash = ['UKOIL', 'USOIL', 'XAGUSD', 'XAUUSD']
    tickers_commodities_future = ['BRENT.fs', 'COCOA.fs', 'COFFEE.fs', 'COPPER.fs', 'GOLD.fs', 'NATGAS.fs',
                                  'SILVER.fs', 'SOYBEAN.fs', 'WTI.fs']

    tickers_others = ['BTCUSD']

    tickers_non_forex = tickers_indices_cash + tickers_indices_future + \
                        tickers_commodities_cash + tickers_commodities_future + \
                        tickers_others

    if timeframe_list is None:
        timeframe_list = ['60', '240', '1440']
    if ticker_list is None:
        ticker_list = tickers_non_forex
    if data_source is None:
        data_source = 'axitrader'
    if drive is None:
        drive = 'project'
    if drive == 'NAS':
        prices_source_dir = Path('R:\\Financial Data\\Historical Prices Raw Data') / sources_dict[data_source][
            'directory']
        datasets_dir = Path('R:\Financial Data\Historical Prices Datasets') / sources_dict[data_source]['directory']
    elif drive == 'project':
        project_root = get_module_root_path()
        prices_source_dir = project_root / 'data' / 'raw-data' / sources_dict[data_source]['directory']
        datasets_dir = project_root / 'data' / 'datasets' / sources_dict[data_source]['directory']
    else:
        msg = f'value for drive: {drive} is not recognized'
        raise ValueError(msg)

    def exclude_idx_in_dataset(dataset, df):
        # ToDo: refactor this by bringing this function to higher level in both this fct and alphavantage function
        ds_idx = dataset.index
        df_idx = df.index
        filtered_idx = list(set(df_idx).difference(set(ds_idx)))
        return df.loc[filtered_idx, :]

    for ticker in ticker_list:
        for timeframe in timeframe_list:
            print_log(f'Handling {ticker} for {timeframe}')

            dataset_file_name = f'{data_source}-{ticker}-{timeframe}.csv'
            path2dataset = datasets_dir / dataset_file_name

            mt4dict = get_price_dict_from_mt4(ticker=ticker, timeframe=timeframe, target_directory=prices_source_dir)
            files_to_process = [f for f in list(mt4dict) if PROCESSED_FILE_FLAG not in f]

            if path2dataset.exists():
                print_log(f' - Loading existing dataset from {dataset_file_name}')
                mt4_full_ds = pd.read_csv(path2dataset,
                                          sep=',',
                                          index_col=0,
                                          header=None,
                                          skiprows=1,
                                          parse_dates=[0],
                                          infer_datetime_format=True,
                                          names=['Bar', 'Open', 'High', 'Low', 'Close', 'Volume'],
                                          na_values=['nan', 'null', 'NULL']
                                          )
            else:
                print_log(f' - Creating a new dataset for {dataset_file_name}')
                mt4_full_ds = pd.DataFrame(columns=mt4dict[files_to_process[0]].columns)

            for file_name in files_to_process:
                df = mt4dict[file_name]
                filtered_df = exclude_idx_in_dataset(mt4_full_ds, df)
                print_log(f' - Updating with {file_name}')
                print_log(
                    f'   Dataset: {mt4_full_ds.shape[0]}. File: {df.shape[0]}. Added {filtered_df.shape[0]} rows.')
                mt4_full_ds = mt4_full_ds.append(filtered_df, verify_integrity=True, sort=False)
                mt4_full_ds.sort_index(ascending=True, inplace=True)

                process_flag = PROCESSED_FILE_FLAG + f'_on_{datetime.today().strftime("%Y-%m-%d")}'
                new_file_name = file_name[0:-4] + process_flag + '.csv'
                os.rename(prices_source_dir / file_name, prices_source_dir / new_file_name)

            assert all(mt4_full_ds.isna()) is True, f'Some values in the dataset (alpha_full_ds) are NaN.'

            print_log(f' - Saving updated dataset into {dataset_file_name}')
            mt4_full_ds.to_csv(path_or_buf=datasets_dir / dataset_file_name, index_label='timestamp')


# ToDo: make function update_price_datasets_wsj()


def update_price_datasets_yahoo(ticker_list=None, timeframe_list=None,
                                data_source=None, drive=None, type='mt4', verbose=False):
    PROCESSED_FILE_FLAG = '_processed_into_ds'

    # Define standard lists for MT4
    tickers_forex = []

    tickers_indices_cash = ['FCHI', 'GDAXI', 'GSPC', 'NDX', 'NYA', 'RUT', 'STOXX50E']
    tickers_indices_future = []

    tickers_commodities_cash = []
    tickers_commodities_future = []

    tickers_others = []

    tickers_non_forex = tickers_indices_cash + tickers_indices_future + \
                        tickers_commodities_cash + tickers_commodities_future + \
                        tickers_others

    if timeframe_list is None:
        timeframe_list = ['1440']
    if ticker_list is None:
        ticker_list = tickers_non_forex
    if data_source is None:
        data_source = 'yahoo'
    if drive is None:
        drive = 'project'
    if drive == 'NAS':
        prices_source_dir = Path('R:\\Financial Data\\Historical Prices Raw Data') / sources_dict[data_source][
            'directory']
        datasets_dir = Path('R:\Financial Data\Historical Prices Datasets') / sources_dict[data_source]['directory']
    elif drive == 'project':
        project_root = get_module_root_path()
        prices_source_dir = project_root / 'data' / 'raw-data' / sources_dict[data_source]['directory']
        datasets_dir = project_root / 'data' / 'datasets' / sources_dict[data_source]['directory']
    else:
        msg = f'value for drive: {drive} is not recognized'
        raise ValueError(msg)

    def exclude_idx_in_dataset(dataset, df):
        # ToDo: refactor this by bringing this function to higher level in both this fct and alphavantage function
        ds_idx = dataset.index
        df_idx = df.index
        filtered_idx = list(set(df_idx).difference(set(ds_idx)))
        return df.loc[filtered_idx, :]

    for ticker in ticker_list:
        for timeframe in timeframe_list:
            print_log(f'Handling {ticker} for {timeframe}')

            dataset_file_name = f'{data_source}-{ticker}-{timeframe}.csv'
            path2dataset = datasets_dir / dataset_file_name

            yhodict = get_price_dict_from_yahoo(ticker=ticker, timeframe=timeframe, target_directory=prices_source_dir)
            files_to_process = [f for f in list(yhodict) if PROCESSED_FILE_FLAG not in f]

            if path2dataset.exists():
                print_log(f' - Loading existing dataset from {dataset_file_name}')
                yho_full_ds = pd.read_csv(path2dataset,
                                          sep=',',
                                          index_col=0,
                                          header=None,
                                          skiprows=1,
                                          parse_dates=[0],
                                          infer_datetime_format=True,
                                          names=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'],
                                          na_values=['nan', 'null', 'NULL']
                                          )
            else:
                print_log(f' - Creating a new dataset for {dataset_file_name}')
                yho_full_ds = pd.DataFrame(columns=yhodict[files_to_process[0]].columns)

            for file_name in files_to_process:
                df = yhodict[file_name]
                filtered_df = exclude_idx_in_dataset(yho_full_ds, df)
                print_log(f' - Updating with {file_name}')
                print_log(
                    f'   Dataset: {yho_full_ds.shape[0]}. File: {df.shape[0]}. Added {filtered_df.shape[0]} rows.')
                yho_full_ds = yho_full_ds.append(filtered_df, verify_integrity=True, sort=False)
                yho_full_ds.sort_index(ascending=True, inplace=True)

                process_flag = PROCESSED_FILE_FLAG + f'_on_{datetime.today().strftime("%Y-%m-%d")}'
                new_file_name = file_name[0:-4] + process_flag + '.csv'
                os.rename(prices_source_dir / file_name, prices_source_dir / new_file_name)

            assert all(yho_full_ds.isna()) is True, f'Some values in the dataset (alpha_full_ds) are NaN.'

            print_log(f' - Saving updated dataset into {dataset_file_name}')
            yho_full_ds.to_csv(path_or_buf=datasets_dir / dataset_file_name, index_label='timestamp')


def get_dataset(ticker=None, timeframe=None, data_source=None, drive=None, type=None, verbose=False):
    """

    """
    # Define standard lists for MT4
    # ToDo: update this function by using above defined default lists and make it multi source
    tickers_forex = ['AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDUSD', 'CADCHF', 'CADJPY', 'CHFJPY', 'EURAUD', 'EURCHF',
                     'EURGBP', 'EURJPY', 'EURSGD', 'EURUSD', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPUSD',
                     'NZDUSD', 'SGDJPY', 'USDCAD', 'USDCHF', 'USDCNH', 'USDHKD', 'USDILS', 'USDJPY', 'USDPLN',
                     'USDSGD', 'USDTHB']

    tickers_indices_cash = ['CN50', 'EU50', 'FRA40', 'GER30', 'UK100', 'US30', 'US500', 'USTECH']
    tickers_indices_future = ['CAC40.fs', 'CHINA50.fs', 'DAX30.fs', 'DJ30.fs', 'EUSTX50.fs', 'FT100.fs',
                              'NAS100.fs', 'S&P.fs']
    tickers_commodities_cash = ['UKOIL', 'USOIL', 'XAGUSD', 'XAUUSD']
    tickers_commodities_future = ['BRENT.fs', 'COCOA.fs', 'COFFEE.fs', 'COPPER.fs', 'GOLD.fs', 'NATGAS.fs',
                                  'SILVER.fs', 'SOYBEAN.fs', 'WTI.fs']
    tickers_others = ['BTCUSD']
    tickers_active = tickers_indices_cash + tickers_indices_future + \
                     tickers_commodities_cash + tickers_commodities_future + \
                     tickers_others

    if timeframe is None:
        timeframe = '1440'
    if ticker is None:
        ticker = 'US500'
    if data_source is None:
        data_source = 'axitrader'
    if drive is None:
        drive = 'project'
    if drive == 'NAS':
        prices_source_dir = Path('R:\\Financial Data\\Historical Prices Raw Data') / sources_dict[data_source]['directory']
        datasets_dir = Path('R:\Financial Data\Historical Prices Datasets') / sources_dict[data_source]['directory']
    elif drive == 'project':
        project_root = get_module_root_path()
        prices_source_dir = project_root / 'data' / 'raw-data' / sources_dict[data_source]['directory']
        datasets_dir = project_root / 'data' / 'datasets' / sources_dict[data_source]['directory']
    else:
        msg = f'value for drive: {drive} is not recognized'
        raise ValueError(msg)

    dataset_file_name = f'{data_source}-{ticker}-{timeframe}.csv'
    path2dataset = datasets_dir / dataset_file_name

    if path2dataset.exists():
        print_log(f'Loading Dataset for {ticker} for {timeframe}',verbose=verbose)
        ds = pd.read_csv(path2dataset,
                         sep=',',
                         index_col=0,
                         header=None,
                         skiprows=1,
                         parse_dates=[0],
                         infer_datetime_format=True,
                         names=['Bar', 'Open', 'High', 'Low', 'Close', 'Volume'],
                         na_values=['nan', 'null', 'NULL']
                          )
        return ds
    else:
        print_log(f'No dataset file at {path2dataset.absolute()} !!!!!!!!!!!')
        return False

# --------------------------------------------------------------------------------------------------------------------
# ---- Functions to download historical prices
# --------------------------------------------


def load_file_from_url(target_url, filename, verbose=False):
    """
    Download file from passed 'target_url' using request.urlopen method. Save file as filename

    :param      (str) target_url: format is http://... or https://....
    :param      (Path) filename:
    :param      (bool) verbose: True for more comments. Default is None
    :return:    True if operation did not raise any issue. False otherwise.
    """

    try:
        response = urllib.request.urlopen(target_url)
        # https://docs.python.org/3.5/library/urllib.request.html
        # also could use http.client library: conn = http.client.HTTPConnection() and conn.request('GET', '/')
        # https://docs.python.org/3.5/library/http.client.html#http.client.HTTPResponse

    except urllib.error.URLError as err:
        if hasattr(err, 'code'):
            print_log(f'Error: {err.code}. Failed URL request for {target_url}.', verbose=verbose)
        if hasattr(err, 'reason'):
            print_log(f'Error: {err.reason}. Failed URL request for {target_url}.', verbose=verbose)
        print('URLError!!!')
        return False
    except urllib.error.HTTPError as err:
        if hasattr(err, 'code'):
            print_log(f'Error: {err.code}. Failed URL request for {target_url}.', verbose=verbose)
        if hasattr(err, 'reason'):
            print_log(f'Error: {err.reason}. Failed URL request for {target_url}.', verbose=verbose)
        print('HTTPError!!!')
        return False

    print_log(f'  - Server HTTP response code: {response.code}', verbose=verbose)
    downloaded_file = response.read()
    print_log(f'  - Done downloading data from {target_url}', verbose=verbose)
    with open(filename, 'wb') as f:
        f.write(downloaded_file)
        # print_log(f'  - Done writing data into file {filename}', verbose=verbose)
    return True


def file_contains_error_message(filename, test_type='alphavantage', forced_error=False):
    """
    Check content of file to identify error messages.

    E.g. "Error Message": "Invalid API call. Please retry or visit the documentation
    (https://www.alphavantage.co/documentation/) for FX_Daily"

    :param      filename: name of the file to test
    :param      test_type: name of API to test for
    :param      forced_error: force an error message for debugging

    :return:    True is no error, False is error
    """

    with open(filename, 'rt') as f:
        error_1 = '"Error Message"'
        error_2 = 'Our standard API call frequency'
        contains_error_1 = any(error_1 in line for line in f.readlines())
        contains_error_2 = any(error_2 in line for line in f.readlines())
        file_contains_error = any([contains_error_1, contains_error_2, forced_error])

    return file_contains_error


def throttle_api(delay=20):
    print_log(f'  - Throttle API for {delay} sec.', verbose=verbose)
    time.sleep(delay)
    print_log(f'  - Awaking from {delay} sec throttle phase\n', verbose=verbose)


def update_alphavantage_fx(alphavantage_mode='compact', pairs=None, timeframes=None, verbose=False):
    """
    Get instrument pricing data from alphavantage for default pairs and tf or specified ones.

    :param alphavantage_mode:       str 'full' or 'compact'
                                    default = 'compact
    :param pairs: (optional)        list: ['EURUSD', 'USDJPY'].
                                    default is full list of pairs
    :param timeframes: (optional)   list: Subset of ['1m', '1w', '1d', '60min', '30min', '15min', '5min', '1min']
                                    default is ['1d', '60min', '30min']
    :param verbose: (optional)      boolean: True to log info in console.

    :return:                        None
    """

    print_log(f'Starting alphavantage price update', verbose=verbose)
    if pairs is None:
        pairs = forex_alphavantage
        pairs.sort()
    if timeframes is None:
        timeframes = ['1d', '60min', '30min']

    SECONDS_TO_SLEEP = 20  # Std API call frequency is 5 calls per minute and 500 calls per day
    number_pairs = len(pairs)
    number_timeframes = len(timeframes)
    # ToDo: align this timeframe_dict with the info on source_dict above !
    # timeframe_dict = {'M1': {'name': 'FX_INTRADAY_1min', 'param': 'FX_INTRADAY&interval=1min', 'mode': 'full'},
    #                   'M2': {'name': 'FX_INTRADAY_5min', 'param': 'FX_INTRADAY&interval=5min', 'mode': 'full'},
    #                   'M15': {'name': 'FX_INTRADAY_15min', 'param': 'FX_INTRADAY&interval=15min', 'mode': 'full'},
    #                   'M30': {'name': 'FX_INTRADAY_30min', 'param': 'FX_INTRADAY&interval=30min', 'mode': 'full'},
    #                   'H1': {'name': 'FX_INTRADAY_60min', 'param': 'FX_INTRADAY&interval=60min', 'mode': 'full'},
    #                   'D': {'name': 'FX_DAILY', 'param': 'FX_DAILY', 'mode': 'compact'},
    #                   'W': {'name': 'FX_WEEKLY', 'param': 'FX_WEEKLY', 'mode': 'compact'},
    #                   'M': {'name': 'FX_MONTHLY', 'param': 'FX_MONTHLY', 'mode': 'compact'},
    #                   }

    timeframe_dict = {'1min': {'name': 'FX_INTRADAY_1min', 'param': 'FX_INTRADAY&interval=1min', 'mode': 'full'},
                      '5min': {'name': 'FX_INTRADAY_5min', 'param': 'FX_INTRADAY&interval=5min', 'mode': 'full'},
                      '15min': {'name': 'FX_INTRADAY_15min', 'param': 'FX_INTRADAY&interval=15min', 'mode': 'full'},
                      '30min': {'name': 'FX_INTRADAY_30min', 'param': 'FX_INTRADAY&interval=30min', 'mode': 'full'},
                      '60min': {'name': 'FX_INTRADAY_60min', 'param': 'FX_INTRADAY&interval=60min', 'mode': 'full'},
                      '1d': {'name': 'FX_DAILY', 'param': 'FX_DAILY', 'mode': 'compact'},
                      '1w': {'name': 'FX_WEEKLY', 'param': 'FX_WEEKLY', 'mode': 'compact'},
                      '1m': {'name': 'FX_MONTHLY', 'param': 'FX_MONTHLY', 'mode': 'compact'},
                      }

    print_log(f'Get new data for selected pairs ({number_pairs}) and timeframes ({number_timeframes}):',
              verbose=verbose)
    print_log(f'  - {timeframes}', verbose=verbose)
    print_log(f'  - {pairs}', verbose=verbose)

    downloads_to_retry_dict = {}
    failed_download_dict = {}

    for i, tf in enumerate(timeframes, 1):

        now = datetime.today()
        now_string = str(now.year) + '-' + str(now.month).zfill(2) + '-' + str(now.day).zfill(2)
        tf_name = timeframe_dict[tf]['name']
        alphavantage_folder = Path('D:\\PyProjects\\fx-bt\\data\\raw-data\\alphavantage')
        alphavantage_tf_parameter = timeframe_dict[tf]['param']
        alphavantage_api_key = config['alphavantage']
        if alphavantage_mode == 'full':
            mode = alphavantage_mode
        else:
            mode = timeframe_dict[tf]['mode']
            # Overwrites the mode from compact to full for shortest time frame (see timeframe_dict above)

        string_to_print = f"\nDownload all pairs for time frame {tf} ( tf {i} of {number_timeframes}) (mode={mode})"
        print_log(string_to_print, verbose=verbose)

        for counter, pair in enumerate(pairs, 1):
            string_to_print = f"\n  Starting process for {pair}-{tf} (pair {counter} of {number_pairs} for tf{i}/{number_timeframes}"
            print_log(string_to_print, verbose=verbose)

            target_currency = pair[0:3]
            base_currency = pair[3:]
            filename = f'{target_currency}{base_currency}_{tf_name}_{now_string}.csv'
            filepath = alphavantage_folder / filename
            url = 'https://www.alphavantage.co/query?function=' + alphavantage_tf_parameter \
                  + '&from_symbol=' + target_currency \
                  + '&to_symbol=' + base_currency \
                  + '&outputsize=' + mode \
                  + '&apikey=' + alphavantage_api_key \
                  + '&datatype=csv'
            name_file_to_download = str(filepath.absolute())

            load_from_url_succeeded = load_file_from_url(target_url=url,
                                                         filename=str(name_file_to_download),
                                                         verbose=verbose)

            file_downloaded_correctly = not file_contains_error_message(filename=name_file_to_download,
                                                                        forced_error=forced_file_error)

            if load_from_url_succeeded is True:
                print_log(f'  - Done writing data into {filename}.', verbose=verbose)
                if file_downloaded_correctly is False:
                    downloads_to_retry_dict[name_file_to_download] = {'url': url,
                                                                      'pair': pair,
                                                                      'tf': tf,
                                                                      'mode': mode,
                                                                      'count': 0,
                                                                      }
                    print_log(f'  - !!!! File is not correct. Pair added to retry list.', verbose=verbose)

            else:
                print_log(f'  - Failed GET to {url}.', verbose=verbose)

            throttle_api(SECONDS_TO_SLEEP)

    # One automatic retry for failed downloads. If fails again, give instructions for manual retry
    if downloads_to_retry_dict:
        print_log(f'Starting retry for failed pair downloads', verbose=verbose)

        instruction_list = []
        for filename, values in downloads_to_retry_dict.items():
            string_to_print = f'\n  Retry for {values["pair"]}'
            print_log(string_to_print, verbose=verbose)

            load_from_url_succeeded = load_file_from_url(target_url=values['url'],
                                                         filename=filename,
                                                         verbose=verbose)

            file_downloaded_correctly = not file_contains_error_message(filename=filename,
                                                                        forced_error=forced_file_error)

            if (not load_from_url_succeeded) or (not file_downloaded_correctly):
                failed_download_dict[filename] = {'pair': values['pair'],
                                                  'tf': values['tf'],
                                                  'mode': values['mode'],
                                                  }
                print_log(f'  - !!!! File is incorrect a second time.', verbose=True)
                print_log(f'  - !!!! To retry manually, instructions are at the end of this log', verbose=True)

                fctn = 'update_alphavantage_fx'
                parameters_1 = f'pairs=["{values["pair"]}"], timeframes=["{values["tf"]}"], '
                parameters_2 = f'alphavantage_mode="{values["mode"]}", verbose=True'
                instruction = fctn + '(' + parameters_1 + parameters_2 + ')'
                instruction_list = instruction_list + [instruction]

            throttle_api(SECONDS_TO_SLEEP)

        print_log('Automatic retry completed', verbose=verbose)

        if instruction_list:
            print('\nInstructions to manually retry failed downloads:')
            print('------------------- Instruction Section Start -----------------------------')
            for instruction in instruction_list:
                print(instruction)
            print('--------------------- Instruction Section End -----------------------------')

    print_log(f'\nCompleted alphavantage price update\n', verbose=verbose)

    return None


# def update_alphavantage_stocks(alphavantage_mode='compact', ticker=None, timeframes=None, verbose=False)


# --------------------------------------------------------------------------------------------------------------------
# --- OHLC handling functions
# ---------------------------


def resample_ohlcv(df, rule_str='W-FRI'):
    """
    Resample an DataFrame with OHLCV format according to given rule string.

    The resampling is applied to each of the OHLC and optional V column,
    Resampling aggregate applies first(), max(), min(), last() and sum() to OHLCV respectively.
    Common 'rules' are: 'D', 'B', 'W', 'W-FRI', 'M'

    Parameters:
    ----------
    df :        dataframe with datetime index and columns 'Open', 'High', 'Low', 'Close'. Optional 'Volume'
    rule_str:   str. Time offset alias for resampling. Default value: 'W-FRI'.

    Return:
    -------
    ohlcv_df : dataframe with datetime index and columns 'Open', 'High', 'Low', 'Close'. Optional 'Volume'
    """

    resampled_df = df.resample(rule_str)
    resampled_index = resampled_df.groups
    # resample() returns a Resampler object, on which a function must be applied, e.g. first(), max(), aggregate()

    ohlcv_df = pd.DataFrame({'Open': resampled_df['Open'].first(),
                             'High': resampled_df['High'].max(),
                             'Low': resampled_df['Low'].min(),
                             'Close': resampled_df['Close'].last(),
                             },
                            index=resampled_index
                            )
    if 'Volume' in resampled_df.count().columns:
        ohlcv_df['Volumes'] = resampled_df['Volume'].sum()

    return ohlcv_df


def autocorrelation_ohlv(series, max_lag, ohlc='Close', **kwarg):
    """
    Return autocorrelation for the passed series, applied on O, H, L or O.

    :param:     series (pd.Series): Series on which autocorrelation has to be applied
    :param:     ohlc (str): Which price to use. Options: 'Open', 'High', 'Low', 'Close' (default)
    :param:     max_lag (int): Maximum lag to consider for the autocorrelation
    :return:    series_autocorrelatioon
    """

    lag_range = range(1, max_lag)
    series_autocorrelation = pd.Series(0, index=lag_range, name='Autocorrelation')
    for lag in lag_range:
        series_autocorrelation.loc[lag] = series[ohlc].autocorr(lag)

    return series_autocorrelation


# --------------------------------------------------------------------------------------------------------------------
# --- Functions to visualize candle stick plots for OHLC.
# -------------------------------------------------------


def candlestick_plot(df, width=950, height=600, chart_title='', fig=None):
    """Create a Bokeh candlestick chart based on the DataFrame provided as parameter.

       Parameters
       ----------
       df: DataFrame with datetime index, and at least the following columns ['Open', 'High', 'Low', 'Close', 'Volume']
       width, height: sizes of the plot figure
       chart_title: optional. String to be used as title of the chart
       fig: optional. Figure to allow superposition of other lines on candlestick plot

       Returns
       ----------
       Nothing

    """
    if fig is None:
        p = figure(plot_width=width, plot_height=height, title=chart_title, x_axis_type="datetime")
    else:
        p = fig
    p.xaxis.major_label_orientation = 3.1415 / 4
    p.grid.grid_line_alpha = 0.5

    # x_axis_type as 'datetime' means that the width dimensions are measured in milliseconds
    # note: sometimes, the two first bars are not contiguous (e.g. bar[0]=Fri and bar[1]=Mon)
    # in such case, if interval(bar[1],bar[0]) > interval(contiguous bars). .
    # instead of taking the first interval, we take the min interval across the full index
    intervals = df.index.unique().to_series() - df.index.unique().to_series().shift(1)
    interval_in_ms = intervals[1:-1].min().total_seconds() * 1000
    ratio = 0.60

    inc = df['Close'] > df['Open']
    dec = df['Close'] < df['Open']
    flat = df['Close'] == df['Open']

    p.segment(df[inc].index, df.loc[inc, 'High'], df[inc].index, df.loc[inc, 'Low'], color='darkgreen')
    p.segment(df[dec].index, df.loc[dec, 'High'], df[dec].index, df.loc[dec, 'Low'], color='darkred')
    p.segment(df[flat].index, df.loc[flat, 'High'], df[flat].index, df.loc[flat, 'Low'], color='black')

    p.vbar(x=df[inc].index,
           bottom=df.loc[inc, 'Open'],
           top=df.loc[inc, 'Close'],
           width=ratio * interval_in_ms,
           fill_color="darkgreen",
           line_color="darkgreen")
    p.vbar(x=df[dec].index,
           bottom=df.loc[dec, 'Close'],
           top=df.loc[dec, 'Open'],
           width=ratio * interval_in_ms,
           fill_color="darkred",
           line_color="darkred")
    p.vbar(x=df[flat].index,
           bottom=df.loc[flat, 'Close'],
           top=df.loc[flat, 'Open'],
           width=ratio * interval_in_ms,
           fill_color="black",
           line_color="black")

    if fig is None:
        show(p)


def multi_plot(time_series_dict, fctn, *args, **kwargs):
    """
    Prepare several subplots based on ..............

    :param time_series_dict: Dict containing all time series to which to apply the fctn
    :param fctn: Function that is applied to time_series_dict to create plot
    :param args: to pass position arguments to the fctn
    :param kwargs: to pass kw argument to both multiplot and plot
                     include: verbose, grid_rows, grid_columns, font_titles, fonmt_axis
    :return None
    """

    # TODO: Review this function and clean up or improve

    # Extract/Set function parameters
    # style.use(str('ggplot'))
    # figsize = par.get('figsize', (6, 3))
    # grid_rows = par.get('grid_rows', 8)
    # grid_columns = par.get('grid_columns', 2)
    # is_hist_cumulative = par.get('is_hist_cumulative', True)
    # hist_style = par.get('hist_style', 'step')  # 'step' 'bar'
    # font_titles = par.get('font_titles', 8)
    # font_axes = par.get('font_axes', 8)
    # adjust_wspace = par.get('adjust_wspace', 0.3)
    # adjust_hspace = par.get('adjust_hspace', 0.5)

    # Set all parameters
    verbose = kwargs.get('verbose', False)
    style.use(str('ggplot'))
    figsize = kwargs.get('figsize', (10, 10))
    grid_rows = kwargs.get('grid_rows', 4)
    grid_columns = kwargs.get('grid_columns', 3)
    max_grid_position = grid_rows * grid_columns
    font_titles = kwargs.get('font_titles', 8)
    font_axis = kwargs.get('font_axis', 8)
    adjust_wspace = kwargs.get('adjust_wspace', 0.5)
    adjust_hspace = kwargs.get('adjust_hspace', 0.2)

    # Initialize figure and plot count
    fig = plt.figure(num=88, figsize=figsize)

    # Utility function to set subplots
    def set_plot_ax(data_to_plot,
                    count,
                    plot_title='Title',
                    x_label='x',
                    y_label='y',
                    figure=fig,
                    row=grid_rows,
                    col=grid_columns,
                    ft_titles=font_titles,
                    ft_axis=font_axis):
        print_log('Entered set_plot_ax for {plot_title}-{row}-{col}-{count}', verbose=verbose)

        axes = figure.add_subplot(row, col, count)
        axes.plot(data_to_plot, alpha=0.75)

        axes.set_title(plot_title, fontsize=ft_titles)
        axes.set_xlabel(x_label, fontsize=ft_axis)
        axes.set_ylabel(y_label, fontsize=ft_axis)

        max_value = data_to_plot.abs().max()
        y_limit = max(0.25, max_value)
        # axes.set_ylim((-y_limit,+y_limit)) modified into following line on Nov 15, 2019
        axes.set_ylim(-y_limit, +y_limit)

        for label in axes.xaxis.get_ticklabels():
            label.set_fontsize(ft_axis)
        for label in axes.yaxis.get_ticklabels():
            label.set_fontsize(ft_axis)

    def adjust_subplot_and_plot():
        adjust_left = 0.1
        adjust_right = 0.9
        adjust_bottom = 0.1
        adjust_top = 0.9
        adjust_wspace = grid_columns / 10
        adjust_hspace = 0.2 + 0.2 * (grid_rows - 2)
        # plt.subplots_adjust(left=adjust_left,
        #                     right=adjust_right,
        #                     bottom=adjust_bottom,
        #                     top=adjust_top,
        #                     wspace=adjust_wspace,
        #                     hspace=adjust_hspace)
        plt.tight_layout()
        plt.show()

    # Main loop to plot each item in dictionary

    plot_set_count = 0
    for key, time_series in time_series_dict.items():
        data_to_plot = fctn(time_series, *args, **kwargs)

        plot_set_count += 1
        if plot_set_count > max_grid_position:
            if verbose is True:
                print('---------------------------------------------')
                print('| Figure is full, cannot print more subplot |')
                print('---------------------------------------------')
            adjust_subplot_and_plot()
            return

        set_plot_ax(data_to_plot=data_to_plot,
                    figure=fig,
                    count=plot_set_count,
                    plot_title=key,
                    x_label='Time',
                    y_label='Autocorr.')

    adjust_subplot_and_plot()


# --------------------------------------------------------------------------------------------------------------------
# --- Function used for gap fading analysis.
# ------------------------------------------

#  CONSTANTS
PCT_SPREAD = 0.00015
# PCT_SPREAD = 0.00015   # for FOREX Axitrader
# PCT_SPREAD = 0.00025  # for S&P
# PCT_SPREAD = 0.00050   # for EU50, SPI200, ...

# --------------------------------------------------------------------------------------------------------------------
# --- Functions used for apply() method
# ---------------------------------------------------------


def estimate_pnl(series, pct_spread=PCT_SPREAD):
    o, h, l, c, g, previous_bar, faded, label = (series['Open'], series['High'], series['Low'], series['Close'],
                                                 series['Gap'], series['Previous Bar'], series['1-Bar Fade?'],
                                                 series['Zone Label'])
    p_open, p_high, p_low, p_close = previous_bar

    if faded:
        return abs(g) - o * pct_spread
    else:
        if g > 0:
            if label == 'HH':
                sl = p_high + 2 * abs(g)
            else:
                sl = p_high
            return o * (1 - pct_spread) - min(sl, c)
        else:
            if label == 'LL':
                sl = p_low - 2 * abs(g)
            else:
                sl = p_low
            return max(sl, c) - o * (1 + pct_spread)


def gap2body(series, pct_spread=PCT_SPREAD):
    o, h, l, c = series['Previous Bar']
    new_open = (series['Open'])
    gap = new_open - c
    if abs(gap) < c * pct_spread:
        gap = 0
    body = abs(o - c)
    g2b = abs(gap / body) if body != 0 else 0
    return g2b


def gap2wick(series, pct_spread=PCT_SPREAD):
    o, h, l, c = series['Previous Bar']
    new_open = (series['Open'])
    gap = new_open - c
    if abs(gap) < c * pct_spread:
        gap = 0
    wick = h - l
    g2w = abs(gap / wick) if wick != 0 else 0
    return g2w


def gap_fade_in_crt_bar(series):
    h, l, g, previous_bar = (series['High'], series['Low'], series['Gap'], series['Previous Bar'])
    p_close = previous_bar[3]

    if g > 0 and l < p_close:
        return True
    elif g < 0 and h > p_close:
        return True
    else:
        return False


def gap_size(series, pct_spread=PCT_SPREAD):
    o, h, l, c = series['Previous Bar']
    new_open = (series['Open'])
    gap = new_open - c
    if abs(gap) < c * pct_spread:
        gap = 0
    return gap


def ohlc_tuple(series):
    return series['Open'], series['High'], series['Low'], series['Close']


def set_gap_zone_label(series):
    """Return the gap zone label for each gap, based on new open position compared to previous bar ohlc"""

    o, h, l, c = series['Previous Bar']
    new_open, g = (series['Open'], series['Gap'])

    if g == 0:
        return '--'
    if new_open > h:
        return 'HH'
    elif new_open > max(o, c):
        return 'BH'
    elif new_open > min(o, c):
        return 'BB'
    elif new_open > l:
        return 'BL'
    else:
        return 'LL'


def true_range(series):
    h0, l0 = series['High'], series['Low']
    o1, h1, l1, c1 = series['Previous Bar']
    tr = max(h0 - l0, abs(h0 - c1), abs(l0 - c1))
    return tr


def extract_gap_dataset(dataframe, pct_spread=None, atr_window=5):
    """
    Adds features related to gap analysis to each bar in the passed dataframe

    Previous Bar:   ohlc of previous bar so that df.apply() can have access to historical info
    TR:             true range
    ATR:            average true range
    Gap:            current bar open - previous bar close
    Gap %:          gap / ATR
    Gap2Body %:     gap / previous bar body size
    Gap2Wick %:     gap / previous bar H L range
    Momentum:       direction of last bar
    Zone Label:     position of gap w.r.t. previous bar (HH, BH, BB, BL, LL)
    1-Bar Fade?:    True if the gap closes within the same bar, False otherwuse
    P&L Est.:       Estimation of the P&L for a trade entered at open makt price

    dataframe:      dataframe with instrument's prices. Required columns are 'Open', 'High', 'Low', 'Close'
    pct_spread:     (optional) percent spread for the given instrument
    atr_windows:    (optional) window to calculate ATR. Default is 5
    """
    if pct_spread is None:
        pct_spread = PCT_SPREAD

    df = dataframe.copy()
    df.loc[:, 'Previous Bar'] = df.loc[:, ['Open', 'High', 'Low', 'Close']].shift(1).apply(ohlc_tuple, axis='columns')
    df.loc[:, 'TR'] = df.loc[:, ['High', 'Low', 'Previous Bar']].apply(true_range, axis='columns')
    df.loc[:, 'ATR'] = df.loc[:, 'TR'].rolling(atr_window).mean()
    df.loc[:, 'Gap'] = df.loc[:, ['Open', 'Previous Bar']].apply(gap_size, axis='columns')
    df.loc[:, 'Gap %'] = (df.loc[:, 'Gap'] / df.loc[:, 'ATR'].shift(1)).abs() * 100
    df.loc[:, 'Gap2Body %'] = df.loc[:, ['Open', 'Previous Bar']].apply(gap2body, axis='columns') * 100
    df.loc[:, 'Gap2Wick %'] = df.loc[:, ['Open', 'Previous Bar']].apply(gap2wick, axis='columns') * 100
    df.loc[:, 'Momentum'] = (df.loc[:, 'Close'] - df.loc[:, 'Open']).shift(1).apply(lambda x: 'U' if x >= 0 else 'D')
    df.loc[:, 'Zone Label'] = df.loc[:, ['Previous Bar', 'Open', 'Gap']].apply(set_gap_zone_label, axis='columns')
    df.loc[:, '1-Bar Fade?'] = df.loc[:, ['High', 'Low', 'Gap', 'Previous Bar']].apply(gap_fade_in_crt_bar,
                                                                                       axis='columns')
    df.loc[:, 'P&L Est.'] = (df.loc[:, ['Open', 'High', 'Low', 'Close',
                                        'Gap', 'Previous Bar', '1-Bar Fade?',
                                        'Zone Label']]
                             .apply(estimate_pnl, axis='columns'))
    gap_df = df.loc[df.loc[:, 'Zone Label'] != '--', :].dropna().copy()
    gap_df.loc[:, 'MomZone Label'] = gap_df.loc[:, 'Momentum'] + gap_df.loc[:, 'Zone Label']

    return gap_df


def estimate_probabilities(gap_df, show_report=None):
    if show_report is None: show_report = True

    def printif(txt, flag=show_report):
        if show_report: print(txt)

    def displayif(obj, flag=show_report):
        if show_report: display(obj)

    count_gaps = gap_df.shape[0]
    p_fade = gap_df.groupby('1-Bar Fade?').count().loc[:, 'Open'] / count_gaps
    p_fade.name = 'Probability'
    p_fade.index.name = 'Fading Gap?'

    STARS = '************************************************************\n'
    printif(f'{STARS}Total Gap Count: {count_gaps}')
    printif(f'{STARS}Probabilities:')
    printif(f'{STARS}Probability a gap fades within 1 bar = P[Gap Fades] = p_fade[True]')
    displayif(p_fade)  # only used with jupyter where display() is defined

    p_momzone = gap_df.groupby('MomZone Label').count().loc[:, 'Open'] / count_gaps
    p_momzone.name = 'Probability'
    p_momzone.index.name = 'MomZone-Label'
    assert_msg = 'Sum of probabilities for all MomZone (p_momzone) is not equal to 1'
    assert abs(p_momzone.sum() - 1) < 1e-5, assert_msg

    printif(f'{STARS}Probability a gap falls into a momzone = P[Gap in MomZone] = p_momzone["label"]')
    displayif(p_momzone)  # only used with jupyter where display() is defined
    printif(f'Sum for all MomZone: ... {p_momzone.sum():6.5f}')

    counts_momzone_if_fade = gap_df.groupby(['1-Bar Fade?', 'MomZone Label']).count().loc[:, 'Open']
    count_fading_gaps, count_not_fading_gaps = counts_momzone_if_fade[True].sum(), counts_momzone_if_fade[False].sum()

    # check that all possible momzones are in the df, else, add with a count of zero
    MOMZONE_SET = {'UHH', 'UBH', 'UBB', 'UBL', 'ULL', 'DHH', 'DBH', 'DBB', 'DBL', 'DLL'}
    missing_momzones = MOMZONE_SET.difference(set(counts_momzone_if_fade[True].index))
    if len(missing_momzones) != 0:
        for momzone in missing_momzones:
            counts_momzone_if_fade[True, momzone] = 0
    missing_momzones = MOMZONE_SET.difference(set(counts_momzone_if_fade[False].index))
    if len(missing_momzones) != 0:
        for momzone in missing_momzones:
            counts_momzone_if_fade[False, momzone] = 0

    p_momzone_if_fade = counts_momzone_if_fade
    p_momzone_if_fade[True] = counts_momzone_if_fade / count_fading_gaps
    p_momzone_if_fade[False] = counts_momzone_if_fade / count_not_fading_gaps
    p_momzone_if_fade.name = 'Probability'
    assert_msg = 'Sum of probabilities for all MomZone if Fading (p_momzone_if_fade[True]) is not equal to 1'
    assert abs(p_momzone_if_fade[True].sum() - 1) < 1e-5, assert_msg
    assert_msg = 'Sum of probabilities for all MomZone if Not Fading (p_momzone_if_fade[False]) is not equal to 1'
    assert abs(p_momzone_if_fade[False].sum() - 1) < 1e-5, assert_msg

    printif(f'{STARS}Probability gap falls into a momzone if gap fades within same bar')
    printif(f'P[Gap in MomZone "label"|Gap Fades] = p_momzone_if_fade[True]["label"]')
    displayif(p_momzone_if_fade)  # only used with jupyter where display() is defined
    printif(f'Sum for all MomZone if Fading: ....... {p_momzone_if_fade[True].sum():6.5f}')
    printif(f'Sum for all MomZone if Not Fading: ... {p_momzone_if_fade[False].sum():6.5f}')

    p_fades_if_momzone = pd.Series(index=p_momzone.index, name='p_fades_if_momzone')
    p_does_not_fade_if_momzone = pd.Series(index=p_momzone.index, name='p_does_not_fade_if_momzone')

    for label in p_momzone.index:
        p_fades_if_momzone[label] = p_momzone_if_fade[True][label] * p_fade[True] / p_momzone[label]
        p_does_not_fade_if_momzone[label] = p_momzone_if_fade[False][label] * p_fade[False] / p_momzone[label]

    printif(f'{STARS}Probability gap fades if it is in a gap momzone')
    printif(f'P[Gap fades|Gap in MomZone "label"] = p_fades_if_momzone["label"]')
    displayif(p_fades_if_momzone)  # only used with jupyter where display() is defined

    cross_check = pd.DataFrame([p_fades_if_momzone, p_does_not_fade_if_momzone], index=['Fading Gap', 'Not Fading Gap'])
    cross_check.loc['Total', :] = cross_check.loc[['Fading Gap', 'Not Fading Gap'], :].apply(lambda x: x[0] + x[1],
                                                                                             axis='index')
    displayif(cross_check)  # only used with jupyter where display() is defined
    assert_msg = 'p_fades_if_momzone + p_does_not_fade_if_momzone not equal to 1'
    assert all(np.isclose(cross_check.loc['Total', :], 1)), assert_msg

    return p_fade, p_fades_if_momzone, p_does_not_fade_if_momzone


def graph_zone_probabilities(prob_per_momzone):
    DBB, DBH, DBL, DHH, DLL, UBB, UBH, UBL, UHH, ULL = prob_per_momzone

    print(f"--------------------------------------------------")
    # print(f"                                                  ")
    print(f" DHH {DHH:3.1%}                              {UHH:3.1%} UHH ")
    print(f"                                                  ")
    print(f"--------------------------------------------------")
    print(f"                   |           |                  ")
    print(f" DBH {DHH:3.1%}         |           |        {UBH:3.1%} UBH ")
    print(f"                   |           |                  ")
    print(f"---------------- ===== ----- ===== ---------------")
    # print(f"                 |   |       |   |                ")
    print(f"                 |   |       |   |                ")
    print(f" DBB {DBB:3.1%}       | D |       | U |      {UBB:3.1%} UBB ")
    print(f"                 |   |       |   |                ")
    # print(f"                 |   |       |   |                ")
    print(f"---------------- ===== ----- ===== ---------------")
    print(f"                   |           |                  ")
    print(f" DBL {DBL:3.1%}         |           |        {UBL:3.1%} UBL ")
    print(f"                   |           |                  ")
    print(f"--------------------------------------------------")
    print(f"                                                  ")
    print(f" DLL {DLL:3.1%}                              {ULL:3.1%} ULL ")
    # print(f"                                                  ")
    print(f"--------------------------------------------------")


if __name__ == '__main__':
    # price_dict = get_price_dict_from_wsj(ticker='NYA', timeframe='1440', verbose=True)
    # print(list(price_dict))
    #
    print('Running')
    ld = date_latest_alphavantage_update()
    print(f"last date is: {ld:%Y-%m-%d}")
    # print(len(ticker_dict), ticker_dict.keys())
    # update_price_datasets_yahoo(['AAAA'])

    # raw_data = get_module_root_path() / 'data/raw-data/'
    # axitrader = raw_data / 'axitrader-mt4'
    # xmcom = raw_data / 'xm-com-mt4'
    #
    # axidf = get_price_dict_from_MT4(ticker='EURUSD', timeframe='1440', target_directory=xmcom)
    #
    # print('done')

    # price_dict = get_price_dict_from_alphavantage(ticker='EURUSD', timeframe='FX_DAILY',
    #                                               target_directory=None, verbose=False)
    # print(price_dict)

    # update_alphavantage_fx(pairs=['KRWUSD'], timeframes=['1w'], alphavantage_mode='compact', verbose=True)
    # update_alphavantage_fx(alphavantage_mode='compact', verbose=True)

    # print(config)
    # print(config['module_root_directory'])
    # print(config['alphavantage'])
    # print(f'current working directory {Path().cwd()}')
    # print(f"data directory {get_module_root_path() / 'data' / 'metatrader-mt4'}")

    pass