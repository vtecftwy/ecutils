import datetime as dt
import inspect
import pytest
import numpy as np
import pandas as pd
import sys

from ecutils import historical_price_handling as fxutils
from numpy.testing import assert_almost_equal
from pathlib import Path

# --------------------------------------------------------------------------------------------------------------------
# --- General Utility Functions
# ------------------------------


# def test__get_module_root_path_no_param():
#     """ Test that get module returns the correct path when no parameters are passed"""
#     # ToDo: reconsider the need for this function. Use standard lib function instead. Then delete this test
#     # Setup
#     desired_path = Path('D:\\PyProjects\\ec-utils').absolute()
#
#     # Exercise
#     computed_root_path = fxutils.get_module_root_path().absolute()
#
#     # Verify
#     assert computed_root_path == desired_path
#
#     # Cleanup (none)


def test__get_module_root_path_param_correct():
    """Test that get_module returns the correct path when the correct root dir param is passed"""
    # Setup
    desired_path = Path('D:\\PyProjects\\ec-utils').absolute()

    # Exercise
    computed_root_path =  fxutils.get_module_root_path(module_root='ec-utils').absolute()

    # Verify
    assert computed_root_path == desired_path

    # Cleanup (none)


def test__get_module_root_path_param_erronous():
    """Test that an error is raised in case the module_root parameter is not the same as the module root"""
    # Setup
    module_root = 'not_correct_name'

    # Exercise and Verify
    with pytest.raises(ValueError):
        fxutils.get_module_root_path(module_root=module_root)

    # Cleanup (None)


def test__remove_multiple_wildcards_double():
    """Test that the function removes double wildcards and not single wildcards"""
    assert fxutils.remove_multiple_wildcards('123*321') == '123*321'
    assert fxutils.remove_multiple_wildcards('123**321') == '123*321'
    assert fxutils.remove_multiple_wildcards('**321') == '*321'
    assert fxutils.remove_multiple_wildcards('321**') == '321*'


def test__remove_multiple_wildcards_triple():
    """Test that the function removes triple wildcards and not single wildcards"""
    assert fxutils.remove_multiple_wildcards('123*321') == '123*321'
    assert fxutils.remove_multiple_wildcards('123***321') == '123*321'
    assert fxutils.remove_multiple_wildcards('***321') == '*321'
    assert fxutils.remove_multiple_wildcards('321***') == '321*'


def test__remove_multiple_wildcards_quad():
    """Test that the function removes quadruple wildcards and not single wildcards"""
    assert fxutils.remove_multiple_wildcards('123*321') == '123*321'
    assert fxutils.remove_multiple_wildcards('123****321') == '123*321'
    assert fxutils.remove_multiple_wildcards('****321') == '*321'
    assert fxutils.remove_multiple_wildcards('321****') == '321*'


def test__fill_nan():
    """Test that nan """
    # Setup
    df_col1 = np.arange(0, 10, 1)
    df_col2 = np.arange(10, 20, 1)
    df = pd.DataFrame(data={'Col 1': df_col1, 'Col 2': df_col2})
    df.iloc[0, 0] = df.iloc[1, 0] = df.iloc[4, 0] = np.NaN
    df.iloc[0, 1] = df.iloc[5, 1] = df.iloc[9, 1] = np.NaN
    """ test DataFrame:
               Col 1  Col 2
        0    NaN   NaN
        1    NaN   11.0
        2    2.0   12.0
        3    3.0   13.0
        4    NaN   14.0
        5    5.0    NaN
        6    6.0   16.0
        7    7.0   17.0
        8    8.0   18.0
        9    9.0    NaN
    """
    desired_filled_values_col_1 = (2, 2, 3)
    desired_filled_values_col_2 = (11, 14, 18)

    # Exercise
    fill_df = fxutils.fill_nan(df)
    actual_filled_values_col_1 = (fill_df.iloc[0, 0], fill_df.iloc[1, 0], fill_df.iloc[4, 0])
    actual_filled_values_col_2 = (fill_df.iloc[0, 1], fill_df.iloc[5, 1], fill_df.iloc[9, 1])

    # Verify
    assert actual_filled_values_col_1 == desired_filled_values_col_1
    assert actual_filled_values_col_2 == desired_filled_values_col_2


def test__str_date():
    """Test that the str_date function returns a string in the correct format"""
    # Setup
    y = 2000
    m = 5
    d = 12
    date = dt.datetime(year=y, month=m, day=d)
    desired_string = f'{y}-{m}-{d}'

    # Exercise
    actual_string = fxutils.str_date(date)

    # Verify
    assert actual_string == desired_string


def test__estimate_pnl__hh_fade():
    """Test gap fading pnl with there is a HH fade"""
    # Setup
    pct_spread = 0.01
    series = pd.Series(data={'Open': 11, 'High': 17, 'Low': 7, 'Close': 15,
                             'Gap': 3, 'Previous Bar': '', '1-Bar Fade?': True,
                             'Zone Label': 'HH'})
    series.at['Previous Bar'] = (4, 10, 2, 8)

    true_pnl = 3 - 11 * pct_spread

    # Exercise
    estimated_pnl = fxutils.estimate_pnl(series, pct_spread=pct_spread)

    # Verify
    assert estimated_pnl == true_pnl


def test__estimate_pnl__bh_fade():
    """Test gap fading pnl with there is a BH fade"""
    # Setup
    pct_spread = 0.01
    series = pd.Series(data={'Open': 9, 'High': 15, 'Low': 7, 'Close': 13,
                             'Gap': 1, 'Previous Bar': '', '1-Bar Fade?': True,
                             'Zone Label': 'BH'})
    series.at['Previous Bar'] = (4, 10, 2, 8)
    true_pnl = 1 - 9 * pct_spread
    # Exercise
    estimated_pnl = fxutils.estimate_pnl(series, pct_spread=pct_spread)
    # Verify
    assert estimated_pnl == estimated_pnl


def test__estimate_pnl__bb_fade():
    """Test gap fading pnl with there is a BB fade"""
    # Setup
    pct_spread = 0.01
    series = pd.Series(data={'Open': 6, 'High': 12, 'Low': 3, 'Close': 9,
                             'Gap': -2, 'Previous Bar': '', '1-Bar Fade?': True,
                             'Zone Label': 'BB'})
    series.at['Previous Bar'] = (4, 10, 2, 8)
    true_pnl =  2 - 6 * pct_spread
    # Exercise
    estimated_pnl = fxutils.estimate_pnl(series, pct_spread=pct_spread)
    # Verify
    assert estimated_pnl == true_pnl


def test__estimate_pnl__hh_no_fade():
    """Test gap fading pnl with there is a HH and no fade"""
    # Setup
    pct_spread = 0.01
    series = pd.Series(data={'Open': 14, 'High': 20, 'Low': 12, 'Close': 17,
                             'Gap': 6, 'Previous Bar': '', '1-Bar Fade?': False,
                             'Zone Label': 'HH'})
    series.at['Previous Bar'] = (4, 10, 2, 8)
    true_pnl =  14 * (1 - pct_spread) - min(22, 17)
    # Exercise
    estimated_pnl = fxutils.estimate_pnl(series, pct_spread=pct_spread)
    # Verify
    assert estimated_pnl == true_pnl


def test__estimate_pnl__bh_no_fade():
    """Test gap fading pnl with there is a BH and no fade"""
    # Setup
    pct_spread = 0.01
    series = pd.Series(data={'Open': 10, 'High': 16, 'Low': 9, 'Close': 13,
                             'Gap': 3, 'Previous Bar': '', '1-Bar Fade?': False,
                             'Zone Label': 'BH'})
    series.at['Previous Bar'] = (4, 11, 2, 8)
    true_pnl = 1.1      # because stoploss is activated and was placed at 1.1
    # Exercise
    estimated_pnl = abs(fxutils.estimate_pnl(series, pct_spread=pct_spread))
    # Verify
    assert_almost_equal(estimated_pnl, true_pnl)


def test__estimate_pnl__bb_no_fade():
    """Test gap fading pnl with there is a BB and no fade"""
    #Setup
    pct_spread = 0.01
    # Zone BB - No Fade
    series = pd.Series(data={'Open': 10, 'High': 16, 'Low': 1, 'Close': 7,
                             'Gap': -9, 'Previous Bar': '', '1-Bar Fade?': False,
                             'Zone Label': 'BB'})
    series.at['Previous Bar'] = (4, 21, 2, 19)
    true_pnl = 3.1      # because stoploss is activated and was placed at 3.1
    # Exercise
    estimated_pnl = abs(fxutils.estimate_pnl(series, pct_spread=pct_spread))
    # Verify
    assert_almost_equal(estimated_pnl, true_pnl)


def test__estimate_pnl__bl_no_fade():
    """Test gap fading pnl with there is a BL and no fade"""
    # Setup
    pct_spread = 0.01
    series = pd.Series(data={'Open': 10, 'High': 16, 'Low': 1, 'Close': 7,
                             'Gap': -8, 'Previous Bar': '', '1-Bar Fade?': False,
                             'Zone Label': 'BL'})
    series.at['Previous Bar'] = (17, 20, 8, 18)
    true_pnl = 2.1      # because stoploss is activated and was placed at 2.1
    # Exercise
    estimated_pnl = abs(fxutils.estimate_pnl(series, pct_spread=pct_spread))
    # Verify
    assert_almost_equal(estimated_pnl, true_pnl)


def test__estimate_pnl__ll_no_fade():
    """Test gap fading pnl with there is a LL and no fade"""
    # Setup
    pct_spread = 0.01
    series = pd.Series(data={'Open': 20, 'High': 26, 'Low': 11, 'Close': 17,
                             'Gap': -8, 'Previous Bar': '', '1-Bar Fade?': False,
                             'Zone Label': 'LL'})
    series.at['Previous Bar'] = (27, 30, 21, 28)
    true_pnl = 3.2      # because stoploss is activated and was placed at 2.1
    # Exercise
    estimated_pnl = abs(fxutils.estimate_pnl(series, pct_spread=pct_spread))
    # Verify
    assert_almost_equal(estimated_pnl, true_pnl)