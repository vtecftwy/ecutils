"""
Utility Functions that can be used for Kaggle and other ML uses

Includes all stable utility functions.
"""

import configparser
import datetime as dt
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil

from pathlib import Path
from IPython.display import Image, display
from pprint import pprint

__all__ = ['set_kaggle_security_key_colab', ]

def are_features_consistent(train_df, test_df, dependent_variables=None):
    """Verifies that features in training and test sets are consistent

    Training set and test set should have the same features/columns, except for the dependent variables

    train_dr: pd.DataFrame      training dataset
    test_df:  pd.DataFrame      test dataset
    dependent_variables: list   list of column names for the dependent variables
    """
    if dependent_variables is None:
        features_training_set = train_df.columns
    else:
        features_training_set = train_df.drop(dependent_variables, axis=1).columns
    features_test_set = test_df.columns
    features_diff = set(features_training_set).symmetric_difference(features_test_set)
    assert features_diff == set(), f"Discrepancy between training and test feature set: {features_diff}"

    return True


def set_kaggle_security_key_colab(path_to_config_file=None):
    """Set up the Kaggle security key json file from config file on Google Drive

    To access Kaggle with API, a security key needs to be placed in the correct location on colab.
    config.cfg file must include the following lines:
        [kaggle]
            kaggle_username = kaggle_user_name
            kaggle_key = API key provided by kaggle

        Info on how to get your api key (kaggle.json) here: https://github.com/Kaggle/kaggle-api#api-credentials

    path_to_config_file: str or Path:   path to the configuration file (e.g. config.cfg)

    """
    path_to_kaggle = Path('/root/.kaggle')
    os.makedirs(path_to_kaggle, exist_ok=True)
    if path_to_config_file is None:
        path_to_config_file = Path(f"/content/gdrive/My Drive/config.cfg")
    elif isinstance(path_to_config_file, str):
        path_to_config_file = Path(f"/content/gdrive/My Drive/{path_to_config_file}")


    msg = f"Cannot find file {path_to_config_file}. Please check the path or add the config file at that location"
    assert path_to_config_file.is_file(), msg

    configuration = configparser.ConfigParser()
    configuration.read(path_to_config_file)
    username = configuration['kaggle']['kaggle_username']
    key = configuration['kaggle']['kaggle_key']
    api_token = {"username": username, "key": key}
    with open(path_to_kaggle / 'kaggle.json', 'w') as file:
        json.dump(api_token, file)
        os.fchmod(file.fileno(), 600)




if __name__ == "__main__":
    pass
