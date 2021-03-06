"""
Utility Functions that can be used for Kaggle and other ML uses

Includes all stable utility functions.

Reference for kaggle API: https://github.com/Kaggle/kaggle-api
"""


import configparser
import datetime as dt
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
import subprocess

from pathlib import Path
from IPython.display import Image, display
from pprint import pprint
from zipfile import ZipFile

__all__ = ['kaggle_setup_colab', 'kaggle_list_files', 'kaggle_download_competition_files', 'are_features_consistent' ]


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


def run_cli(cmd='ls -l'):
    """Wrapper to use subprocess.run with passed command, and print the shell messages

    cmd: str    cli command to execute
    """
    p = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
    print(str(p.stdout, 'utf-8'))


def get_config_value(section, key, path_to_config_file=None):
    """Returns the value corresponding to the key-value pair in the configuration file (configparser format)

    By defaults, it is assumed that the configuration file is saved on google drive. If not, pass a proper Path object.
    The configuration file must be in the format of configparser (https://docs.python.org/3/library/configparser.html)

    Parameters:
        section (str):          name of the section where the key-value pair is stored
        key (str):              name of the key
        path_to_config_file(Path or str): path to the configparser configuration file

    Return (str):               value in the key-value pair stored
    """
    if path_to_config_file is None:
        path_to_config_file = Path(f"/content/gdrive/My Drive/config-api-keys.cfg")
    elif isinstance(path_to_config_file, str):
        path_to_config_file = Path(f"/content/gdrive/My Drive/{path_to_config_file}")

    msg = f"Cannot find file {path_to_config_file}. Please check the path or add the config file at that location"
    assert path_to_config_file.is_file(), msg

    configuration = configparser.ConfigParser()
    configuration.read(path_to_config_file)
    return configuration[section][key]

def fastbook_on_colab():
    """
    Set up environment to run fastbook notebooks for colab

    Code from notebook:
    # Install fastbook and dependencies
    !pip install -Uqq fastbook

    # Load utilities and install them
    !wget -O utils.py https://raw.githubusercontent.com/vtecftwy/fastbook/walk-thru/utils.py
    !wget -O fastbook_utils.py https://raw.githubusercontent.com/vtecftwy/fastbook/walk-thru/fastbook_utils.py

    from fastbook_utils import *
    from utils import *

    # Setup My Drive
    setup_book()

    # Download images and code required for this notebook
    import os
    os.makedirs('images', exist_ok=True)
    !wget -O images/chapter1_cat_example.jpg https://raw.githubusercontent.com/vtecftwy/fastai-course-v4/master/nbs/images/chapter1_cat_example.jpg
    !wget -O images/cat-01.jpg https://raw.githubusercontent.com/vtecftwy/fastai-course-v4/walk-thru/nbs/images/cat-01.jpg
    !wget -O images/cat-02.jpg https://raw.githubusercontent.com/vtecftwy/fastai-course-v4/walk-thru/nbs/images/cat-02.jpg
    !wget -O images/dog-01.jpg https://raw.githubusercontent.com/vtecftwy/fastai-course-v4/walk-thru/nbs/images/dog-01.jpg
    !wget -O images/dog-02.jpg https://raw.githubusercontent.com/vtecftwy/fastai-course-v4/walk-thru/nbs/images/dog-01.jpg

    """
    instructions = ['pip install -Uqq fastbook',
                    'wget -O utils.py https://raw.githubusercontent.com/vtecftwy/fastbook/walk-thru/utils.py',
                    'wget -O fastbook_utils.py https://raw.githubusercontent.com/vtecftwy/fastbook/walk-thru/fastbook_utils.py'
                    ]


def kaggle_setup_colab(path_to_config_file=None):
    """Update kaggle API and create security key json file from config file on Google Drive

    Kaggle API documentation: https://github.com/Kaggle/kaggle-api

    To access Kaggle with API, a security key needs to be placed in the correct location on colab.
    config.cfg file must include the following lines:
        [kaggle]
            kaggle_username = kaggle_user_name
            kaggle_key = API key provided by kaggle

        Info on how to get your api key (kaggle.json) here: https://github.com/Kaggle/kaggle-api#api-credentials

    path_to_config_file: str or Path:   path to the configuration file (e.g. config.cfg)

    """
    # Create API security key file
    path_to_kaggle = Path('/root/.kaggle')
    os.makedirs(path_to_kaggle, exist_ok=True)

    username = get_config_value('kaggle', 'kaggle_username', path_to_config_file=path_to_config_file)
    key = get_config_value('kaggle', 'kaggle_key', path_to_config_file=path_to_config_file)

    api_token = {"username": username, "key": key}
    with open(path_to_kaggle / 'kaggle.json', 'w') as file:
        json.dump(api_token, file)
        os.fchmod(file.fileno(), 600)

    # Update kaggle API software
    run_cli('pip install kaggle --upgrade')


def kaggle_list_files(code=None, mode='competitions'):
    """List all files available in the competition or dataset for the passed code"""
    if code is None:
        print(f"code is None, please provide the code of the kaggle competition or dataset")
        return 'Failed'
    elif mode not in ['competitions', 'datasets']:
        print(f"mode must be either 'competitions' or 'datasets', not {mode}")
        return 'Failed'
    else:
        print(f"Listing the files available for {mode}: <{code}>")
        run_cli(f"kaggle {mode} files {code}")

    if mode == 'competitions':
        print(f"{'=' * 140}")
        print(f"Make sure to set the parameters for <{code}> in next cell:")
        print(f" - kaggle_project_folder_name: string with name of the project folder")
        print(f" - train_files: list of files to place into the <train> folder")
        print(f" - test_files: list of files to place into the <test> folder")
        print(f" - submit_files: list of files to place into the <submit> folder")
        print(f"{'=' * 140}")


def kaggle_download_competition_files(competition_code=None, train_files=[], test_files=[], submit_files=[], project_folder='ds' ):
    """download all files for passed competition, unzip them if required, move them to train, test and submit folders

    competition_code: str       code of the kaggle competition
    train_files: list of str    names of files to be moved into train folder
    test_files: list of str     names of files to be moved into test folder
    submit_files: list of str   names of files to be moved into submit folder
    """
    if competition_code is None:
        print(f"competition_code is None, please provide the code of the kaggle competition")
        return 'Failed'
    else:
        list_of_datasets = {'train': train_files,
                            'test': test_files,
                            'submit': submit_files}

        # creating a project directory and set paths
        if not os.path.exists(project_folder):
            os.makedirs(project_folder)

        path2datasets = Path(f"/content/{project_folder}")
        path2datasets_str = str(path2datasets.absolute())

        # download all files from kaggle
        run_cli(f"kaggle competitions download -c {competition_code} -p {path2datasets}")

        print(f"{'=' * 140}")
        print('Downloaded files:')
        for f in [item for item in path2datasets.iterdir() if item.is_file()]:
            print(f" - {f}")
        print(f"{'=' * 140}")

        # Unzip all zipped files
        for f in path2datasets.glob('*.zip'):
            print(f"Unzipping {f.name}")
            zip_f = ZipFile(f)
            zip_f.extractall(path=path2datasets)
            os.remove(f)
        print(f"{'=' * 140}")

        # Move all data files to the correct data folder
        for dataset_folder, files in list_of_datasets.items():
            if not os.path.exists(f'{project_folder}/{dataset_folder}'):
                os.makedirs(f'{project_folder}/{dataset_folder}')

            for f in files:
                print(f"Moving {f} to {dataset_folder}")
                p2f = path2datasets / f
                if p2f.suffix == '.csv':
                    shutil.move(path2datasets / f, path2datasets / dataset_folder / f)
                else:
                    msg = f"Does not support {p2f.name}'s extension {p2f.suffix}"
                    raise RuntimeError(msg)

        print(f"{'=' * 140}")
        print('Done loading Kaggle files and moving them to corresponding folders')


if __name__ == "__main__":
    pass
