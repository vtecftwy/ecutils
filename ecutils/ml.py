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
import subprocess

from pathlib import Path
from IPython.display import Image, display
from pprint import pprint
from zipfile import ZipFile

__all__ = ['kaggle_setup_colab', 'kaggle_list_files', 'kaggle_download_files' ]

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


def kaggle_setup_colab(path_to_config_file=None):
    """Update kaggle API and create security key json file from config file on Google Drive

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

    # Update kaggle API software
    run_cli('pip install kaggle --upgrade')


def kaggle_list_files(competition_code=None):
    """List all files available for the passed competition"""
    if competition_code is None:
        print(f"competition_code is None, please provide the code of the kaggle competition")
        return 'Failed'
    else:
        print(f"Listing the files available for <{competition_code}>")
        run_cli(f"kaggle competitions files {competition_code}")

        print(f"{'=' * 140}")
        print(f"Make sure to set the parameters for <{kaggle_competion_code}> in next cell:")
        print(f" - kaggle_project_folder_name: string with name of the project folder")
        print(f" - train_files: list of files to place into the <train> folder")
        print(f" - test_files: list of files to place into the <test> folder")
        print(f" - submit_files: list of files to place into the <submit> folder")
        print(f"{'=' * 140}")


def kaggle_download_files(competition_code=None, train_files=[], test_files=[], submit_files=[]):
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
        if not os.path.exists(kaggle_project_folder_name):
            os.makedirs(kaggle_project_folder_name)

        path2datasets = Path(f"/content/{kaggle_project_folder_name}")
        path2datasets_str = str(path2datasets.absolute())

        # download all files from kaggle
        run_cli(f"kaggle competitions download -c {kaggle_competion_code} -p {path2datasets}")

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
            if not os.path.exists(f'{kaggle_project_folder_name}/{dataset_folder}'):
                os.makedirs(f'{kaggle_project_folder_name}/{dataset_folder}')

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
