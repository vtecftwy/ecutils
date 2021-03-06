import datetime as dt
import inspect
import pytest
import numpy as np
import pandas as pd
import pkg_resources

from numpy.testing import assert_almost_equal
from pathlib import Path
from unittest.mock import patch

from ecutils import ml


class TestMLAreFeaturesConsistent:
    def test_ml_are_features_consistent_correct_returns_true(self):
        """Test that the function returns True when features are consistent"""

        # Setup
        a = np.arange(1, 11)
        train_df = pd.DataFrame({'a': a,
                                 'b': a * 2,
                                 'c': a * 3,
                                 'target': a*10})


        test_df = train_df.drop(labels='target', axis=1)

        # Exercise
        res = ml.are_features_consistent(train_df=train_df,
                                         test_df=test_df,
                                         dependent_variables='target')

        # Verify
        assert res

        # Cleanup

    def test_ml_are_features_consistent_no_target_returns_true(self):
        """Test that the function returns True when features are consistent and there is no dependent variable"""

        # Setup
        a = np.arange(1, 11)
        train_df = pd.DataFrame({'a': a,
                                 'b': a * 2,
                                 'c': a * 3})
        test_df = train_df

        # Exercise
        res = ml.are_features_consistent(train_df=train_df,
                                         test_df=test_df)

        # Verify
        assert res

        # Cleanup


class TestMLSetupKaggleSecurity:
    def test_ml_setup_kaggle_security(self):
        """Test """
        # Setup
        p2config = Path(pkg_resources.resource_filename(__name__, '/fixed-assets/config-api-keys.cfg'))

        # Exercise
        kaggle_username = ml.get_config_value(section='kaggle', key='kaggle_username', path_to_config_file=p2config)
        kaggle_key = ml.get_config_value(section='kaggle', key='kaggle_key', path_to_config_file=p2config)
        azure_key = ml.get_config_value(section='azure', key='fastai-image-search-sh-1', path_to_config_file=p2config)

        # Verify
        assert kaggle_username == 'dummy_kaggle_user'
        assert kaggle_key == 'dummy_kaggle_key'
        assert azure_key == 'dummy_azure_key_01'