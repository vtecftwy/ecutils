import datetime as dt
import inspect
import pytest
import numpy as np
import pandas as pd
import sys

from numpy.testing import assert_almost_equal
from pathlib import Path

from ecutils import ml


class TestAreFeaturesConsistent:
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
