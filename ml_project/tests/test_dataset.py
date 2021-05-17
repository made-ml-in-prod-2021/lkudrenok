import numpy as np
import pandas as pd

from ml_project.utils.make_dataset import load_data, split_train_val_data
from ml_project.utils.param_classes import SplittingParams

from tests.conftest import data, data_file, splitting_params


def test_load_data(data: pd.DataFrame, data_file):
    data_loaded = load_data(data_file)
    np.testing.assert_array_equal(data_loaded.values, data.values)


def test_split_train_val_data(data: pd.DataFrame, splitting_params: SplittingParams):
    assert list(data.index)[:10] == list(range(10))
    train_df, val_df = split_train_val_data(data, splitting_params)
    np.testing.assert_almost_equal(val_df.shape[0], data.shape[0] * splitting_params.val_size)
    assert list(train_df.index)[:10] != list(range(10))
