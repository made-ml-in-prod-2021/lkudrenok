import pytest
import numpy as np
import pandas as pd

from src.utils.param_classes import FeatureParams
from src.utils.build_features import (
    process_numerical_features, process_categorical_features,
    build_transformer, make_features
)


NUM1_ARRAY = np.array([15.6, 45.2, 12.4, 66.4])
NUM2_ARRAY = np.array([5, 2, 6, 5])


def scale(data: np.ndarray,
          min_value: np.ndarray = None,
          max_value: np.ndarray = None) -> np.ndarray:
    if min_value is None:
        min_value = data.min(axis=0)
    if max_value is None:
        max_value = data.max(axis=0)
    return np.clip((data - min_value) / (max_value - min_value), 0, 1)


@pytest.fixture
def feature_params() -> FeatureParams:
    feature_params = FeatureParams(
        categorical_features=['cat1', 'cat2'],
        numerical_features=['num1', 'num2'],
        target_col='target'
    )
    return feature_params


@pytest.fixture
def train_data() -> pd.DataFrame:
    train_data = pd.DataFrame({
        'cat1': [1, 0, 1, 0],
        'cat2': ['a', 'b', 'c', 'a'],
        'num1': NUM1_ARRAY,
        'num2': NUM2_ARRAY,
        'target': [1, 0, 1, 0]
    })
    return train_data


@pytest.fixture
def etalon_transformed_data() -> np.ndarray:
    etalon_transformed_data = pd.DataFrame({
        0: [1, 0, 1, 0],
        1: [0, 1, 0, 0],
        2: [0, 0, 1, 0],
        3: scale(NUM1_ARRAY),
        4: scale(NUM2_ARRAY),
    }).to_numpy()
    return etalon_transformed_data


def test_process_numerical_features(feature_params: FeatureParams,
                                    train_data: pd.DataFrame,
                                    etalon_transformed_data: np.ndarray):
    data = train_data[feature_params.numerical_features]
    processed = process_numerical_features(data)
    etalon = etalon_transformed_data[:, 3:5]
    np.testing.assert_allclose(processed.values, etalon)


def test_process_categorical_features(feature_params: FeatureParams,
                                      train_data: pd.DataFrame,
                                      etalon_transformed_data: np.ndarray):
    data = train_data[feature_params.categorical_features]
    processed = process_categorical_features(data)
    etalon = etalon_transformed_data[:, :3]
    assert (processed.values == etalon).all()


def test_build_transformer(feature_params: FeatureParams,
                           train_data: pd.DataFrame,
                           etalon_transformed_data: np.ndarray):
    transformer = build_transformer(feature_params)
    transformed = transformer.fit_transform(train_data)
    np.testing.assert_allclose(transformed, etalon_transformed_data)


def test_make_features(feature_params: FeatureParams,
                       train_data: pd.DataFrame,
                       etalon_transformed_data: np.ndarray):
    transformer = build_transformer(feature_params).fit(train_data)
    features = make_features(transformer, train_data)
    np.testing.assert_allclose(features, etalon_transformed_data)
