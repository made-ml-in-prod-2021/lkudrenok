import pytest
import numpy as np
import pandas as pd

from ml_project.utils.param_classes import FeatureParams
from ml_project.utils.build_features import (
    scale, SimpleScaler,
    process_numerical_features, process_categorical_features,
    build_transformer, make_features, extract_target
)


NUM1_ARRAY = np.array([15.6, 45.2, 12.4, 66.4])
NUM2_ARRAY = np.array([5, 2, 6, 5])


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


def test_scale():
    data = pd.DataFrame({
        0: [1, 2, 3],
        1: [50, 10, 20]
    }).values
    min_value = data.min(axis=0)
    max_value = data.max(axis=0)
    data_scaled = scale(data, min_value, max_value)
    etalon = pd.DataFrame({
        0: [0, 0.5, 1],
        1: [1, 0, 0.25]
    }).values
    np.testing.assert_allclose(data_scaled, etalon)

    data_new = pd.DataFrame({
        0: [1, 2, 3, 4, 2],
        1: [50, 10, 20, 0, 40]
    }).values
    data_scaled_new = scale(data_new, min_value, max_value)
    etalon_new = pd.DataFrame({
        0: [0, 0.5, 1, 1, 0.5],
        1: [1, 0, 0.25, 0, 0.75]
    }).values
    np.testing.assert_allclose(data_scaled_new, etalon_new)


def test_simple_scaler(train_data, etalon_transformed_data):
    data = train_data[['num1', 'num2']]
    simple_scaler = SimpleScaler().fit(data)
    transformed_data = simple_scaler.transform(data)
    etalon_data = etalon_transformed_data[:, 3:5]
    np.testing.assert_allclose(transformed_data, etalon_data)


def test_simple_scaler_fitted_with_params():
    scaler_params = {'min_value': 10, 'max_value': 50}
    data = np.array([0, 10, 20, 40, 50, 60])
    simple_scaler = SimpleScaler(**scaler_params)
    transformed_data = simple_scaler.transform(data)
    etalon_data = np.array([0, 0, 0.25, 0.75, 1, 1])
    np.testing.assert_allclose(transformed_data, etalon_data)


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


def test_extract_target(feature_params: FeatureParams,
                        train_data: pd.DataFrame):
    target = extract_target(train_data, feature_params)
    etalon = pd.Series([1, 0, 1, 0])
    assert target.equals(etalon)
