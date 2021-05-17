import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml_project.utils.param_classes import FeatureParams


class CustomTransformer(ColumnTransformer):
    """ColumnTransformer which allows to export SimpleScaler params."""

    def __init__(self, transformers):
        super().__init__(transformers)

    def export_scaler_params(self):
        return self.named_transformers_.numerical_pipeline.steps[0][1].export_params()


def scale(data: np.ndarray,
          min_value: np.ndarray = None,
          max_value: np.ndarray = None) -> np.ndarray:
    if min_value is None:
        min_value = data.min(axis=0)
    if max_value is None:
        max_value = data.max(axis=0)
    return np.clip((data - min_value) / (max_value - min_value), 0, 1)


class SimpleScaler:
    """Custom transformer to scale data from 0 to 1 by each column."""

    def __init__(self, min_value=None, max_value=None):
        self._is_fitted = False
        if max_value is not None and max_value is not None:
            self._min_value = np.array(min_value)
            self._max_value = np.array(max_value)
            self._is_fitted = True

    def export_params(self):
        return {
            'min_value': self._min_value.tolist(),
            'max_value': self._max_value.tolist()
        }

    def fit(self, data, *args):
        if not self._is_fitted:
            data = np.array(data)
            self._min_value = data.min(axis=0)
            self._max_value = data.max(axis=0)
            self._is_fitted = True
        return self

    def transform(self, data):
        data = np.array(data)
        data_scaled = scale(data, self._min_value, self._max_value)
        return data_scaled


def build_categorical_pipeline(**kwargs) -> Pipeline:
    categorical_pipeline = Pipeline([
        ('one_hot', OneHotEncoder(drop='first'))
    ])
    return categorical_pipeline


def process_categorical_features(categorical_df: pd.DataFrame) -> pd.DataFrame:
    categorical_pipeline = build_categorical_pipeline()
    processed = pd.DataFrame(
        categorical_pipeline.fit_transform(categorical_df).toarray()
    )
    return processed


def build_numerical_pipeline(**kwargs) -> Pipeline:
    if kwargs.get('scaler_params'):
        numerical_pipeline = Pipeline([
            ('scale', SimpleScaler(**kwargs.get('scaler_params')))
        ])
    else:
        numerical_pipeline = Pipeline([
            ('scale', SimpleScaler())
        ])
    return numerical_pipeline


def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    numerical_pipeline = build_numerical_pipeline()
    processed = pd.DataFrame(
        numerical_pipeline.fit_transform(numerical_df)
    )
    return processed


def build_transformer(feature_params: FeatureParams, **kwargs) -> CustomTransformer:
    transformer = CustomTransformer([
        (
            'categorical_pipeline',
            build_categorical_pipeline(**kwargs),
            feature_params.categorical_features
        ),
        (
            'numerical_pipeline',
            build_numerical_pipeline(**kwargs),
            feature_params.numerical_features
        )
    ])
    return transformer


def make_features(transformer: ColumnTransformer, data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(transformer.transform(data))


def extract_target(data: pd.DataFrame, params: FeatureParams) -> pd.Series:
    return data[params.target_col]
