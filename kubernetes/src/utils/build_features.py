import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from src.utils.param_classes import FeatureParams


def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def build_categorical_pipeline() -> Pipeline:
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


def build_numerical_pipeline() -> Pipeline:
    numerical_pipeline = Pipeline([
            ('scale', MinMaxScaler(clip=True))
    ])
    return numerical_pipeline


def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    numerical_pipeline = build_numerical_pipeline()
    processed = pd.DataFrame(
        numerical_pipeline.fit_transform(numerical_df)
    )
    return processed


def build_transformer(feature_params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer([
        (
            'categorical_pipeline',
            build_categorical_pipeline(),
            feature_params.categorical_features
        ),
        (
            'numerical_pipeline',
            build_numerical_pipeline(),
            feature_params.numerical_features
        )
    ])
    return transformer


def make_features(transformer: ColumnTransformer,
                  data: pd.DataFrame,
                  with_target: bool = True) -> pd.DataFrame:
    if not with_target:
        data['target'] = None
    return pd.DataFrame(transformer.transform(data))


def extract_target(data: pd.DataFrame, params: FeatureParams) -> pd.Series:
    return data[params.target_col]
