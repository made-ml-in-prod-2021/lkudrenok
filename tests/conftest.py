"""Syntetic data and fixtures for tests."""
import pytest
import pandas as pd
import numpy as np

from ml_project.utils.param_classes import SplittingParams


TEMP_DATA_FILENAME = 'temp_heart.csv'
TEMP_DATA_PREDICT_FILENAME = 'temp_heart_without_target.csv'
TEMP_DATA_SIZE = 100

TEMP_TRAIN_CONFIG_FILENAME = 'temp_train_config.yml'
TEMP_TRAIN_CONFIG_STR = r"""
input_data_path: '{path}\temp_heart.csv'
output_model_path: '{path}\temp_model.pkl'
metric_path: '{path}\temp_metrics.json'
logs_path: '{path}\temp_logs.txt'
splitting_params:
  val_size: 0.3
  random_state: 101
train_params:
  model_type: 'LogisticRegression'
feature_params:
  categorical_features:
    - 'sex'
    - 'cp'
  numerical_features:
    - 'age'
    - 'trestbps'
  features_to_drop:
    - 'fbs'
  target_col: 'target'
"""

TEMP_PREDICT_CONFIG_FILENAME = 'temp_predict_config.yml'
TEMP_PREDICT_CONFIG_STR = r"""
input_model_path: '{path}\temp_model.pkl'
input_data_path: '{path}\temp_heart_without_target.csv'
output_data_path: '{path}\prediction.csv'
logs_path: '{path}\temp_logs.txt'
feature_params:
  categorical_features:
    - 'sex'
    - 'cp'
  numerical_features:
    - 'age'
    - 'trestbps'
"""


@pytest.fixture
def data() -> pd.DataFrame:
    data = pd.DataFrame({
        'age': np.random.uniform(low=25, high=80, size=TEMP_DATA_SIZE).astype(int),
        'sex': np.random.choice([0, 1], size=TEMP_DATA_SIZE),
        'cp': np.random.choice([0, 1, 2, 3], size=TEMP_DATA_SIZE),
        'trestbps': np.random.uniform(low=94, high=200, size=TEMP_DATA_SIZE).astype(int),
        'chol': np.random.uniform(low=126, high=564, size=TEMP_DATA_SIZE).astype(int),
        'fbs': np.random.choice([0, 1], size=TEMP_DATA_SIZE),
        'restecg': np.random.choice([0, 1, 2], size=TEMP_DATA_SIZE),
        'thalach': np.random.uniform(low=70, high=210, size=TEMP_DATA_SIZE).astype(int),
        'exang': np.random.choice([0, 1], size=TEMP_DATA_SIZE),
        'oldpeak': np.random.uniform(low=0.0, high=6.2, size=TEMP_DATA_SIZE).round(1),
        'slope': np.random.choice([0, 1, 2], size=TEMP_DATA_SIZE),
        'ca': np.random.choice([0, 1, 2, 3, 4], size=TEMP_DATA_SIZE),
        'thal': np.random.choice([0, 1, 2, 3], size=TEMP_DATA_SIZE),
        'target': np.random.choice([0, 1], size=TEMP_DATA_SIZE)
    })
    return data


@pytest.fixture
def data_file(tmpdir, data: pd.DataFrame):
    fio = tmpdir.join(TEMP_DATA_FILENAME)
    data.to_csv(fio, index=False)
    return fio


@pytest.fixture
def data_file_without_target(tmpdir, data: pd.DataFrame):
    data.drop('target', axis=1, inplace=True)
    fio = tmpdir.join(TEMP_DATA_PREDICT_FILENAME)
    data.to_csv(fio, index=False)
    return fio


@pytest.fixture
def splitting_params() -> SplittingParams:
    splitting_params = SplittingParams(
        val_size=0.1,
        random_state=101,
        shuffle=True
    )
    return splitting_params


@pytest.fixture
def train_config_file(tmpdir):
    fio = tmpdir.join(TEMP_TRAIN_CONFIG_FILENAME)
    config_str = TEMP_TRAIN_CONFIG_STR.strip().format(path=tmpdir)
    fio.write_text(config_str, encoding='utf-8')
    return fio


@pytest.fixture
def predict_config_file(tmpdir):
    fio = tmpdir.join(TEMP_PREDICT_CONFIG_FILENAME)
    config_str = TEMP_PREDICT_CONFIG_STR.strip().format(path=tmpdir)
    fio.write_text(config_str, encoding='utf-8')
    return fio
