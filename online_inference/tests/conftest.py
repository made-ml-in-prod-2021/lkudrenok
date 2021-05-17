"""Syntetic data and fixtures for tests."""
import pytest
import pandas as pd
import numpy as np


TEMP_DATA_FILENAME = 'temp_heart.csv'
TEMP_DATA_SIZE = 100

CONFIG_PATH = 'tests/resources/train_config.yml'
TEMP_TRAIN_CONFIG_FILENAME = 'temp_train_config.yml'


@pytest.fixture
def data() -> pd.DataFrame:
    data = pd.DataFrame({
        'age': np.random.randint(low=25, high=80, size=TEMP_DATA_SIZE),
        'sex': np.random.choice([0, 1], size=TEMP_DATA_SIZE),
        'cp': np.random.choice([0, 1, 2, 3], size=TEMP_DATA_SIZE),
        'trestbps': np.random.randint(low=94, high=200, size=TEMP_DATA_SIZE),
        'chol': np.random.randint(low=126, high=564, size=TEMP_DATA_SIZE),
        'fbs': np.random.choice([0, 1], size=TEMP_DATA_SIZE),
        'restecg': np.random.choice([0, 1, 2], size=TEMP_DATA_SIZE),
        'thalach': np.random.randint(low=70, high=210, size=TEMP_DATA_SIZE),
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
def train_config_file(tmpdir):
    fio = tmpdir.join(TEMP_TRAIN_CONFIG_FILENAME)
    with open(CONFIG_PATH) as input_stream:
        config_str = input_stream.read().strip().format(path=tmpdir)
    fio.write_text(config_str, encoding='utf-8')
    return fio
