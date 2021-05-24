import os

import pytest
from sklearn.utils.validation import check_is_fitted
from sklearn.compose import ColumnTransformer

from src.utils.param_classes import TrainingParams, read_training_pipeline_params
from src.utils.create_model import Classifier
from src.utils.build_features import build_transformer, load_data


@pytest.fixture
def training_params() -> TrainingParams:
    training_params = TrainingParams(
        model_type='RandomForestClassifier',
        model_params={
            'max_depth': 6,
            'random_state': 101
        }
    )
    return training_params


def test_save_and_load_model(data_file, train_config_file):
    params = read_training_pipeline_params(train_config_file)
    train_data = load_data(data_file)
    transformer = build_transformer(params.feature_params)
    transformer.fit(train_data)
    model = Classifier(params.train_params)
    model.save(params.output_model_path, transformer)
    assert os.path.exists(params.output_model_path)

    loaded_model, loaded_transformer = model.load(params.output_model_path)
    assert isinstance(loaded_model, Classifier)
    assert isinstance(loaded_transformer, ColumnTransformer)
    try:
        check_is_fitted(loaded_transformer)
    except:
        pytest.fail('loaded transformer is not fitted')
