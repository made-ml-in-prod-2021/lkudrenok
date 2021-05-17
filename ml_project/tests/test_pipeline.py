import os

import pandas as pd

from ml_project import train_pipeline, predict_pipeline
from ml_project.utils.param_classes import (
    SplittingParams, TrainingPipelineParams, PredictPipelineParams,
    read_training_pipeline_params, read_predict_pipeline_params
)

from tests.conftest import (
    data_file, data_file_without_target, TEMP_DATA_SIZE,
    train_config_file, predict_config_file
)


def test_read_training_pipeline_params(data_file, train_config_file):
    params = read_training_pipeline_params(train_config_file)
    assert isinstance(params, TrainingPipelineParams)
    assert params.input_data_path[-14:] == 'temp_heart.csv'
    assert isinstance(params.splitting_params, SplittingParams)
    assert params.train_params.model_type
    assert params.feature_params.features_to_drop == ['fbs']


def test_logging_messages(caplog, data_file, train_config_file):
    params = read_training_pipeline_params(train_config_file)
    with caplog.at_level('DEBUG'):
        train_pipeline(params)

    assert any('Initializing training pipeline' in message for message in caplog.messages)
    assert any('LogisticRegression' in message for message in caplog.messages)
    assert any('Model initialized and fitted' in message for message in caplog.messages)
    assert any('Model saved' in message for message in caplog.messages)

    for record in caplog.records:
        if 'Initializing training pipeline' in record.msg:
            assert record.levelname == 'INFO'


def test_train_pipeline(data_file, train_config_file):
    params = read_training_pipeline_params(train_config_file)
    metrics = train_pipeline(params)
    assert 'f1_score' in metrics
    assert 'accuracy' in metrics
    assert isinstance(metrics['accuracy'], float)
    assert os.path.exists(params.output_model_path)
    assert os.path.exists(params.metric_path)


def test_predict_pipeline(data_file,
                          data_file_without_target,
                          train_config_file,
                          predict_config_file):
    train_params = read_training_pipeline_params(train_config_file)
    train_pipeline(train_params)

    predict_params = read_predict_pipeline_params(predict_config_file)
    prediction = predict_pipeline(predict_params)
    assert isinstance(predict_params, PredictPipelineParams)
    assert isinstance(prediction, pd.DataFrame)
    assert prediction.shape == (TEMP_DATA_SIZE, 1)
    assert os.path.exists(predict_params.output_data_path)
