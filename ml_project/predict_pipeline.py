import logging

import pandas as pd

from ml_project.utils.make_dataset import load_data
from ml_project.utils.build_features import build_transformer, make_features
from ml_project.utils.create_model import Classifier
from ml_project.utils.param_classes import PredictPipelineParams


logger = logging.getLogger('main')


def predict_pipeline(predict_pipeline_params: PredictPipelineParams) -> pd.DataFrame:
    """
    Pipeline to load test data; extract features; load model from file;
    get and save prediction.
    """

    logger.info('Initializing prediction pipeline...')
    logger.info('Predict pipeline params: %s', predict_pipeline_params)

    data = load_data(predict_pipeline_params.input_data_path)
    logger.info('Loaded data with shape: %s', data.shape)

    classifier, scaler_params = Classifier.load(predict_pipeline_params.input_model_path)
    logger.info('Model loaded from %s', predict_pipeline_params.input_model_path)
    logger.info('Loaded scaler_params: %s', scaler_params)

    transformer = build_transformer(
        predict_pipeline_params.feature_params,
        scaler_params=scaler_params
    )
    transformer.fit(data)
    features = make_features(transformer, data)
    logger.info('Features shape: %s', features.shape)

    prediction = classifier.predict(features)
    logger.info('Got prediction')

    prediction = pd.DataFrame(prediction)
    prediction.to_csv(predict_pipeline_params.output_data_path, index=False)
    logger.info('Prediction saved into %s', predict_pipeline_params.output_data_path)

    return prediction
