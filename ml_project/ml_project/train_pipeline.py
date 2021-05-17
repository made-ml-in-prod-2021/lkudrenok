import logging

from ml_project.utils.param_classes import TrainingPipelineParams
from ml_project.utils.create_model import Classifier
from ml_project.utils.make_dataset import load_data, split_train_val_data
from ml_project.utils.build_features import build_transformer, make_features, extract_target


logger = logging.getLogger('main')


def train_pipeline(training_pipeline_params: TrainingPipelineParams) -> dict[str, float]:
    """
    Pipeline to load and split data; extract features;
    create, fit, evaluate and save model.
    """

    logger.info('Initializing training pipeline...')
    logger.info('Training pipeline params: %s', training_pipeline_params)

    data = load_data(training_pipeline_params.input_data_path)
    logger.info('Loaded data with shape: %s', data.shape)

    train_data, val_data = split_train_val_data(data, training_pipeline_params.splitting_params)
    logger.info(
        'Data splitted into train with shape %s and validation with shape %s',
        train_data.shape, val_data.shape
    )

    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(train_data)
    train_features = make_features(transformer, train_data)
    train_target = extract_target(train_data, training_pipeline_params.feature_params)
    logger.info('Train features shape: %s', train_features.shape)

    model = Classifier(training_pipeline_params.train_params)
    model.fit(train_features, train_target)
    logger.info('Model initialized and fitted')

    val_features = make_features(transformer, val_data)
    val_target = extract_target(val_data, training_pipeline_params.feature_params)
    prediction = model.predict(val_features)
    metrics = model.evaluate(prediction, val_target, training_pipeline_params.metric_path)
    logger.info('Model evaluated; metrics: %s', metrics)

    scaler_params = transformer.export_scaler_params()
    model.save(training_pipeline_params.output_model_path, scaler_params=scaler_params)
    logger.info('Model saved into %s', training_pipeline_params.output_model_path)

    return metrics
