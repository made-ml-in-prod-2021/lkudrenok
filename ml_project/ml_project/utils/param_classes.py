from dataclasses import dataclass

import yaml


@dataclass
class FeatureParams:
    categorical_features: [str]
    numerical_features: [str]
    target_col: str = None
    features_to_drop: [str] = None


@dataclass
class SplittingParams:
    val_size: float = 0.2
    random_state: int = 42
    shuffle: bool = False


@dataclass
class TrainingParams:
    model_type: str
    model_params: dict = None


@dataclass
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    metric_path: str
    logs_path: str
    splitting_params: SplittingParams
    train_params: TrainingParams
    feature_params: FeatureParams


@dataclass
class PredictPipelineParams:
    input_model_path: str
    input_data_path: str
    output_data_path: str
    logs_path: str
    feature_params: FeatureParams


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, 'r', encoding='utf-8') as input_stream:
        config = yaml.safe_load(input_stream)
        config['splitting_params'] = SplittingParams(**config['splitting_params'])
        config['train_params'] = TrainingParams(**config['train_params'])
        config['feature_params'] = FeatureParams(**config['feature_params'])
        return TrainingPipelineParams(**config)


def read_predict_pipeline_params(path: str) -> PredictPipelineParams:
    with open(path, 'r', encoding='utf-8') as input_stream:
        config = yaml.safe_load(input_stream)
        config['feature_params'] = FeatureParams(**config['feature_params'])
        return PredictPipelineParams(**config)
