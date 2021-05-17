from dataclasses import dataclass
from typing import List

import yaml
from marshmallow_dataclass import class_schema


@dataclass
class FeatureParams:
    categorical_features: List[str]
    numerical_features: List[str]
    target_col: str = None
    features_to_drop: List[str] = None

        
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


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, 'r', encoding='utf-8') as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
