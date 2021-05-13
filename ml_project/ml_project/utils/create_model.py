import pickle
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    accuracy_score, confusion_matrix
)

from ml_project.utils.param_classes import TrainingParams


models = {
    'LogisticRegression': LogisticRegression,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'RandomForestClassifier': RandomForestClassifier
}


class Classifier:
    def __init__(self, train_params: TrainingParams = None, model: BaseEstimator = None):
        if model is not None:
            self.model = model
        elif train_params.model_params is not None:
            self.model = models[train_params.model_type](**train_params.model_params)
        else:
            self.model = models[train_params.model_type]()

    @classmethod
    def load(cls, path_to_model: str):
        with open(path_to_model, 'rb') as input_stream:
            model_data = pickle.load(input_stream)
            model = model_data['model']
            scaler_params = model_data.get('scaler_params')
            return cls(model=model), scaler_params

    def fit(self, features: pd.DataFrame, target: pd.Series):
        self.model.fit(features, target)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.model.predict(features)

    @staticmethod
    def evaluate(prediction: np.ndarray,
                 target: pd.Series,
                 output_path: str = None) -> dict[str, float]:
        conf_matrix = confusion_matrix(target, prediction).ravel()
        metrics = {
            'true neg, false pos, false neg, true pos': ' '.join(map(str, conf_matrix)),
            'f1_score': np.round(f1_score(target, prediction), 4),
            'precision': np.round(precision_score(target, prediction), 4),
            'recall': np.round(recall_score(target, prediction, zero_division=0), 4),
            'accuracy': np.round(accuracy_score(target, prediction), 4)
        }
        if output_path:
            with open(output_path, 'w') as output_stream:
                json.dump(metrics, output_stream)
        return metrics

    def save(self, output_path: str, scaler_params: dict = None):
        with open(output_path, 'wb') as output_stream:
            to_dump = {
                'model': self.model,
                'scaler_params': scaler_params
            }
            pickle.dump(to_dump, output_stream)
