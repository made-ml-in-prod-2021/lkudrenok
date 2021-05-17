import pytest
from sklearn.ensemble import RandomForestClassifier

from ml_project.utils.param_classes import TrainingParams
from ml_project.utils.create_model import Classifier


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


def test_create_model(training_params: TrainingParams):
    classifier = Classifier(training_params)
    assert isinstance(classifier.model, RandomForestClassifier)
    assert classifier.model.random_state == 101
    assert classifier.model.max_depth == 6
    assert hasattr(classifier, 'fit')
    assert hasattr(classifier, 'predict')
    assert hasattr(classifier, 'save')
