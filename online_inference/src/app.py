import os
from typing import List, Union, Optional

import click
import pandas as pd
from sklearn.compose import ColumnTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from src.utils.build_features import make_features
from src.utils.create_model import Classifier


DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8000
PATH_TO_MODEL = os.getenv('PATH_TO_MODEL')

app = FastAPI()
model: Optional[Classifier] = None
transformer: Optional[ColumnTransformer] = None


class InputModel(BaseModel):
    columns: List[str]
    data: List[List[Union[float, int]]]


class ResponseModel(BaseModel):
    target: int


def validate_columns(request_columns: List[str]) -> bool:
    global transformer
    categorical = transformer.transformers_[0][2]
    numerical = transformer.transformers_[1][2]
    transformer_columns = categorical + numerical
    return set(request_columns) == set(transformer_columns)


def make_prediction(columns: List[str],
                    data: List[List[Union[float, int]]]) -> List[ResponseModel]:
    global model, transformer
    data = pd.DataFrame(data, columns=columns)

    if not validate_columns(columns):
        raise HTTPException(
            status_code=400,
            detail='Given column names do not match with transformer column names'
        )

    features = make_features(transformer, data, with_target=False)
    prediction = model.predict(features)
    return [ResponseModel(target=x) for x in prediction]


@app.on_event('startup')
def load_model():
    assert PATH_TO_MODEL is not None
    global model, transformer
    model, transformer = Classifier.load(PATH_TO_MODEL)
    assert model is not None
    assert transformer is not None


@app.get('/')
def root():
    return {'message': 'ML project inference running...'}


@app.get('/predict/', response_model=List[ResponseModel])
def predict(request: InputModel):
    return make_prediction(request.columns, request.data)


@click.command(name='run_service')
@click.argument('host', required=False, type=str)
@click.argument('port', required=False, type=int)
def run_service(host: str = None, port: int = None):
    host = host or DEFAULT_HOST
    port = port or DEFAULT_PORT
    uvicorn.run(
        'src.app:app',
        host=host, port=port,
        reload=True, use_colors=False
    )


if __name__ == '__main__':
    run_service()
