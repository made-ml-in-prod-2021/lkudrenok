from typing import List

import fastapi
import pandas as pd
from fastapi.testclient import TestClient

from src.app import app
from src.make_requests import ENDPOINT


def _make_request(columns: List[str], data: List) -> fastapi.Response:
    with TestClient(app) as client:
        response = client.get(
            ENDPOINT,
            json={
                'columns': columns,
                'data': data
            }
        )
    return response


def test_predict_batch(data: pd.DataFrame):
    batch_size = 10
    data = data.iloc[:batch_size, :-1]
    response = _make_request(
        columns=list(data.columns),
        data=data.values.tolist()
    )
    assert response.status_code == 200
    assert len(response.json()) == batch_size


def test_predict_one(data: pd.DataFrame):
    data = data.iloc[:1, :-1]
    response = _make_request(
        columns=list(data.columns),
        data=data.values.tolist()
    )
    assert response.status_code == 200
    assert response.json() in [[{'target': 0}], [{'target': 1}]]


def test_wrong_number_of_columns(data: pd.DataFrame):
    # extra columns
    data['extra_column1'] = 0
    data['extra_column2'] = 1
    response = _make_request(
        columns=list(data.columns),
        data=data.values.tolist()
    )
    assert response.status_code // 100 == 4

    # less columns than needed
    data = data.iloc[:, :6]
    response = _make_request(
        columns=list(data.columns),
        data=data.values.tolist()
    )
    assert response.status_code // 100 == 4


def test_wrong_type_of_columns(data: pd.DataFrame):
    data = data.iloc[:, :-1]
    data['sex'] = data['sex'].map({0: 'M', 1: 'F'})
    response = _make_request(
        columns=list(data.columns),
        data=data.values.tolist()
    )
    assert response.status_code // 100 == 4


def test_wrong_column_names(data: pd.DataFrame):
    data = data.iloc[:, :-1]
    columns = list(data.columns)
    columns[0] = 'wrong_column_name'
    assert columns != list(data.columns)
    response = _make_request(
        columns=columns,
        data=data.values.tolist()
    )
    assert response.status_code == 400
    assert 'column names do not match' in response.json()['detail']
