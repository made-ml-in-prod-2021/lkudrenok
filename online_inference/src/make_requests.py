import logging
from typing import List

import requests

from src.utils.build_features import load_data
from src.app import DEFAULT_HOST, DEFAULT_PORT


REQUESTS_TO_DOCKER = True
ENDPOINT = 'predict/'
DATA_PATH = r'..\data\to_predict\heart_without_target.csv'
LOG_PATH = r'..\logs\logs_inference.txt'
NUM_SAMPLE = 10
SAMPLE_RANDOM_STATE = 101


if REQUESTS_TO_DOCKER:
    API_ADDRESS = f'http://localhost:80/{ENDPOINT}'
else:
    API_ADDRESS = f'http://{DEFAULT_HOST}:{DEFAULT_PORT}/{ENDPOINT}'


logger = logging.getLogger('main')


def setup_logger(log_filepath: str):
    """Setup logger to write into file."""

    handler = logging.FileHandler(log_filepath)
    formatter = logging.Formatter(fmt='%(asctime)s\t%(levelname)s\t%(message)s')
    handler.setFormatter(formatter)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)


def make_request(address: str,
                 columns: List[str],
                 data: List) -> requests.Response:
    logger.info('COLUMNS: %s', columns)
    logger.info('REQUEST_DATA: %s', data)
    response = requests.get(
        address,
        json={
            'columns': columns,
            'data': data
        },
    )
    logger.info('RESPONSE_CODE: %s', response.status_code)
    logger.info('RESPONSE_JSON: %s', response.json())
    return response


def run_requests():
    data = load_data(DATA_PATH).sample(NUM_SAMPLE, random_state=SAMPLE_RANDOM_STATE)
    columns = list(data.columns)
    logger.info('----- BATCH INFERENCE:')
    make_request(
        address=API_ADDRESS,
        columns=columns,
        data=data.values.tolist()
    )

    logger.info('----- ONE-BY-ONE INFERENCE:')
    for _, df_row in data.iterrows():
        make_request(
            address=API_ADDRESS,
            columns=columns,
            data=[df_row.values.tolist()]
        )


if __name__ == '__main__':
    setup_logger(LOG_PATH)
    run_requests()
