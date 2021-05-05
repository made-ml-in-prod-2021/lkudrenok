import logging

import click

from ml_project import train_pipeline, predict_pipeline
from ml_project.utils.param_classes import (
    read_training_pipeline_params, read_predict_pipeline_params
)


logger = logging.getLogger('main')


def setup_logger(log_filepath: str):
    """Setup logger to write into file."""

    handler = logging.FileHandler(log_filepath)
    formatter = logging.Formatter(fmt='%(asctime)s\t%(levelname)s\t%(message)s')
    handler.setFormatter(formatter)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)


@click.command(name='train_pipeline')
@click.argument('action')
@click.argument('config_path')
def run_pipeline_command(action: str, config_path: str):
    if action == 'train':
        params = read_training_pipeline_params(config_path)
        setup_logger(params.logs_path)
        train_pipeline(params)
    elif action == 'predict':
        params = read_predict_pipeline_params(config_path)
        setup_logger(params.logs_path)
        predict_pipeline(params)


if __name__ == '__main__':
    run_pipeline_command()
