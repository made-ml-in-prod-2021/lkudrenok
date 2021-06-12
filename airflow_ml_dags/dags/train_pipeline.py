from pathlib import Path

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.sensors.filesystem import FileSensor

from constants import (
    DEFAULT_ARGS, RAW_DATA_DIR,
    PREPROCESS_DATA_DIR, MODEL_DIR, DOCKER_OPERATOR_ARGS
)


with DAG(
    'train_pipeline',
    default_args=DEFAULT_ARGS,
    schedule_interval='@weekly',
    start_date=days_ago(30),
    tags=['mail']
) as dag:
    data_sensor = FileSensor(
        task_id='data-sensor',
        filepath=str(Path(RAW_DATA_DIR, 'data.csv')),
        timeout=300,
        poke_interval=10,
        retries=100
    )

    target_sensor = FileSensor(
        task_id='target-sensor',
        filepath=str(Path(RAW_DATA_DIR, 'target.csv')),
        timeout=300,
        poke_interval=10,
        retries=100
    )

    preprocess = DockerOperator(
        image='airflow-preprocess',
        command=f'--input-data-dir {RAW_DATA_DIR} --output-data-dir {PREPROCESS_DATA_DIR}',
        task_id='preprocess',
        **DOCKER_OPERATOR_ARGS
    )

    split = DockerOperator(
        image='airflow-split',
        command=f'--data-dir {PREPROCESS_DATA_DIR}',
        task_id='split',
        **DOCKER_OPERATOR_ARGS
    )

    train = DockerOperator(
        image='airflow-train',
        command=f'--input-data-dir {PREPROCESS_DATA_DIR} --output-model-dir {MODEL_DIR}',
        task_id='train',
        **DOCKER_OPERATOR_ARGS
    )

    validate = DockerOperator(
        image='airflow-validate',
        command=f'--model-dir {MODEL_DIR} --input-data-dir {PREPROCESS_DATA_DIR}',
        task_id='validate',
        **DOCKER_OPERATOR_ARGS
    )

    [data_sensor, target_sensor] >> preprocess >> split >> train >> validate
