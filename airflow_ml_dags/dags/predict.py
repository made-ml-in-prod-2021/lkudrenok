from pathlib import Path

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.sensors.filesystem import FileSensor

from constants import (
    DEFAULT_ARGS, DOCKER_OPERATOR_ARGS,
    PREDICTIONS_DATA_DIR, RAW_DATA_DIR,
    PREDICT_MODEL_DIR, PREDICT_TRANSFORMER_DIR
)


with DAG(
    'predict',
    default_args=DEFAULT_ARGS,
    schedule_interval='@daily',
    start_date=days_ago(5),
    tags=['mail']
) as dag:
    data_sensor = FileSensor(
        task_id='data-sensor',
        filepath=str(Path(RAW_DATA_DIR, 'data.csv')),
        timeout=300,
        poke_interval=10,
        retries=100
    )

    predict = DockerOperator(
        image='airflow-predict',
        command=f"--model-dir {PREDICT_MODEL_DIR} "
                f"--transformer-dir {PREDICT_TRANSFORMER_DIR} "
                f"--input-data-dir {RAW_DATA_DIR} --output-data-dir {PREDICTIONS_DATA_DIR}",
        task_id='predict',
        **DOCKER_OPERATOR_ARGS
    )

    data_sensor >> predict
