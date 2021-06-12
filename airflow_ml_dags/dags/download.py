from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from constants import DEFAULT_ARGS, RAW_DATA_DIR, DOCKER_OPERATOR_ARGS


with DAG(
    'download',
    default_args=DEFAULT_ARGS,
    schedule_interval='@daily',
    start_date=days_ago(30),
    tags=['mail']
) as dag:
    download = DockerOperator(
        image='airflow-download',
        command=f'--output-data-dir {RAW_DATA_DIR}',
        task_id='download',
        **DOCKER_OPERATOR_ARGS
    )

    download
