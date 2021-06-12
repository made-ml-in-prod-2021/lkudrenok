from datetime import timedelta

from airflow.models import Variable


HOST_DATA_DIR = Variable.get('data_dir', default_var='')
PREDICT_MODEL_DIR = Variable.get('model_dir', default_var='')
PREDICT_TRANSFORMER_DIR = Variable.get('transformer_dir', default_var='')

RAW_DATA_DIR = '/data/raw/{{ ds }}'
PREPROCESS_DATA_DIR = '/data/processed/{{ ds }}'
PREDICTIONS_DATA_DIR = '/data/predictions/{{ ds }}'
MODEL_DIR = '/data/model/{{ ds }}'

DEFAULT_ARGS = {
    'owner': 'airflow',
    'email': ['airflow@example.com'],
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(seconds=30),
}

DOCKER_OPERATOR_ARGS = {
    'network_mode': 'bridge',
    'do_xcom_push': False,
    'auto_remove': True,
    'volumes': [f'{HOST_DATA_DIR}:/data']
}
