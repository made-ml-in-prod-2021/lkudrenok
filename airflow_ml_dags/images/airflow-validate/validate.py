from pathlib import Path
import pickle
import json

import click
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


MODEL_FILENAME = 'model.pkl'
VAL_DATA_FILENAME = 'val.csv'
METRICS_FILENAME = 'metrics.json'


@click.command('validate')
@click.option('--model-dir')
@click.option('--input-data-dir')
def validate(model_dir: str, input_data_dir: str):
    val_data = pd.read_csv(Path(input_data_dir, VAL_DATA_FILENAME))

    val_x = val_data.iloc[:, :-1]
    val_y = val_data.iloc[:, -1]

    with open(Path(model_dir, MODEL_FILENAME), 'rb') as in_file:
        model = pickle.load(in_file)

    prediction = model.predict(val_x)

    accuracy = accuracy_score(val_y, prediction)
    conf_matrix = confusion_matrix(val_y, prediction).ravel()
    metrics = {
        'true neg, false pos, false neg, true pos': ' '.join(map(str, conf_matrix)),
        'accuracy': round(accuracy, 4)
    }

    with open(Path(model_dir, METRICS_FILENAME), 'w') as out_file:
        json.dump(metrics, out_file, indent=4)


if __name__ == '__main__':
    validate()
