import os
from pathlib import Path
import pickle

import click
import pandas as pd


DATA_FILENAME = 'data.csv'
PREDICTION_FILENAME = 'predictions.csv'
TRANSFORMER_FILENAME = 'transformer.pkl'
MODEL_FILENAME = 'model.pkl'


@click.command('predict')
@click.option('--model-dir')
@click.option('--transformer-dir')
@click.option('--input-data-dir')
@click.option('--output-data-dir')
def predict(model_dir: str, transformer_dir: str, input_data_dir: str, output_data_dir: str):
    data = pd.read_csv(Path(input_data_dir, DATA_FILENAME))

    with open(Path(transformer_dir, TRANSFORMER_FILENAME), 'rb') as file_in:
        transformer = pickle.load(file_in)

    with open(Path(model_dir, MODEL_FILENAME), 'rb') as file_in:
        model = pickle.load(file_in)

    processed_data = transformer.transform(data)
    predictions = model.predict(processed_data)
    predictions = pd.DataFrame(predictions, columns=['target'])

    os.makedirs(output_data_dir, exist_ok=True)
    predictions.to_csv(Path(output_data_dir, PREDICTION_FILENAME), index=False)


if __name__ == '__main__':
    predict()
