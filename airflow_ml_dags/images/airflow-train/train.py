import os
import pickle
from pathlib import Path

import click
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


TRAIN_DATA_FILENAME = 'train.csv'
MODEL_FILENAME = 'model.pkl'


@click.command('train')
@click.option('--input-data-dir')
@click.option('--output-model-dir')
def train_pipeline(input_data_dir: str, output_model_dir: str):
    train_data = pd.read_csv(Path(input_data_dir, TRAIN_DATA_FILENAME))

    train_x = train_data.iloc[:, :-1]
    train_y = train_data.iloc[:, -1]

    model = RandomForestClassifier()
    model.fit(train_x, train_y)

    os.makedirs(output_model_dir, exist_ok=True)

    with open(Path(output_model_dir, MODEL_FILENAME), 'wb') as out_file:
        pickle.dump(model, out_file)


if __name__ == '__main__':
    train_pipeline()
