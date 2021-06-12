import os
from pathlib import Path
from shutil import copyfile
import pickle

import click
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TRANSFORMER_FILENAME = 'transformer.pkl'
DATA_FILENAME = 'data.csv'
TARGET_FILENAME = 'target.csv'

CATEGORICAL_FEATURES = ['cat_1', 'cat_2']
NUMERICAL_FEATURES = ['num_3', 'num_4', 'num_5', 'num_6', 'num_7', 'num_8']


def build_transformer() -> ColumnTransformer:
    transformer = ColumnTransformer([
        (
            'categorical_pipeline',
            Pipeline([('one_hot', OneHotEncoder(drop='first'))]),
            CATEGORICAL_FEATURES
        ),
        (
            'numerical_pipeline',
            Pipeline([('scale', StandardScaler())]),
            NUMERICAL_FEATURES
        )
    ])
    return transformer


@click.command('preprocess')
@click.option('--input-data-dir')
@click.option('--output-data-dir')
def preprocess(input_data_dir: str, output_data_dir: str):
    data_x = pd.read_csv(Path(input_data_dir, DATA_FILENAME))

    transformer = build_transformer()
    processed = transformer.fit_transform(data_x)

    os.makedirs(output_data_dir, exist_ok=True)

    pd.DataFrame(processed).to_csv(os.path.join(output_data_dir, DATA_FILENAME), index=False)
    copyfile(Path(input_data_dir, TARGET_FILENAME), Path(output_data_dir, TARGET_FILENAME))
    with open(Path(output_data_dir, TRANSFORMER_FILENAME), 'wb') as file_out:
        pickle.dump(transformer, file_out)


if __name__ == '__main__':
    preprocess()
