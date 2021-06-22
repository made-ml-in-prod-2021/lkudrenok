from pathlib import Path

import click
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_FILENAME = 'data.csv'
TARGET_FILENAME = 'target.csv'
TRAIN_DATA_FILENAME = 'train.csv'
VAL_DATA_FILENAME = 'val.csv'


@click.command('split')
@click.option('--data-dir')
@click.option('--test-size', default=0.2)
@click.option('--random-state', default=101)
def split(data_dir: str, test_size: float, random_state: int):
    data_x = pd.read_csv(Path(data_dir, DATA_FILENAME))
    data_y = pd.read_csv(Path(data_dir, TARGET_FILENAME))
    data = pd.concat([data_x, data_y], axis=1, ignore_index=True)

    train_data, val_data = train_test_split(data, test_size=test_size, random_state=random_state)

    train_data.to_csv(Path(data_dir, TRAIN_DATA_FILENAME), index=False)
    val_data.to_csv(Path(data_dir, VAL_DATA_FILENAME), index=False)


if __name__ == '__main__':
    split()
