"""
docker build -t lkudrenok/download0 .
docker run --rm -v D:/TEMP/tmp:/data lkudrenok/download0 --output-data-dir /data
"""
import os
from pathlib import Path

import click
import numpy as np
from sklearn.datasets import make_classification


@click.command('download')
@click.option('--output-data-dir', required=True)
@click.option('--n-samples', default=1000)
def download(output_data_dir: str, n_samples: int):
    data_x, data_y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=6,
        n_classes=2
    )
    data_x[:, 0] = np.random.randint(0, 3, size=data_x.shape[0])
    data_x[:, 1] = np.where(data_x[:, 1] > 0, 1, 0)
    data_x[:, 2] *= np.random.randint(90, 100)
    data_x[:, 3] *= np.random.randint(16, 20)
    data_x[:, 4][data_x[:, 4] < 0] = 0
    data_x[:, 5] += 5

    os.makedirs(output_data_dir, exist_ok=True)

    columns = ['cat_1', 'cat_2', 'num_3', 'num_4', 'num_5', 'num_6', 'num_7', 'num_8']

    np.savetxt(
        fname=Path(output_data_dir, 'data.csv'),
        X=data_x,
        fmt='%.4f',
        delimiter=',',
        header=','.join(columns),
        comments=''
    )
    np.savetxt(
        fname=Path(output_data_dir, 'target.csv'),
        X=data_y,
        fmt='%d',
        delimiter=',',
        header='target',
        comments=''
    )


if __name__ == '__main__':
    download()
