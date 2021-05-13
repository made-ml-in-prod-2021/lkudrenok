import pandas as pd
from sklearn.model_selection import train_test_split

from ml_project.utils.param_classes import SplittingParams


def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def split_train_val_data(data: pd.DataFrame,
                         params: SplittingParams) -> (pd.DataFrame, pd.DataFrame):
    train_data, val_data = train_test_split(
        data,
        test_size=params.val_size,
        random_state=params.random_state,
        shuffle=params.shuffle
    )
    return train_data, val_data
