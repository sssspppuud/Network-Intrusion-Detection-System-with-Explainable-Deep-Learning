import polars as pl
import numpy as np
from typing import Tuple


def seperate_features_and_labels(
    df: pl.DataFrame, target_column: str
) -> Tuple[np.ndarray, np.ndarray]:
    valid_targets = ["Attack", "Category", "Name"]
    if target_column not in valid_targets:
        raise ValueError(f"Invalid target column {target_column}.")

    y = df.select(target_column).to_series().to_numpy()
    X = df.drop(valid_targets).to_numpy()
    return X, y
