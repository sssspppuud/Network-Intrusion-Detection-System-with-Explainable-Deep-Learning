import polars as pl
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder


def balance_dataset(df: pl.DataFrame, target_column: str) -> pl.DataFrame:
    if target_column == "Attack":
        class_counts = df.group_by("Attack").len()
        minority_size = class_counts.get_column("len").min()

        minority_label = (
            class_counts.filter(pl.col("len") == minority_size)
            .get_column("Attack")
            .item()
        )

        df_minority = df.filter(pl.col("Attack") == minority_label)
        df_majority = df.filter(pl.col("Attack") != minority_label)

        df_majority_undersampled = df_majority.sample(
            n=minority_size, with_replacement=False, seed=42
        )

        df = pl.concat([df_minority, df_majority_undersampled]).sample(
            fraction=1.0, shuffle=True, seed=42
        )
        return df
    return df


def seperate_features_and_labels(
    df: pl.DataFrame, target_column: str
) -> Tuple[np.ndarray, np.ndarray]:
    valid_targets = ["Attack", "Category", "Name"]
    if target_column not in valid_targets:
        raise ValueError(f"Invalid target column {target_column}.")

    y = df.select(target_column).to_series().to_numpy()
    X = df.drop(valid_targets).to_numpy()
    return X, y


def feature_scaling(X: np.ndarray):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def label_encoding(y: np.ndarray):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded


def feature_engineering(df: pl.DataFrame) -> pl.DataFrame:
    pass


# Class imbalance needs addressing at this stage


def dimensionality_reduction(df: pl.DataFrame) -> pl.DataFrame:
    pass


def feature_selection(df: pl.DataFrame) -> pl.DataFrame:
    pass


def split_dataset(df: pl.DataFrame) -> pl.DataFrame:
    pass
