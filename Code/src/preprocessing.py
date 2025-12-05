import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


def split_and_label_dataset(df: pd.DataFrame, target_column: str, test_size: float):
    if target_column == "Attack":
        y = df[target_column].astype(int)
    elif target_column in ["Name", "Category"]:
        y = df[target_column].astype(str)

    X = df.drop(columns=["Name", "Category", "Attack"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


def undersample_majority(X_train: pd.DataFrame, y_train: pd.DataFrame):
    rus = RandomUnderSampler(sampling_strategy="auto", random_state=42)
    X_train_balanced, y_train_balanced = rus.fit_resample(X_train, y_train)
    return X_train_balanced, y_train_balanced
