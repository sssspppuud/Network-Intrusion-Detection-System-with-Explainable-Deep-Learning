import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def split_dataset(df: pd.DataFrame, target_column: str, test_size: float = 0.2):
    categorical_columns = ["Label", "Attack", "Category"]
    columns_to_drop = [col for col in categorical_columns if col != target_column]
    X = df.drop(columns=columns_to_drop)
    y = df[target_column]  # Target column
    y_encoded = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )

    return X_train, X_test, y_train, y_test


from collections import Counter


def apply_smote(X_train: pd.DataFrame, y_train: pd.DataFrame):
    under = RandomUnderSampler(sampling_strategy="auto", random_state=42)
    over = SMOTE(sampling_strategy="auto", random_state=42)

    steps = [("u", under), ("o", over)]
    pipeline = Pipeline(steps=steps)
    print(f"Original training distribution: {Counter(y_train)}")

    X_train_resampled, y_train_resampled = pipeline.fit_resample(X_train, y_train)  # type: ignore
    print(f"Resampled training distribution: {Counter(y_train_resampled)}")

    return X_train_resampled, y_train_resampled
