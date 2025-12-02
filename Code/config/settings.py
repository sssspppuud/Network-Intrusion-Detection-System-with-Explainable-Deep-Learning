import os

DATA_ROOT = "data"
TRAIN_SUBSET = "train"
TEST_SUBSET = "test"

FIGURES_ROOT = "Figures"

PROCESSED_DATA_FILE = os.path.join(DATA_ROOT, "processed_dataset.parquet")


def get_figure_path(filename: str):
    return os.path.join(FIGURES_ROOT, filename)
