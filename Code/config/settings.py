import os

DATA_ROOT = "data"
TRAIN_SUBSET = "train"
TEST_SUBSET = "test"

FIGURES_ROOT = "Figures"


def get_figure_path(filename: str):
    return os.path.join(FIGURES_ROOT, filename)
