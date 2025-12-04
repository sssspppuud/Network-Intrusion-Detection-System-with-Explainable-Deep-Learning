import os

DATA_ROOT = "data"

FIGURES_ROOT = "out"


def get_figure_path(filename: str):
    return os.path.join(FIGURES_ROOT, filename)
