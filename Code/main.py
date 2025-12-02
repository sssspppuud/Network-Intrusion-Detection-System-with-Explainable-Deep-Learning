import pandas as pd

from src.data_loader import load_dataset
from src.visualisation import *

from config.settings import (
    DATA_ROOT,
    TRAIN_SUBSET,
    TEST_SUBSET,
    get_figure_path,
)


if __name__ == "__main__":
    train = load_dataset(DATA_ROOT, TRAIN_SUBSET)
    test = load_dataset(DATA_ROOT, TEST_SUBSET)
    df = pd.concat([train, test], ignore_index=True)

    # Show dataset attack classes and distribution
    print(df[["Category", "Attack", "Class"]].value_counts())

    plot_attack_categories_no_ddos_dos_bar(
        df, get_figure_path("AttackCategoryDistNoDosDDoS.png")
    )
    plot_attack_distribution_bar(df, get_figure_path("AttackDistBar.png"))
    plot_ddos_dos_bar(df, get_figure_path("DDoS_DoS_Plot.png"))
    plot_all_attack_categories_bar(df, get_figure_path("AllAttackCategoryDistBar.png"))
    plot_attacks_pie(df, get_figure_path("AttackNestedPie.png"))
