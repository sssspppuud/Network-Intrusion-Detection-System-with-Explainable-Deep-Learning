import pandas as pd
import seaborn as sns

from src.data_loader import load_dataset
from src.visualisation import *

from config.settings import (
    DATA_ROOT,
    TRAIN_SUBSET,
    TEST_SUBSET,
    PROCESSED_DATA_FILE,
    get_figure_path,
)


def load_or_generate_dataset(data_root: str, processed_file: str) -> pd.DataFrame:
    if os.path.exists(processed_file):
        return pd.read_parquet(processed_file)

    train = load_dataset(DATA_ROOT, TRAIN_SUBSET)
    test = load_dataset(DATA_ROOT, TEST_SUBSET)
    df = pd.concat([train, test], ignore_index=True)

    os.makedirs(os.path.dirname(processed_file), exist_ok=True)
    df.to_parquet(processed_file, index=False)

    return df


if __name__ == "__main__":
    sns.set_theme(style="whitegrid", context="talk", palette="pastel")

    df = load_or_generate_dataset(DATA_ROOT, PROCESSED_DATA_FILE)
    # Show dataset attack classes and distribution
    print(df[["Category", "Attack", "Class"]].value_counts())

    plot_attack_categories_no_ddos_dos_bar(
        df, get_figure_path("AttackCategoryDistNoDosDDoS.png")
    )
    plot_attack_distribution_bar(df, get_figure_path("AttackDistBar.png"))
    plot_ddos_dos_bar(df, get_figure_path("DDoS_DoS_Plot.png"))
    plot_all_attack_categories_bar(df, get_figure_path("AllAttackCategoryDistBar.png"))
    plot_attack_category_pie(df, get_figure_path("AttackCategoryPie.png"))
