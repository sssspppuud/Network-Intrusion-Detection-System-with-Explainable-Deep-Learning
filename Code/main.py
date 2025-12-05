import pandas as pd
import seaborn as sns

from src.data_loader import combine_dataset_to_pq
from src.visualisation import *

from config.settings import (
    DATA_ROOT,
    get_figure_path,
)

from src.preprocessing import split_dataset, apply_smote


if __name__ == "__main__":
    """
    Class : Attack or Benign
    Category: Broad Attack category,
    Attack : Specific Attack name
    """
    sns.set_theme(style="whitegrid", context="talk", palette="bright")

    if not os.path.exists(f"{DATA_ROOT}/combined_dataset.parquet"):
        combine_dataset_to_pq(DATA_ROOT)

    # df = pd.read_csv(f"{DATA_ROOT}/combined_dataset.csv")
    # Plots before preprocessing, entire dataset
    # plot_categories_no_ddos_dos_bar(df, get_figure_path("AttackCatDistNoDosDDoS.png"))
    # plot_attack_distribution_bar(df, get_figure_path("AttackDistBar.png"))
    # plot_ddos_dos_bar(df, get_figure_path("DDoSDoSPlot.png"))
    # plot_all_attack_categories_bar(df, get_figure_path("AllAttackCategoryDistBar.png"))
    # plot_attack_benign_pie(df, get_figure_path("AttackBenignPie.png"))
    # plot_correlation_heatmap(df, get_figure_path("CorrelationHeatmap.png"))
    # del df

    dataset = pd.read_parquet(f"{DATA_ROOT}/combined_dataset.parquet")

    series_memory_fp_before = dataset.memory_usage(deep=True).sum()

    print(f"{float(series_memory_fp_before) / (1024 * 1024 * 1024)} GB")
