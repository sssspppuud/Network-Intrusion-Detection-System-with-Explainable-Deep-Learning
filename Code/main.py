import pandas as pd
import seaborn as sns

from src.data_loader import combine_dataset_to_pq_pyarrow, combine_dataset_to_pq_pandas
from src.visualisation import *

from config.settings import DATA_ROOT

from src.preprocessing import split_dataset, apply_smote


if __name__ == "__main__":
    """
    Class : Attack or Benign
    Category: Broad Attack category,
    Attack : Specific Attack name
    """
    sns.set_theme(style="whitegrid", context="talk", palette="bright")

    if not os.path.exists(f"{DATA_ROOT}/combined_dataset.parquet"):
        combine_dataset_to_pq_pyarrow(DATA_ROOT)

    df = pd.read_parquet(
        f"{DATA_ROOT}/combined_dataset.parquet",
        engine="pyarrow",
        dtype_backend="pyarrow",
    )

    # print(dataset[["Attack", "Category", "Name"]].value_counts())

    # print(f"{float(df.memory_usage(deep=True).sum()) / (1024 * 1024 * 1024)} GB")

    columns = df.select_dtypes(include=["float64"]).columns
    df[columns] = df[columns].astype("float32")
    print(f"{float(df.memory_usage(deep=True).sum()) / (1024 * 1024 * 1024)} GB")

    # Plots before preprocessing, entire dataset
    plot_categories_no_ddos_dos_bar(df, "AttackCatDistNoDosDDoS.png")
    plot_attack_distribution_bar(df, "AttackDistBar.png")
    plot_ddos_dos_bar(df, "DDoSDoSPlot.png")
    plot_all_attack_categories_bar(df, "AllAttackCategoryDistBar.png")
    plot_attack_benign_pie(df, "AttackBenignPie.png")
    plot_correlation_heatmap(df.sample(n=1000000), "CorrelationHeatmap.png")
