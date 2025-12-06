import polars as pl
import seaborn as sns

from config.settings import DATA_ROOT
from src.dataset_management import combine_dataset_to_pq

from src.eda_visualisation import *


if __name__ == "__main__":
    sns.set_theme(style="whitegrid", context="talk", palette="bright")

    if not os.path.exists(f"{DATA_ROOT}/combined_dataset.parquet"):
        combine_dataset_to_pq(DATA_ROOT)

    df = pl.read_parquet(f"{DATA_ROOT}/combined_dataset.parquet")

    df = df.with_columns(pl.col(pl.Float64).cast(pl.Float32))
    print(f"{df.estimated_size("gb"):.2f} GB")  # 1.53 to beat

    counts = (
        df.group_by(["Category", "Attack", "Name"]).len().sort("len", descending=True)
    )
    with pl.Config(tbl_rows=-1):
        print(counts)

    # Converting to pandas df just for plotting, saves reworking entire plotting functionality
    df_pd = df.to_pandas(use_pyarrow_extension_array=True)

    # Plots before preprocessing, entire dataset
    plot_categories_no_ddos_dos_bar(df_pd, "AttackCatDistNoDosDDoS.png")
    plot_attack_distribution_bar(df_pd, "AttackDistBar.png")
    plot_ddos_dos_bar(df_pd, "DDoSDoSPlot.png")
    plot_all_attack_categories_bar(df_pd, "AllAttackCategoryDistBar.png")
    plot_attack_benign_pie(df_pd, "AttackBenignPie.png")
    plot_correlation_heatmap(df_pd.sample(n=1000000), "CorrelationHeatmap.png")
