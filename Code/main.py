import pandas as pd
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import StandardScaler

from config.settings import DATA_ROOT
from Code.src.dataset_management import (
    combine_dataset_to_pq_pyarrow,
    combine_dataset_to_pq_pandas,
)
from Code.src.eda_visualisation import *
from Code.src.features import (
    split_and_label_dataset,
    undersample_majority,
)


if __name__ == "__main__":
    sns.set_theme(style="whitegrid", context="talk", palette="bright")

    if not os.path.exists(f"{DATA_ROOT}/combined_dataset.parquet"):
        combine_dataset_to_pq_pyarrow(DATA_ROOT)

    df = pd.read_parquet(
        f"{DATA_ROOT}/combined_dataset.parquet",
        engine="pyarrow",
        dtype_backend="pyarrow",
    )

    columns = df.select_dtypes(include=["float64"]).columns
    df[columns] = df[columns].astype("float32")

    # Plots before preprocessing, entire dataset
    # plot_categories_no_ddos_dos_bar(df, "AttackCatDistNoDosDDoS.png")
    # plot_attack_distribution_bar(df, "AttackDistBar.png")
    # plot_ddos_dos_bar(df, "DDoSDoSPlot.png")
    # plot_all_attack_categories_bar(df, "AllAttackCategoryDistBar.png")
    # plot_attack_benign_pie(df, "AttackBenignPie.png")
    # plot_correlation_heatmap(df.sample(n=1000000), "CorrelationHeatmap.png")

    X_train, X_test, y_train, y_test = split_and_label_dataset(
        df, target_column="Attack", test_size=0.2
    )

    X_train_balanced, y_train_balanced = undersample_majority(X_train, y_train)
    # print("Original training set class distribution:", Counter(y_train))
    # print("Balanced training set class distribution:", Counter(y_train_balanced))

    plot_pca_attack_or_benign(X_train_balanced, y_train_balanced, "PCAPlot.png")
