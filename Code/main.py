from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import polars as pl

from src import *

if __name__ == "__main__":
    df = load_dataset("combined_dataset")

    df = df.with_columns(
        pl.col(pl.Float64).cast(pl.Float32)
    )  # Optimising dataset in memory usage
    print(f"{df.estimated_size("gb"):.2f} GB")

    counts = (
        df.group_by(["Attack", "Category", "Name"]).len().sort("len", descending=True)
    )
    with pl.Config(tbl_rows=-1):
        print(counts)
    counts.write_csv("./out/Dataset_Counts.csv")

    # EDA Plots
    plot_categories_no_ddos_dos_bar(df, "AttackCatDistNoDosDDoS.png")
    plot_attack_distribution_bar(df, "AttackDistBar.png")
    plot_ddos_dos_bar(df, "DDoSDoSPlot.png")
    plot_all_attack_categories_bar(df, "AllAttackCategoryDistBar.png")
    plot_correlation_heatmap(df, "CorrelationHeatmap.png")
    plot_attack_benign_pie(df, "AttackBenignPie.png")
    plot_breakdown_pie(df, "Category", "AttackCategoryPieChart.png")
    plot_breakdown_pie(df, "Name", "AttackNamePieChart.png")

    X, y = seperate_features_and_labels(df, target_column="Category")
    del df

    sampling_strategy = {
        "DDoS": 300_000,
        "DoS": 300_000,
        # "MQTT": 150_000,
        # "Benign": 150_000,
    }

    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=0)
    X, y = rus.fit_resample(X, y)

    df = pl.DataFrame(X)
    df = df.with_columns(pl.Series("Category", y))
    plot_breakdown_pie(df, "Category", "AttackCategoryPieChart_Resample.png")
    counts = df.group_by("Category").len().sort("len", descending=True)
    with pl.Config(tbl_rows=-1):
        print(counts)

    del df

    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # encoder = LabelEncoder()
    # y = encoder.fit_transform(y)
    # _, X_sample, _, y_sample = train_test_split(
    #     X, y, test_size=0.8, random_state=42, stratify=y
    # )
    # plot_pca(X_sample, y_sample, encoder, "PCAPlotAttack.png")

    # _, X_sample, _, y_sample = train_test_split(
    #     X, y, test_size=0.02, random_state=42, stratify=y
    # )
    # plot_umap(X_sample, y_sample, encoder, "UMAPPlotAttack.png")
    # plot_tsne(X_sample, y_sample, encoder, "TSNEPlot.png")
    # print(X.shape[0], len(y))
