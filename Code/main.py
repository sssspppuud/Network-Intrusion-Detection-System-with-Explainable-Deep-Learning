import polars as pl
import seaborn as sns
import matplotlib

from sklearn.preprocessing import StandardScaler, LabelEncoder

from src import *

if __name__ == "__main__":
    # Loading dataset
    df = load_dataset("combined_dataset")

    # Optimising dataset in memory usage
    df = df.with_columns(pl.col(pl.Float64).cast(pl.Float32))
    print(f"{df.estimated_size("gb"):.2f} GB")  # 1.53 to beat

    # Exploratory Dataset Analysis
    sns.set_theme(style="whitegrid", context="paper", palette="bright")
    matplotlib.use("Agg")

    counts = (
        df.group_by(["Attack", "Category", "Name"]).len().sort("len", descending=True)
    )
    with pl.Config(tbl_rows=-1):
        print(counts)

    # EDA Plots
    plot_categories_no_ddos_dos_bar(df, "AttackCatDistNoDosDDoS.png")
    plot_attack_distribution_bar(df, "AttackDistBar.png")
    plot_ddos_dos_bar(df, "DDoSDoSPlot.png")
    plot_all_attack_categories_bar(df, "AllAttackCategoryDistBar.png")
    plot_attack_benign_pie(df, "AttackBenignPie.png")
    plot_correlation_heatmap(df, "CorrelationHeatmap.png")

    # Data Cleaning
    df = df.drop_nulls()
    df = df.drop_nans()

    df = balance_dataset(df, "Attack")

    # Data Transformation
    X, y = seperate_features_and_labels(df, target_column="Attack")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # plot_pca(X, y, "PCAPlot.png")
    # plot_umap(X, y, "UMAPPlot.png")
