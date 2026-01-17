from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
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

    # EDA Plots
    plot_categories_no_ddos_dos_bar(df, "AttackCatDistNoDosDDoS.png")
    plot_attack_distribution_bar(df, "AttackDistBar.png")
    plot_ddos_dos_bar(df, "DDoSDoSPlot.png")
    plot_all_attack_categories_bar(df, "AllAttackCategoryDistBar.png")
    plot_correlation_heatmap(df, "CorrelationHeatmap.png")
    plot_attack_benign_pie(df, "AttackBenignPie.png")
    plot_breakdown_pie(df, "Category", "AttackCategoryPieChart.png")
    plot_breakdown_pie(df, "Name", "AttackNamePieChart.png")

    df = df.drop(["Name", "Attack"])
    X = df.drop("Category").to_numpy()
    y = df.select("Category").to_series().to_numpy()

    sampling_strategy = {
        "DDoS": 300_000,
        "DoS": 300_000,
    }
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X, y)

    df_resampled = pl.DataFrame(X_resampled, schema=df.drop("Category").columns)
    df_resampled = df_resampled.with_columns(pl.Series("Category", y_resampled))
    plot_breakdown_pie(df_resampled, "Category", "AttackCategoryPieChart_Resample.png")

    counts = df_resampled.group_by("Category").len().sort("len", descending=True)
    with pl.Config(tbl_rows=-1):
        print(counts)

    del df
    # Removing redundant features
    #

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_resampled)
    print(f"Encoded Labels: {encoder.classes_}")

    # ------------------
    # Extra EDA plots
    _, X_sample, _, y_sample = train_test_split(
        X_scaled, y_encoded, test_size=0.8, random_state=42, stratify=y_resampled
    )
    plot_pca(X_sample, y_sample, encoder, "PCAPlotAttack.png")
    _, X_sample, _, y_sample = train_test_split(
        X_scaled, y_encoded, test_size=0.02, random_state=42, stratify=y_resampled
    )
    plot_umap(X_sample, y_sample, encoder, "UMAPPlotAttack.png")
    plot_tsne(X_sample, y_sample, encoder, "TSNEPlot.png")

    # ----------------

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=0, stratify=y_encoded
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=0, stratify=y_train_val
    )

    print(f"Training shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Testing shape: {X_test.shape}")
