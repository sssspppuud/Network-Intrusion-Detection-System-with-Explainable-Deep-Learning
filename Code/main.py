from sklearn.preprocessing import StandardScaler, LabelEncoder

from src import *

if __name__ == "__main__":
    df = load_dataset("combined_dataset")

    # df = df.with_columns(pl.col(pl.Float64).cast(pl.Float32)) # Optimising dataset in memory usage
    # print(f"{df.estimated_size("gb"):.2f} GB")

    # counts = (
    #     df.group_by(["Attack", "Category", "Name"]).len().sort("len", descending=True)
    # )
    # with pl.Config(tbl_rows=-1):
    #     print(counts)

    # EDA Plots
    plot_categories_no_ddos_dos_bar(df, "AttackCatDistNoDosDDoS.png")
    plot_attack_distribution_bar(df, "AttackDistBar.png")
    plot_ddos_dos_bar(df, "DDoSDoSPlot.png")
    plot_all_attack_categories_bar(df, "AllAttackCategoryDistBar.png")
    plot_attack_benign_pie(df, "AttackBenignPie.png")
    plot_correlation_heatmap(df, "CorrelationHeatmap.png")
    plot_breakdown_pie(df, "Name", "AttackNamePieChart.png")

    # Data Cleaning
    df = df.drop_nulls()
    df = df.drop_nans()

    # df = balance_dataset(df, "Attack")

    # Data Transformation
    X1, y1 = seperate_features_and_labels(df, target_column="Attack")
    scaler = StandardScaler()
    X1 = scaler.fit_transform(X1)
    encoder = LabelEncoder()
    y1 = encoder.fit_transform(y1)
    plot_pca(X1, y1, encoder, "PCAPlotAttack.png")
    del X1
    del y1

    X2, y2 = seperate_features_and_labels(df, target_column="Category")
    scaler = StandardScaler()
    X2 = scaler.fit_transform(X2)
    encoder = LabelEncoder()
    y2 = encoder.fit_transform(y2)
    plot_pca(X2, y2, encoder, "PCAPlotCategory.png")
    del X2
    del y2

    X3, y3 = seperate_features_and_labels(df, target_column="Name")
    scaler = StandardScaler()
    X3 = scaler.fit_transform(X3)
    encoder = LabelEncoder()
    y3 = encoder.fit_transform(y3)
    plot_pca(X3, y3, encoder, "PCAPlotName.png")
    del X3
    del y3
