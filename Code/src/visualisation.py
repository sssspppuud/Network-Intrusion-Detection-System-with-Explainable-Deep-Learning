import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def save_figure(fig, save_path: str | None):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_attack_distribution_bar(df: pd.DataFrame, save_path: str | None = None):
    attack_counts = df["Attack"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=attack_counts.index, y=attack_counts.values, ax=ax)
    plt.ticklabel_format(style="plain", axis="y")
    plt.xlabel("Attack Name")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    save_figure(fig, save_path)


def plot_ddos_dos_bar(df: pd.DataFrame, save_path: str | None = None):
    ddos_count = len(df[df["Category"] == "DDoS"])
    dos_count = len(df[df["Category"] == "DoS"])
    plot_data = pd.DataFrame(
        {"Attack Type": ["DDoS", "DoS"], "Count": [ddos_count, dos_count]}
    ).sort_values(by="Count", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x="Attack Type", y="Count", data=plot_data, ax=ax)
    plt.ticklabel_format(style="plain", axis="y")
    plt.ylabel("Count")
    save_figure(fig, save_path)


def plot_attack_categories_no_ddos_dos_bar(
    df: pd.DataFrame, save_path: str | None = None
):
    filtered_df = df[df["Category"].isin(["MQTT", "Spoofing", "Recon", "Benign"])]
    category_counts = filtered_df["Category"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax)
    plt.ticklabel_format(style="plain", axis="y")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    save_figure(fig, save_path)


def plot_all_attack_categories_bar(df: pd.DataFrame, save_path: str | None = None):
    category_counts = df["Category"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax)
    plt.ticklabel_format(style="plain", axis="y")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    save_figure(fig, save_path)


def plot_attack_benign_pie(df: pd.DataFrame, save_path: str | None = None):
    category_counts = df["Class"].value_counts()

    total_count = category_counts.sum()
    percentages = (category_counts / total_count) * 100
    slices = category_counts.values
    labels = category_counts.index

    colours = ["#B40000", "#13A300"]

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.pie(
        slices,  # pyright: ignore[reportArgumentType]
        colors=colours,  # pyright: ignore[reportArgumentType]
        wedgeprops=dict(edgecolor="black", linewidth=1),
        textprops=dict(color="white", fontsize=10),
        startangle=90,
    )

    legend_labels = [
        f"{label} ({percentage:.2f}%)" for label, percentage in zip(labels, percentages)
    ]
    main_legend = plt.legend(
        legend_labels,
        title="Traffic Type",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )

    plt.gca().set_aspect("equal")

    save_figure(fig, save_path)


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str | None = None):
    numeric_data = df.select_dtypes(["float64", "int64"])
    numeric_data.columns = numeric_data.columns.str.lower()

    corr = numeric_data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(
        numeric_data.corr(),
        cmap="coolwarm",
        square=True,
        ax=ax,
        mask=mask,
        cbar_kws=dict(label="Correlation Coefficient"),
    )

    save_figure(fig, save_path)


def sample_group(group):
    n_samples_per_class = 230000
    n = min(len(group), n_samples_per_class)
    return group.sample(n=n, random_state=42)


# def plt_pca(df: pd.DataFrame, save_path: str | None = None):

#     df_sample = df.groupby("Class", group_keys=False).apply(sample_group)

#     numerical_columns = df.select_dtypes(["float64", "int64"]).columns

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(df_sample[numerical_columns])

#     pca = PCA(n_components=2)
#     pca_results = pca.fit_transform(X_scaled)

#     fig, ax = plt.subplots(figsize=(15, 15))
#     sns.scatterplot(
#         x=pca_results[:, 0],
#         y=pca_results[:, 1],
#         hue=df_sample["Class"],
#         alpha=0.5,
#         palette="tab10",
#         rasterized=True,
#     )
#     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
#     save_figure(fig, save_path)
