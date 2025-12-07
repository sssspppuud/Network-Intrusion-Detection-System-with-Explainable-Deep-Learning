import polars as pl
import numpy as np
import os

from sklearn.decomposition import PCA
import umap

import matplotlib.pyplot as plt
import seaborn as sns


from config.settings import FIGURES_ROOT


def plot_exists(save_name: str | None):
    if not save_name:
        return False
    path = os.path.join(FIGURES_ROOT, save_name)
    return os.path.exists(path)


def save_figure(fig, save_name: str | None):
    if not save_name:
        return

    path = os.path.join(FIGURES_ROOT, save_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_attack_distribution_bar(df: pl.DataFrame, save_name: str | None = None):
    if not plot_exists(save_name):
        attack_counts = df.group_by("Name").len().sort("len", descending=False)
        attack_counts = attack_counts.to_pandas()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=attack_counts["Name"], y=attack_counts["len"], ax=ax)
        plt.ticklabel_format(style="plain", axis="y")
        plt.xlabel("Attack Name")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        save_figure(fig, save_name)


def plot_ddos_dos_bar(df: pl.DataFrame, save_name: str | None = None):
    if not plot_exists(save_name):
        ddos_dos_counts = (
            df.filter(pl.col("Category").is_in(["DDoS", "DoS"]))
            .group_by("Category")
            .len()
            .sort("len", descending=False)
        )
        plot_data = ddos_dos_counts.to_pandas()
        plot_data.columns = ["Attack Type", "Count"]

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=plot_data["Attack Type"], y=plot_data["Count"], ax=ax)
        plt.ticklabel_format(style="plain", axis="y")
        plt.xlabel("Attack Type")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        save_figure(fig, save_name)


def plot_categories_no_ddos_dos_bar(df: pl.DataFrame, save_name: str | None = None):
    if not plot_exists(save_name):
        included_categories = ["MQTT", "Spoofing", "Recon", "Benign"]
        category_counts = (
            df.filter(pl.col("Category").is_in(included_categories))
            .group_by("Category")
            .len()
            .sort("len", descending=False)
        )
        category_counts = category_counts.to_pandas()

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=category_counts["Category"], y=category_counts["len"], ax=ax)
        plt.ticklabel_format(style="plain", axis="y")
        plt.xlabel("Category")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        save_figure(fig, save_name)


def plot_all_attack_categories_bar(df: pl.DataFrame, save_name: str | None = None):
    if not plot_exists(save_name):
        category_counts = df.group_by("Category").len().sort("len", descending=False)
        category_counts = category_counts.to_pandas()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=category_counts["Category"], y=category_counts["len"], ax=ax)
        plt.ticklabel_format(style="plain", axis="y")
        plt.xlabel("Attack Category")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        save_figure(fig, save_name)


def plot_attack_benign_pie(df: pl.DataFrame, save_name: str | None = None):
    if not plot_exists(save_name):
        attack_counts = df.group_by("Attack").len()
        attack_counts = attack_counts.to_pandas()
        attack_counts.columns = ["Attack", "Count"]

        attack_counts = attack_counts.set_index("Attack")["Count"]
        total_count = attack_counts.sum()
        percentages = (attack_counts / total_count) * 100
        slices = attack_counts.values
        labels = attack_counts.index

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
            f"{label} ({percentage:.2f}%)"
            for label, percentage in zip(labels, percentages)
        ]
        plt.legend(
            legend_labels,
            title="Traffic Type",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
        )

        plt.gca().set_aspect("equal")

        save_figure(fig, save_name)


def plot_correlation_heatmap(df: pl.DataFrame, save_name: str | None = None):
    if not plot_exists(save_name):
        numeric_data = df.select(
            pl.exclude("Attack", "Category", "Name").name.map(lambda name: name.lower())
        )

        std_devs = numeric_data.select(pl.all().std())
        zer_var_cols = [col for col in std_devs.columns if std_devs[col][0] == 0]
        numeric_data = numeric_data.drop(zer_var_cols)

        corr = numeric_data.corr()
        corr_labels = corr.columns
        corr = corr.to_numpy()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        fig, ax = plt.subplots(figsize=(15, len(corr_labels) // 2))
        sns.heatmap(
            corr,
            cmap="coolwarm",
            square=True,
            ax=ax,
            mask=mask,
            cbar_kws=dict(label="Correlation Coefficient"),
            xticklabels=corr_labels,
            yticklabels=corr_labels,
        )

        save_figure(fig, save_name)


def plot_pca(X: np.ndarray, y, save_name: str | None = None):
    if not plot_exists(save_name):
        temp_df = pl.DataFrame(X).with_columns(label=pl.Series(y))
        sampled_df = temp_df.sample(n=100_000, seed=42)
        X = sampled_df.drop("label").to_numpy()
        y = sampled_df["label"].to_numpy()

        pca = PCA(n_components=2, svd_solver="randomized", random_state=42)
        X_pca = pca.fit_transform(X)

        df = pl.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "label": y})
        df = df.to_pandas()

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.scatterplot(
            data=df, x="PC1", y="PC2", hue="label", palette="Spectral", alpha=0.6, s=10
        )
        ax.legend(loc="upper right")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        plt.legend(title="Class")
        save_figure(fig, save_name)


def plot_umap(X: np.ndarray, y, save_name: str | None = None):
    if not plot_exists(save_name):
        temp_df = pl.DataFrame(X).with_columns(label=pl.Series(y))
        sampled_df = temp_df.sample(n=10_000, seed=42)
        X = sampled_df.drop("label").to_numpy()
        y = sampled_df["label"].to_numpy()

        reducer = umap.UMAP(n_components=2, n_neighbors=50, min_dist=0.4, n_jobs=-1)
        X_umap = reducer.fit_transform(X)

        df = pl.DataFrame({"UMAP1": X_umap[:, 0], "UMAP2": X_umap[:, 1], "label": y})  # type: ignore
        df = df.to_pandas()

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.scatterplot(
            data=df,
            x="UMAP1",
            y="UMAP2",
            hue="label",
            palette="Spectral",
            alpha=0.6,
            s=10,
        )
        ax.legend(loc="upper right")
        ax.set_xlabel("UMAP Component 1")
        ax.set_ylabel("UMAP Component 2")
        plt.legend(title="Class")
        save_figure(fig, save_name)
