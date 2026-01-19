import polars as pl
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import umap

from config.settings import FIGURES_ROOT

sns.set_theme(style="darkgrid", context="paper", palette="bright")
matplotlib.use("Agg")


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
        attack_counts = df.group_by("Attack").len().sort("len", descending=False)
        attack_counts = attack_counts.to_pandas()
        attack_counts.columns = ["Attack", "Count"]

        attack_counts = attack_counts.set_index("Attack")["Count"]
        total_count = attack_counts.sum()
        percentages = (attack_counts / total_count) * 100
        slices = attack_counts.values
        labels = attack_counts.index

        colours = sns.color_palette("tab20c", len(labels))

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
        ax.legend(
            legend_labels,
            title="Malicious Traffic?",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
        )

        plt.gca().set_aspect("equal")

        save_figure(fig, save_name)


def plot_breakdown_pie(
    df: pl.DataFrame, target_column: str, save_name: str | None = None
):
    if not plot_exists(save_name):
        attack_counts = df.group_by(target_column).len().sort(target_column)
        attack_counts = attack_counts.to_pandas()
        attack_counts.columns = [target_column, "Count"]

        attack_counts = attack_counts.set_index(target_column)["Count"]
        total_count = attack_counts.sum()
        percentages = (attack_counts / total_count) * 100
        slices = attack_counts.values
        labels = attack_counts.index

        fig, ax = plt.subplots(figsize=(10, 10))
        colours = sns.color_palette("tab20", len(labels))
        plt.pie(
            slices,  # pyright: ignore[reportArgumentType]
            wedgeprops=dict(edgecolor="black", linewidth=1),
            textprops=dict(color="white", fontsize=10),
            startangle=90,
            colors=colours,
        )

        legend_labels = [
            f"{label} ({percentage:.2f}%)"
            for label, percentage in zip(labels, percentages)
        ]
        ax.legend(
            legend_labels,
            title=f"Attack {target_column}",
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


def plot_pca(
    X: np.ndarray, y, label_encoder: LabelEncoder, save_name: str | None = None
):
    if not plot_exists(save_name):
        pca = PCA(n_components=2, svd_solver="randomized", random_state=42)
        X_pca = pca.fit_transform(X)

        fig, ax = plt.subplots(figsize=(12, 10))
        original_labels = label_encoder.classes_
        unique_classes = np.unique(y)
        cmap = plt.cm.get_cmap("tab10", len(unique_classes))

        ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=y,
            cmap=cmap,
            alpha=0.6,
            s=10,
            rasterized=True,
        )

        legend_handles = []
        legend_labels = []

        for i, class_label in enumerate(unique_classes):
            colour = cmap(i)
            handle = ax.scatter([], [], c=[colour], label=class_label)
            legend_handles.append(handle)
            legend_labels.append(class_label)

        ax.legend(
            handles=legend_handles,
            labels=original_labels.tolist(),
            title="Attack Name",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
        )
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        save_figure(fig, save_name)


def plot_tsne(
    X: np.ndarray, y, label_encoder: LabelEncoder, save_name: str | None = None
):
    if not plot_exists(save_name):
        tsne = TSNE(
            n_components=2,
            perplexity=40,
            learning_rate=100,
            random_state=42,
            n_jobs=-1,
            verbose=1,
            init="pca",
        )
        X_tsne = tsne.fit_transform(X)

        fig, ax = plt.subplots(figsize=(12, 10))
        original_labels = label_encoder.classes_
        unique_classes = np.unique(y)
        cmap = plt.cm.get_cmap("tab10", len(unique_classes))

        ax.scatter(
            X_tsne[:, 0],
            X_tsne[:, 1],
            c=y,
            cmap=cmap,
            alpha=0.6,
            s=10,
            rasterized=True,
        )

        legend_handles = []
        legend_labels = []

        for i, class_label in enumerate(unique_classes):
            colour = cmap(i)
            handle = ax.scatter([], [], c=[colour], label=class_label)
            legend_handles.append(handle)
            legend_labels.append(class_label)

        ax.legend(
            handles=legend_handles,
            labels=original_labels.tolist(),
            title="Attack Name",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
        )
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        save_figure(fig, save_name)


def plot_umap(
    X: np.ndarray, y, label_encoder: LabelEncoder, save_name: str | None = None
):
    if not plot_exists(save_name):
        # pca_reducer = PCA(n_components=20)
        # X = pca_reducer.fit_transform(X)
        reducer = umap.UMAP(
            n_components=2,
            n_jobs=-1,
            n_neighbors=100,
            min_dist=0.5,
            set_op_mix_ratio=0.9,
            verbose=True,
        )
        X_umap = reducer.fit_transform(X)

        fig, ax = plt.subplots(figsize=(12, 10))
        original_labels = label_encoder.classes_
        unique_classes = np.unique(y)
        cmap = plt.cm.get_cmap("Spectral", len(unique_classes))

        ax.scatter(
            X_umap[:, 0],  # type: ignore
            X_umap[:, 1],  # type: ignore
            c=y,
            cmap=cmap,
            alpha=0.6,
            s=10,
            rasterized=True,
        )

        legend_handles = []
        legend_labels = []

        for i, class_label in enumerate(unique_classes):
            colour = cmap(i)
            handle = ax.scatter([], [], c=[colour], label=class_label)
            legend_handles.append(handle)
            legend_labels.append(class_label)

        ax.legend(
            handles=legend_handles,
            labels=original_labels.tolist(),
            title="Attack Name",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
        )
        ax.set_xlabel("UMAP Component 1")
        ax.set_ylabel("UMAP Component 2")
        save_figure(fig, save_name)
