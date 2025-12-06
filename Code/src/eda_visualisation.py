import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import numpy as np

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


def plot_attack_distribution_bar(df: pd.DataFrame, save_name: str | None = None):
    if not plot_exists(save_name):
        return
    attack_counts = df["Name"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=attack_counts.index, y=attack_counts.values, ax=ax)
    plt.ticklabel_format(style="plain", axis="y")
    plt.xlabel("Attack Name")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    save_figure(fig, save_name)


def plot_ddos_dos_bar(df: pd.DataFrame, save_name: str | None = None):
    if not plot_exists(save_name):
        category_column = df["Category"].astype(str)
        ddos_count = (category_column == "DDoS").sum()
        dos_count = (category_column == "DoS").sum()
        plot_data = pd.DataFrame(
            {"Attack Type": ["DDoS", "DoS"], "Count": [ddos_count, dos_count]}
        ).sort_values(by="Count", ascending=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x="Attack Type", y="Count", data=plot_data, ax=ax)
        plt.ticklabel_format(style="plain", axis="y")
        plt.ylabel("Count")
        save_figure(fig, save_name)


def plot_categories_no_ddos_dos_bar(df: pd.DataFrame, save_name: str | None = None):
    if not plot_exists(save_name):
        category_column = df["Category"].astype(str)
        filtered_df = df[category_column.isin(["MQTT", "Spoofing", "Recon", "Benign"])]
        category_counts = (
            filtered_df["Category"]
            .astype(str)
            .value_counts()
            .sort_values(ascending=True)
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax)
        plt.ticklabel_format(style="plain", axis="y")
        plt.xlabel("Category")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        save_figure(fig, save_name)


def plot_all_attack_categories_bar(df: pd.DataFrame, save_name: str | None = None):
    if not plot_exists(save_name):
        category_counts = df["Category"].value_counts().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax)
        plt.ticklabel_format(style="plain", axis="y")
        plt.xlabel("Category")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        save_figure(fig, save_name)


def plot_attack_benign_pie(df: pd.DataFrame, save_name: str | None = None):
    if not plot_exists(save_name):
        category_counts = df["Attack"].value_counts()

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
            f"{label} ({percentage:.2f}%)"
            for label, percentage in zip(labels, percentages)
        ]
        main_legend = plt.legend(
            legend_labels,
            title="Traffic Type",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
        )

        plt.gca().set_aspect("equal")

        save_figure(fig, save_name)


def plot_correlation_heatmap(df: pd.DataFrame, save_name: str | None = None):
    if not plot_exists(save_name):
        numeric_data = df.drop(columns=["Attack", "Category", "Name"])
        numeric_data.columns = numeric_data.columns.str.lower()

        corr = numeric_data.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        fig, ax = plt.subplots(figsize=(15, len(corr.columns) // 2))
        sns.heatmap(
            numeric_data.corr(),
            cmap="coolwarm",
            square=True,
            ax=ax,
            mask=mask,
            cbar_kws=dict(label="Correlation Coefficient"),
        )

        save_figure(fig, save_name)
