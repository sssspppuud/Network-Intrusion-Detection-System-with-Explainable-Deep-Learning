import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import os
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
