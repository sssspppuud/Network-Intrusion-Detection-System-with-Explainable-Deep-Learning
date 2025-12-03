import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import numpy as np


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
    sns.barplot(x="Attack Type", y="Count", data=plot_data)
    plt.ticklabel_format(style="plain", axis="y")
    plt.ylabel("Count")
    save_figure(fig, save_path)


def plot_attack_categories_no_ddos_dos_bar(
    df: pd.DataFrame, save_path: str | None = None
):
    filtered_df = df[df["Category"].isin(["MQTT", "Spoofing", "Recon", "Benign"])]
    category_counts = filtered_df["Category"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values)
    plt.ticklabel_format(style="plain", axis="y")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    save_figure(fig, save_path)


def plot_all_attack_categories_bar(df: pd.DataFrame, save_path: str | None = None):
    category_counts = df["Category"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values)
    plt.ticklabel_format(style="plain", axis="y")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    save_figure(fig, save_path)


def plot_attack_category_pie(df: pd.DataFrame, save_path: str | None = None):
    category_counts = df["Category"].value_counts()

    total_count = category_counts.sum()
    percentages = (category_counts / total_count) * 100
    slices = category_counts.values
    labels = category_counts.index

    cmap = plt.cm.get_cmap("Spectral")
    colours = cmap(np.linspace(0, 1, len(slices)))

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.pie(
        slices,  # pyright: ignore[reportArgumentType]
        colors=colours,  # pyright: ignore[reportArgumentType]
        wedgeprops=dict(edgecolor="black", linewidth=0.5),
        textprops=dict(color="white", fontsize=10),
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
