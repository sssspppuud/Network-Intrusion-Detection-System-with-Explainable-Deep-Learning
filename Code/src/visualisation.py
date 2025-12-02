import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import pandas as pd
import seaborn as sns


def save_figure(fig, save_path: str | None):
    if save_path:
        plt.figure(fig.number)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_attack_distribution_bar(df: pd.DataFrame, save_path: str | None = None):
    attack_counts = df["Attack"].value_counts().sort_values(ascending=True)
    fig = plt.figure(figsize=(12, 6))
    sns.barplot(x=attack_counts.index, y=attack_counts.values)
    plt.ticklabel_format(style="plain", axis="y")
    plt.title("Distribution of Attacks")
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

    fig = plt.figure(figsize=(12, 6))
    sns.barplot(x="Attack Type", y="Count", data=plot_data)
    plt.ticklabel_format(style="plain", axis="y")
    plt.title("Count of DDoS vs. DoS Traffic")
    plt.ylabel("Count")
    save_figure(fig, save_path)


def plot_attack_categories_no_ddos_dos_bar(
    df: pd.DataFrame, save_path: str | None = None
):
    filtered_df = df[df["Category"].isin(["MQTT", "Spoofing", "Recon", "Benign"])]
    category_counts = filtered_df["Category"].value_counts().sort_values(ascending=True)
    fig = plt.figure(figsize=(12, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values)
    plt.ticklabel_format(style="plain", axis="y")
    plt.title("Distribution of Attack Categories (Excluding DDoS and DoS)")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    save_figure(fig, save_path)


def plot_all_attack_categories_bar(df: pd.DataFrame, save_path: str | None = None):
    category_counts = df["Category"].value_counts().sort_values(ascending=True)
    fig = plt.figure(figsize=(12, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values)
    plt.ticklabel_format(style="plain", axis="y")
    plt.title("Distribution of Attack Categories")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    save_figure(fig, save_path)


def plot_attacks_pie(df: pd.DataFrame, save_path: str | None = None):
    category_counts = df["Category"].value_counts()

    total_count = category_counts.sum()
    percentages = (category_counts / total_count) * 100

    min_percentage = 2.5
    mask_small = percentages < min_percentage
    other_counts = category_counts[mask_small]
    other_count_total = other_counts.sum()
    grouped_counts = category_counts[~mask_small].copy()

    if other_count_total > 0:
        grouped_counts["Other"] = other_count_total

    slices = grouped_counts.values
    labels = grouped_counts.index
    final_percentages = (slices / total_count) * 100  # type: ignore

    fig = plt.figure(figsize=(20, 12))
    wedges, _ = plt.pie(
        slices,  # type: ignore
        wedgeprops={"edgecolor": "black"},
        textprops={"color": "white", "fontsize": 10},
    )

    legend_labels = [
        f"{label} ({percentage:.2f}%)"
        for label, percentage in zip(labels, final_percentages)
    ]
    main_legend = plt.legend(
        legend_labels,
        title="Attack Name",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )
    plt.gca().add_artist(main_legend)

    other_percentages = (other_counts / total_count) * 100
    other_legend_labels = [
        f"{name} ({percentage:.2f}%)" for name, percentage in other_percentages.items()
    ]

    category_colours = {label: w.get_facecolor() for label, w in zip(labels, wedges)}
    other_colour = category_colours["Other"]
    temp = [
        Patch(facecolor=other_colour, edgecolor="black") for _ in other_counts.index
    ]

    breakdown_legend = plt.legend(
        temp,
        other_legend_labels,
        title="Other Breakdown",
        loc="center left",
        bbox_to_anchor=(1.0, 0.6),
        frameon=True,
    )

    plt.gca().add_artist(breakdown_legend)

    plt.title("Attack Category Distribution")
    plt.gca().set_aspect("equal")

    save_figure(fig, save_path)
