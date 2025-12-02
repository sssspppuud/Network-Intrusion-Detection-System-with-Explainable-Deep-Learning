import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


def save_figure(fig, save_path: str | None):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=600)
    plt.close(fig)


def plot_attack_distribution_bar(df: pd.DataFrame, save_path: str | None = None):
    sns.set_theme(style="whitegrid", context="paper", palette="viridis")

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
    sns.set_theme(style="whitegrid", context="paper", palette="viridis")

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
    sns.set_theme(style="whitegrid", context="paper", palette="viridis")

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
    sns.set_theme(style="whitegrid", context="paper", palette="viridis")

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
    sns.set_theme(style="whitegrid", context="paper", palette="viridis")

    table = df.groupby(["Category", "Attack"]).size()
    outer = table.groupby(level=0).sum()
    inner = table

    cmap_outer = plt.get_cmap("tab20c")

    category_count = len(outer)
    outer_colours = [cmap_outer(i / category_count) for i in range(category_count)]
    category_colour_map = dict(zip(outer.index, outer_colours))
    inner_colours = [category_colour_map[category] for category, attack in inner.index]

    fig, ax = plt.subplots(figsize=(14, 10))
    size = 0.3

    explode_index = outer.index.get_loc(
        "Spoofing"
    )  # Really small slice needs highlighting
    explode_values = [0] * len(outer)
    explode_values[explode_index] = 0.15

    wedges_outer = ax.pie(
        outer.values,
        radius=1,
        colors=outer_colours,
        labels=None,
        wedgeprops=dict(width=size, edgecolor="w"),
        startangle=90,
        explode=explode_values,
    )[0]

    inner_explode_values = []
    for category, _ in inner.index:
        if category == "Spoofing":
            inner_explode_values.append(explode_values[explode_index])
        else:
            inner_explode_values.append(0)

    wedges_inner = ax.pie(
        inner.values,
        radius=1 - size,
        colors=inner_colours,
        labels=None,
        wedgeprops=dict(width=size, edgecolor="w"),
        startangle=90,
        explode=inner_explode_values,
    )[0]

    ax.set_aspect("equal")

    legend_outer = ax.legend(
        wedges_outer,
        outer.index,
        title="Category",
        loc="upper left",
        bbox_to_anchor=(1.0, 0.95),
    )
    ax.add_artist(legend_outer)

    legend_inner = ax.legend(
        wedges_inner,
        [attack_name for _, attack_name in inner.index],
        title="Attack",
        loc="upper left",
        bbox_to_anchor=(1.0, 0.5),
        ncol=2,
    )
    ax.add_artist(legend_inner)

    save_figure(fig, save_path)
