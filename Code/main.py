import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset(root: str, subset: str) -> pd.DataFrame:
    subset_path = os.path.join(root, subset)

    if not os.path.exists(subset_path):
        raise FileNotFoundError(f"Directory not found: {subset_path}")

    dfs = []

    # Finds all relevant dataset files
    files = [file for file in os.listdir(subset_path) if file.endswith(".pcap.csv")]

    for filename in files:
        try:
            label, category, attack = None, None, None

            file_path = os.path.join(subset_path, filename)
            df = pd.read_csv(file_path)

            cleaned_filename = filename.replace(".pcap.csv", "")  # Remove extension
            if cleaned_filename[-1].isdigit():
                # Remove number from end of filename
                cleaned_filename = cleaned_filename[:-1]

            if "Benign" in cleaned_filename:
                label = "Benign"
                category = "Benign"
                attack = "Benign"

            elif "ARP_Spoofing" in cleaned_filename:
                label = "Attack"
                category = "Spoofing"
                attack = "ARP Spoofing"

            elif "Recon-Ping_Sweep" in filename:
                label = "Attack"
                category = "Recon"
                attack = "Ping Sweep"
            elif "Recon-VulScan" in filename:
                label = "Attack"
                category = "Recon"
                attack = "VulScan"
            elif "Recon-OS_Scan" in filename:
                label = "Attack"
                category = "Recon"
                attack = "OS Scan"
            elif "Recon-Port_Scan" in filename:
                label = "Attack"
                category = "Recon"
                attack = "Port Scan"

            elif "MQTT-Malformed_Data" in filename:
                label = "Attack"
                category = "MQTT"
                attack = "Malformed Data"
            elif "MQTT-DoS-Connect_Flood" in filename:
                label = "Attack"
                category = "MQTT"
                attack = "DoS Connect Flood"
            elif "MQTT-DDoS-Publish_Flood" in filename:
                label = "Attack"
                category = "MQTT"
                attack = "Publish Flood"
            elif "MQTT-DoS-Publish_Flood" in filename:
                label = "Attack"
                category = "MQTT"
                attack = "Publish Flood"
            elif "MQTT-DDoS-Connect_Flood" in filename:
                label = "Attack"
                category = "MQTT"
                attack = "Connect Flood"

            elif "TCP_IP-DoS-TCP" in filename:
                label = "Attack"
                category = "DoS"
                attack = "DoS TCP"
            elif "TCP_IP-DoS-ICMP" in filename:
                label = "Attack"
                category = "DoS"
                attack = "DoS ICMP"
            elif "TCP_IP-DoS-SYN" in filename:
                label = "Attack"
                category = "DoS"
                attack = "DoS SYN"
            elif "TCP_IP-DoS-UDP" in filename:
                label = "Attack"
                category = "DoS"
                attack = "DoS UDP"

            elif "TCP_IP-DDoS-SYN" in filename:
                label = "Attack"
                category = "DDoS"
                attack = "DDoS SYN"
            elif "TCP_IP-DDoS-TCP" in filename:
                label = "Attack"
                category = "DDoS"
                attack = "DDoS TCP"
            elif "TCP_IP-DDoS-ICMP" in filename:
                label = "Attack"
                category = "DDoS"
                attack = "DDoS ICMP"
            elif "TCP_IP-DDoS-UDP" in filename:
                label = "Attack"
                category = "DDoS"
                attack = "DDoS UDP"

            df["Attack"] = attack
            df["Category"] = category
            df["Class"] = label

            dfs.append(df)

        except Exception as e:
            print(f"Failed to process file {filename} : {e}")

    if not dfs:
        print(f"No files found in {subset_path}")
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def save_figure(fig, save_path: str | None):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=600)
    plt.close(fig)


def plot_attack_distribution(df: pd.DataFrame, save_path: str | None = None):
    attack_counts = df["Attack"].value_counts().sort_values(ascending=True)
    fig = plt.figure(figsize=(12, 6))
    sns.barplot(x=attack_counts.index, y=attack_counts.values)
    plt.ticklabel_format(style="plain", axis="y")
    plt.title("Distribution of Attacks")
    plt.xlabel("Attack Name")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    save_figure(fig, save_path)


def plot_ddos_dos(df: pd.DataFrame, save_path: str | None = None):
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


def plot_attack_categories(df: pd.DataFrame, save_path: str | None = None):
    filtered_df = df[df["Category"].isin(["MQTT", "Spoofing", "Recon", "Benign"])]
    category_counts = filtered_df["Category"].value_counts().sort_values(ascending=True)
    fig = plt.figure(figsize=(12, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values)
    plt.ticklabel_format(style="plain", axis="y")
    plt.title("Distribution of Attack Categories")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    save_figure(fig, save_path)


if __name__ == "__main__":
    sns.set_theme(style="whitegrid", context="paper", palette="viridis")

    train = load_dataset("data", "train")
    test = load_dataset("data", "test")
    df = pd.concat([train, test], ignore_index=True)

    # Show dataset attack classes and distribution
    print(df[["Category", "Attack", "Class"]].value_counts())

    save_root = "./Figures"
    plot_attack_categories(df, save_root + "/AttackCategoryDistribution.png")
    plot_attack_distribution(df, save_root + "/AttackDistribution.png")
    plot_ddos_dos(df, save_root + "/DDoS_DoS_Plot.png")
