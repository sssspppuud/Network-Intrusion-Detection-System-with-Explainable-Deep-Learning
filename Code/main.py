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


train = load_dataset("data", "train")
test = load_dataset("data", "test")
df = pd.concat(
    [train, test], ignore_index=True
)  # Combining train and test sets to analyse full dataset

print(df[["Category", "Attack", "Class"]].value_counts())

# Visualising  the Dataset:
sns.set_theme(style="whitegrid", context="paper", palette="viridis")
FIGURES_DIR = "./Figures"

# DDoS and DoS plot
ddos_count = len(df[df["Category"] == "DDoS"])
dos_count = len(df[df["Category"] == "DoS"])
plot_data = pd.DataFrame(
    {"Attack Type": ["DDoS", "DoS"], "Count": [ddos_count, dos_count]}
).sort_values(by="Count", ascending=True)

plt.figure(figsize=(12, 6))
sns.barplot(x="Attack Type", y="Count", data=plot_data)
plt.ticklabel_format(style="plain", axis="y")
plt.title("Count of DDoS vs. DoS Traffic")
plt.ylabel("Count")
plt.savefig(f"{FIGURES_DIR}/DDoS_DoS_Plot.png", dpi=600)

# Attack distribution
attack_counts = df["Attack"].value_counts().sort_values(ascending=True)
plt.figure(figsize=(12, 6))
sns.barplot(x=attack_counts.index, y=attack_counts.values)
plt.ticklabel_format(style="plain", axis="y")
plt.title("Distribution of Attacks")
plt.xlabel("Attack Name")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.savefig(f"{FIGURES_DIR}/AttackDistribution.png", dpi=600)

# Attack categories
filtered_df = df[df["Category"].isin(["MQTT", "Spoofing", "Recon", "Benign"])]
category_counts = filtered_df["Category"].value_counts().sort_values(ascending=True)
plt.figure(figsize=(12, 6))
sns.barplot(x=category_counts.index, y=category_counts.values)
plt.ticklabel_format(style="plain", axis="y")
plt.title("Distribution of Attack Categories")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.savefig(f"{FIGURES_DIR}/AttackCategoryDistribution.png", dpi=600)

"""
dataset.shape = (9162994, 41) dataset.columns = Index(['Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate', 'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'ack_flag_number', 'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count', 'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Variance', 'Category', 'Attack'], dtype='object')

"""
