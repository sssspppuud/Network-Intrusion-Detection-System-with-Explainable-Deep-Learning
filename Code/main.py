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
            file_path = os.path.join(subset_path, filename)
            df = pd.read_csv(file_path)

            cleaned_filename = filename.replace(".pcap.csv", "")  # Remove extension
            if cleaned_filename[:-1].isdigit():
                # Remove number from end of filename
                cleaned_filename = cleaned_filename[:-1]

            if "Benign" in cleaned_filename:
                category = "Benign"
                attack = "Benign"
            elif "ARP_Spoofing" in cleaned_filename:
                category = "Spoofing"
                attack = "Spoofing"
            elif "MQTT-DDoS-Connect-Flood" in filename:
                category = "MQTT"
                attack = "MQTT-DDoS-Connect-Flood"
            elif "MQTT-DDoS-Publish-Flood" in filename:
                category = "MQTT"
                attack = "MQTT-DDoS-Publish-Flood"
            elif "MQTT-DoS-Connect-Flood" in filename:
                category = "MQTT"
                attack = "MQTT-DoS-Connect-Flood"
            elif "MQTT-DoS-Publish-Flood" in filename:
                category = "MQTT"
                attack = "MQTT-DoS-Publish-Flood"
            elif "MQTT-Malformed_Data" in filename:
                category = "MQTT"
                attack = "MQTT-Malformed_Data"
            elif "Recon-OS_Scan" in filename:
                category = "Recon"
                attack = "Recon-OS_Scan"
            elif "Recon-Ping_Sweep" in filename:
                category = "Recon"
                attack = "Recon-Ping_Sweep"
            elif "Recon-Port_Scan" in filename:
                category = "Recon"
                attack = "Recon-Port_Scan"
            elif "Recon-VulScan" in filename:
                category = "Recon"
                attack = "Recon-VulScan"
            elif "TCP_IP-DDoS-ICMP" in filename:
                category = "DDoS"
                attack = "DDoS-ICMP"
            elif "TCP_IP-DDoS-SYN" in filename:
                category = "DDoS"
                attack = "DDoS-SYN"
            elif "TCP_IP-DDoS-TCP" in filename:
                category = "DDoS"
                attack = "DDoS-TCP"
            elif "TCP_IP-DDoS-UDP" in filename:
                category = "DDoS"
                attack = "DDoS-UDP"
            elif "TCP_IP-DoS-ICMP" in filename:
                category = "DoS"
                attack = "DoS-ICMP"
            elif "TCP_IP-DoS-SYN" in filename:
                category = "DoS"
                attack = "DoS-SYN"
            elif "TCP_IP-DoS-TCP" in filename:
                category = "DoS"
                attack = "DoS-TCP"
            elif "TCP_IP-DoS-UDP" in filename:
                category = "DoS"
                attack = "DoS-UDP"

            df["Attack"] = attack
            df["Category"] = category

            dfs.append(df)

        except Exception as e:
            print(f"Failed to process file {filename} : {e}")

    if not dfs:
        print(f"No files found in {subset_path}")
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


train = load_dataset("CICIoMT2024", "train")
test = load_dataset("CICIoMT2024", "test")
dataset = pd.concat(
    [train, test], ignore_index=True
)  # Combining train and test sets to analyse full dataset

# Visualising Dataset:

sns.set_theme(style="whitegrid", context="paper", palette="viridis")
FIGURES_DIR = "./Figures"

# Distribution of Attack Types Bar chart

attack_counts = dataset["Attack"].value_counts()
attack_counts = attack_counts.iloc[::-1]  # Reverse order

plt.figure(figsize=(14, 6))
sns.barplot(x=attack_counts.index, y=attack_counts.values)
plt.title("Distribution of Attack Types")
plt.xlabel("Attack Type")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/AttackDistribution.png", dpi=600)

# Distribution of Attack Categories Bar chart (excluding ddos and dos)
category_counts = dataset[~dataset["Attack"].str.startswith(("DDoS", "DoS"))][
    "Attack"
].value_counts()
category_counts = category_counts.iloc[::-1]
plt.figure(figsize=(10, 6))
sns.barplot(x=category_counts.index, y=category_counts.values)
plt.title("Distribution of Attack Categories")
plt.xlabel("Attack Category")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/AttackCategoryDistribution.png", dpi=600)


"""
dataset.shape = (9162994, 41) dataset.columns = Index(['Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate', 'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'ack_flag_number', 'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count', 'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Variance', 'Category', 'Attack'], dtype='object')

"""
