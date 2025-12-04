import glob
import pandas as pd
from typing import Tuple


def get_labels_from_filename(filename: str) -> Tuple[str, str, str]:
    if "Benign" in filename:
        return "Benign", "Benign", "Benign"

    label = "Attack"

    if "ARP_Spoofing" in filename:
        return label, "Spoofing", "ARP Spoofing"

    if "Recon" in filename:
        category = "Recon"
        if "Recon-Ping_Sweep" in filename:
            return label, category, "Ping Sweep"
        if "Recon-VulScan" in filename:
            return label, category, "VulScan"
        if "Recon-OS_Scan" in filename:
            return label, category, "OS Scan"
        if "Recon-Port_Scan" in filename:
            return label, category, "Port Scan"

    if "MQTT" in filename:
        category = "MQTT"
        if "MQTT-Malformed_Data" in filename:
            return label, category, "Malformed Data"
        if "MQTT-DoS-Connect_Flood" in filename:
            return label, category, "DoS Connect Flood"
        if "MQTT-DDoS-Publish_Flood" in filename:
            return label, category, "DDoS Publish Flood"
        if "MQTT-DoS-Publish_Flood" in filename:
            return label, category, "DoS Publish Flood"
        if "MQTT-DDoS-Connect_Flood" in filename:
            return label, category, "DDoS Connect Flood"

    if "TCP_IP-DoS" in filename:
        category = "DoS"
        if "TCP_IP-DoS-TCP" in filename:
            return label, category, "DoS TCP"
        if "TCP_IP-DoS-ICMP" in filename:
            return label, category, "DoS ICMP"
        if "TCP_IP-DoS-SYN" in filename:
            return label, category, "DoS SYN"
        if "TCP_IP-DoS-UDP" in filename:
            return label, category, "DoS UDP"

    if "TCP_IP-DDoS" in filename:
        category = "DDoS"
        if "TCP_IP-DDoS-SYN" in filename:
            return label, category, "DDoS SYN"
        if "TCP_IP-DDoS-TCP" in filename:
            return label, category, "DDoS TCP"
        if "TCP_IP-DDoS-ICMP" in filename:
            return label, category, "DDoS ICMP"
        if "TCP_IP-DDoS-UDP" in filename:
            return label, category, "DDoS UDP"

    return "", "", ""  # Should never reach here with correct dataset format


# def dataset_to_single_file(root: str, out: str):
#     path = f"{root}/*/*.csv"
#     csv_files = glob.glob(path)

#     dfs = [pd.read_csv(file) for file in csv_files]
#     combined_df = pd.concat(dfs, ignore_index=True)
#     combined_df.to_csv("dataset.csv", index=False)


def combine_dataset(root: str) -> None:
    """
    Load all csv files for the dataset, ands categorical columns, then
    saves it to a single csv file.

    :param root: Path to the dataset stored in the form train/ and test/
    :type root: str
    """
    path = f"{root}/*/*.csv"
    dfs = []
    for file in glob.glob(path, recursive=True):
        label, category, attack = get_labels_from_filename(file)
        df = pd.read_csv(file)
        df[["Label", "Category", "Attack"]] = [label, category, attack]
        dfs.append(df)
    dfs = pd.concat(dfs)

    for column in dfs.select_dtypes(include=["int"]):
        dfs[column] = pd.to_numeric(dfs[column], downcast="integer")

    for column in dfs.select_dtypes(include=["float"]):
        dfs[column] = pd.to_numeric(dfs[column], downcast="float")

    dfs.to_csv(f"{root}/combined_dataset.csv", index=False)
