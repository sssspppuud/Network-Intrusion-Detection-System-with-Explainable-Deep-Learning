import glob
from typing import Tuple

import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq


def get_labels_from_filename(filename: str) -> Tuple[str, str, bool]:
    if "Benign" in filename:
        return "Benign", "Benign", False

    if "ARP_Spoofing" in filename:
        return "ARP Spoofing", "Spoofing", True

    if "Recon" in filename:
        category = "Recon"
        if "Recon-Ping_Sweep" in filename:
            return "Ping Sweep", category, True
        if "Recon-VulScan" in filename:
            return "VulScan", category, True
        if "Recon-OS_Scan" in filename:
            return "OS Scan", category, True
        if "Recon-Port_Scan" in filename:
            return "Port Scan", category, True

    if "MQTT" in filename:
        category = "MQTT"
        if "MQTT-Malformed_Data" in filename:
            return "Malformed Data", category, True
        if "MQTT-DoS-Connect_Flood" in filename:
            return "DoS Connect Flood", category, True
        if "MQTT-DDoS-Publish_Flood" in filename:
            return "DDoS Publish Flood", category, True
        if "MQTT-DoS-Publish_Flood" in filename:
            return "DoS Publish Flood", category, True
        if "MQTT-DDoS-Connect_Flood" in filename:
            return "DDoS Connect Flood", category, True

    if "TCP_IP-DoS" in filename:
        category = "DoS"
        if "TCP_IP-DoS-TCP" in filename:
            return "DoS TCP", category, True
        if "TCP_IP-DoS-ICMP" in filename:
            return "DoS ICMP", category, True
        if "TCP_IP-DoS-SYN" in filename:
            return "DoS SYN", category, True
        if "TCP_IP-DoS-UDP" in filename:
            return "DoS UDP", category, True

    if "TCP_IP-DDoS" in filename:
        category = "DDoS"
        if "TCP_IP-DDoS-SYN" in filename:
            return "DDoS SYN", category, True
        if "TCP_IP-DDoS-TCP" in filename:
            return "DDoS TCP", category, True
        if "TCP_IP-DDoS-ICMP" in filename:
            return "DDoS ICMP", category, True
        if "TCP_IP-DDoS-UDP" in filename:
            return "DDoS UDP", category, True

    return "", "", False  # Should never reach here with correct dataset format


def combine_dataset_to_pq(root: str) -> None:
    """
    Load all csv files for the dataset, ands categorical columns, then
    saves it to a single csv file.

    :param root: Path to the dataset stored in the form train/ and test/
    :type root: str
    """
    tables = []
    path = f"{root}/*/*.csv"
    files = glob.glob(path, recursive=True)
    out_path = f"{root}/combined_dataset.parquet"

    for file in files:
        name, category, attack = get_labels_from_filename(file)
        table = csv.read_csv(file)
        n = table.num_rows

        table = table.append_column("Name", pa.array([name] * n))
        table = table.append_column("Category", pa.array([category] * n))
        table = table.append_column("Attack", pa.array([attack] * n))

        tables.append(table)
    combined = pa.concat_tables(tables)
    pq.write_table(combined, out_path)
