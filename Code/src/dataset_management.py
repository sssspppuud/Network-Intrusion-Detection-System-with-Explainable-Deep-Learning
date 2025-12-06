import glob
from typing import Tuple

import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq

import pandas as pd


def get_labels_from_filename(filename: str) -> Tuple[str, str, bool]:
    if "Benign" in filename:
        return "Benign", "Benign", False

    if "ARP_Spoofing" in filename:
        return "ARP_Spoofing", "Spoofing", True

    if "Recon" in filename:
        category = "Recon"
        if "Recon-Ping_Sweep" in filename:
            return "Ping_Sweep", category, True
        if "Recon-VulScan" in filename:
            return "VulScan", category, True
        if "Recon-OS_Scan" in filename:
            return "OS_Scan", category, True
        if "Recon-Port_Scan" in filename:
            return "Port_Scan", category, True

    if "MQTT" in filename:
        category = "MQTT"
        if "MQTT-Malformed_Data" in filename:
            return "Malformed_Data", category, True
        if "MQTT-DoS-Connect_Flood" in filename:
            return "DoS_Connect_Flood", category, True
        if "MQTT-DDoS-Publish_Flood" in filename:
            return "DDoS_Publish_Flood", category, True
        if "MQTT-DoS-Publish_Flood" in filename:
            return "DoS_Publish Flood", category, True
        if "MQTT-DDoS-Connect_Flood" in filename:
            return "DDoS_Connect_Flood", category, True

    if "TCP_IP-DoS" in filename:
        category = "DoS"
        if "TCP_IP-DoS-TCP" in filename:
            return "DoS_TCP", category, True
        if "TCP_IP-DoS-ICMP" in filename:
            return "DoS_ICMP", category, True
        if "TCP_IP-DoS-SYN" in filename:
            return "DoS_SYN", category, True
        if "TCP_IP-DoS-UDP" in filename:
            return "DoS_UDP", category, True

    if "TCP_IP-DDoS" in filename:
        category = "DDoS"
        if "TCP_IP-DDoS-SYN" in filename:
            return "DDoS_SYN", category, True
        if "TCP_IP-DDoS-TCP" in filename:
            return "DDoS_TCP", category, True
        if "TCP_IP-DDoS-ICMP" in filename:
            return "DDoS_ICMP", category, True
        if "TCP_IP-DDoS-UDP" in filename:
            return "DDoS_UDP", category, True

    return "", "", False  # Should never reach here with correct dataset format


def combine_dataset_to_pq_pyarrow(root: str) -> None:
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

        table = table.append_column("Name", pa.array([name] * n, type=pa.string()))
        table = table.append_column(
            "Category", pa.array([category] * n, type=pa.string())
        )
        table = table.append_column("Attack", pa.array([attack] * n, type=pa.bool_()))

        tables.append(table)
    combined = pa.concat_tables(tables)
    dict_type = pa.dictionary(pa.int32(), pa.string())

    schema = combined.schema
    new_fields = []
    for field in schema:
        if field.name in ["Name", "Category"]:
            new_fields.append(pa.field(field.name, dict_type))
        else:
            new_fields.append(field)

    new_schema = pa.schema(new_fields)
    combined = combined.cast(new_schema)

    pq.write_table(combined, out_path)
