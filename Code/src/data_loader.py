import os
import pandas as pd
import matplotlib.pyplot as plt


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
