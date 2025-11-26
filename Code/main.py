import pandas as pd
import os, glob
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
#-------------------------------------------------------------
# Loading dataset
#-------------------------------------------------------------
path = "CICIoMT2024/train"
train_files = glob.glob(os.path.join(path, "*.pcap.csv"))

def load_labelled_data(path):
    all_files = glob.glob(os.path.join(path, "*.pcap.csv"))
    dfs = []

    for name in all_files:
        df = pd.read_csv(name, index_col = None, header = 0)
        basename = os.path.basename(name).split("_")[0]
        if "Benign" in basename:
            label = "Benign"
        elif "ARP_Spoofing" in basename:
            label = "ARP_Spoofing"
        elif "MQTT" in basename:
            label = "MQTT"
        elif "Recon" in basename:
            label = "Recon"
        elif "TCP_IP-DDoS" in basename:
            label = "TCP_IP-DDoS"
        elif "TCP_IP-DoS" in basename:
            label = "TCP_IP-DoS"
        else:
            if "TCP_IP-DDoS" in name:
                label = "TCP_IP-DDoS"
            elif "TCP_IP-DoS" in name:
                label = "TCP_IP-DoS"
        
        df["Label"] = label
        dfs.append(df)
    
    main_df = pd.concat(dfs, axis = 0, ignore_index = True)
    return main_df

df_train = load_labelled_data("CICIoMT2024/train")
df_test= load_labelled_data("CICIoMT2024/test")

#------------------------------------------------------------

print("Shape =", df_train.shape)
print(f"Columns: {df_train.columns}")
print(df_train.head())

#------------------------------------------------------------
# Preprocessing
#------------------------------------------------------------
def preprocess(df):
    df.drop_duplicates(inplace = True)
    X = df.drop("Label", axis = 1)
    y = df["Label"]

    X.replace([np.inf, -np.inf], np.nan, inplace = True)
    # X.fillna(X.mean)
    return X, y

X_train, y_train = preprocess(df_train)

