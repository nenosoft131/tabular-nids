import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import os
from tabicl import TabICLClassifier
import numpy as np
from sklearn.preprocessing import LabelEncoder
import ipaddress

# Parameters
batch_size = 60000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Convert IPs to integers
def ip_to_int(ip_str):
    try:
        return int(ipaddress.IPv4Address(ip_str))
    except:
        return 0

# Resume setup
checkpoint_path = "resume_checkpoint.txt"
start_batch = 1
if os.path.exists(checkpoint_path):
    with open(checkpoint_path, "r") as f:
        start_batch = int(f.read().strip())
    print(f"Resuming from batch {start_batch}")

# Load and clean data
df = pd.read_csv("dataset/NF-UNSW-NB15-v2/data/NF-UNSW-NB15-v2.csv")
df_train = pd.read_csv("dataset/NF-UNSW-NB15-v2/data/NF-CSE-CIC-IDS2018-v2csv")
df["IPV4_SRC_ADDR"] = df["IPV4_SRC_ADDR"].apply(lambda x: int(ipaddress.IPv4Address(x)))
df["IPV4_DST_ADDR"] = df["IPV4_DST_ADDR"].apply(lambda x: int(ipaddress.IPv4Address(x)))
df_train["IPV4_SRC_ADDR"] = df_train["IPV4_SRC_ADDR"].apply(lambda x: int(ipaddress.IPv4Address(x)))
df_train["IPV4_DST_ADDR"] = df_train["IPV4_DST_ADDR"].apply(lambda x: int(ipaddress.IPv4Address(x)))


label_column = "Attack"

y = df[label_column]

drop_cols = ["Label", "Attack"]
X = df.drop(columns=drop_cols)

for col in X.columns:
    if X[col].dtype == object:
        X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.fillna(0)

# Split
X_train_df, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
label_encoder = LabelEncoder()
y_train_encoded = pd.Series(label_encoder.fit_transform(y_train)) 


# Init model
clf = TabICLClassifier()
clf.fit(X_train_df.iloc[:batch_size], y_train.iloc[:batch_size])  # Init model
clf.model_ = clf.model_.to(device)
clf.model_.train()

# Batch loop
n_rows = len(X_train_df)
num_batches = (n_rows + batch_size - 1) // batch_size
all_outputs = []

for i in range(start_batch, num_batches):
    try:
        start = i * batch_size
        end = min((i + 1) * batch_size, n_rows)
        X_batch = X_train_df.iloc[start:end]
        y_batch = y_train.iloc[start:end]
        y_batch_pass = y_train_encoded.iloc[start:end]

        if len(X_batch) < 100:
            print(f"Skipping small batch {i} of size {len(X_batch)}")
            continue
        
        # y_numpy = y_batch.to_numpy()
        # npy_filename = f"Tabicl/tabicl_embeddings/labels_batch_{i}.npy"
        # os.makedirs(os.path.dirname(npy_filename), exist_ok=True)
        # np.save(npy_filename, y_numpy)
        y_numpy = y_batch.astype(str).to_numpy(dtype='<U14')
        npy_filename = f"Tabicl/tabicl_embeddings/labels_batch_{i}.npy"
        os.makedirs(os.path.dirname(npy_filename), exist_ok=True)
        np.save(npy_filename, y_numpy)
        

        X_tensor = torch.tensor(X_batch.values, dtype=torch.float32).unsqueeze(0).to(device)
        y_tensor = torch.tensor(y_batch_pass.values, dtype=torch.long).unsqueeze(0).to(device)


        print(f"Processing batch {i+1}/{num_batches}")

        with torch.no_grad():
            output = clf.model_.forward(
                X=X_tensor,
                y_train=y_tensor,
                d=None,
                feature_shuffles=None,
                embed_with_test=False,
                return_logits=True,
                softmax_temperature=0.9,
                inference_config=None,
            )
        # all_outputs.append(output.detach().cpu())

        with open(checkpoint_path, "w") as f:
            f.write(str(i + 1))  # save next batch index

    except Exception as e:
        print(f"Crash at batch {i}: {e}")
        break

# Combine outputs
# if all_outputs:
#     final_outputs = torch.cat(all_outputs, dim=1)
#     print("Combined output shape:", final_outputs.shape)
