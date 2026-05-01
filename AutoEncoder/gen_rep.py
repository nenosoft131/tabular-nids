import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import ipaddress
# === Step 1: Load and preprocess CSV ===

# Load your NetFlow CSV file
csv_path = "dataset/NF-CSE-CIC-IDS2018-v2.csv"  # TODO: update this path
df = pd.read_csv(csv_path)

def ip_to_int(ip_str):
    try:
        return int(ipaddress.IPv4Address(ip_str))
    except:
        return 0

# df["Src IP Addr"] = df["Src IP Addr"].apply(ip_to_int)
# df["Dst IP Addr"] = df["Dst IP Addr"].apply(ip_to_int)

df["IPV4_SRC_ADDR"] = df["IPV4_SRC_ADDR"].apply(ip_to_int)
df["IPV4_DST_ADDR"] = df["IPV4_DST_ADDR"].apply(ip_to_int)

# df['class'] = df['class'].apply(lambda x: 0 if x in ['normal'] else 1)

label_to_id = {
    'Bot': 1, 'Brute Force -W': 2, 'Benign': 0, 'Brute Force -X': 2, 'DDOS attack-HO': 3,
    'DDOS attack-LO': 3, 'DDoS attacks-L': 3, 'DoS attacks-Go': 4, 'DoS attacks-Hu': 4, 'DoS attacks-Sl': 4,
    'FTP-BruteForce': 5, 'Infilteration': 6, 'SQL Injection': 7, 'SSH-Bruteforce': 8 
}
# label_to_id = {
#     'attacker': 1, 'normal': 0, 'victim': 2
# }

# In-place replace (exact match, case-sensitive)
df['Attack'] = df['Attack'].map(label_to_id).astype('Int64')


# Optional: Save and remove label column
if 'Attack' in df.columns:
    labels = df['Attack'].values
    drop_cols = [
    "Attack", "Label"
    ]
    df = df.drop(columns=drop_cols)
else:
    raise ValueError("Label column not found in dataset for evaluation.")

for col in df.columns:
    if df[col].dtype == object:
        df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.fillna(0)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# === Step 2: Define and train Autoencoder ===

input_dim = X_scaled.shape[1]
encoding_dim = 64  # You can tune this

# Define Autoencoder model
input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

autoencoder.compile(optimizer='adam', loss='mse')

# Train Autoencoder
autoencoder.fit(X_scaled, X_scaled,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_split=0.1,
                verbose=1)

# === Step 3: Generate compressed representations ===

X_representations = encoder.predict(X_scaled)

# === Step 4: Split train/test and save to .npz ===

if labels is not None:
    X_train, X_test, y_train, y_test = train_test_split(
        X_representations, labels, test_size=0.2, random_state=42
    )

    np.savez('netflow_autoencoder_repr_NF-CSE-CIC-IDS2018-v2.npz',
             X_train=X_train,
             y_train=y_train,
             X_test=X_test,
             y_test=y_test)

else:
    X_train, X_test = train_test_split(X_representations, test_size=0.2, random_state=42)

    np.savez('netflow_autoencoder_repr_NF-CSE-CIC-IDS2018-v2.npz',
             X_train=X_train,
             X_test=X_test)

print("✅ Saved compressed representations to 'NF-CSE-CIC-IDS2018-v2.npz'")
