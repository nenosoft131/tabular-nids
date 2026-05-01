import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import ipaddress
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
import time

# -------------------------
# 1. Load NetFlow Dataset
# -------------------------
# df = pd.read_csv("netflow.csv")
df = pd.read_csv("dataset/NF-CSE-CIC-IDS2018-v2/data/NF-CSE-CIC-IDS2018-v2.csv")  

# Optional: Save and remove label column
if 'Label' in df.columns:
    labels = df['Label'].values
    df = df.drop(columns=['Label', 'Attack'])
else:
    raise ValueError("Label column not found in dataset for evaluation.")

def ip_to_int(ip_str):
    try:
        return int(ipaddress.IPv4Address(ip_str))
    except:
        return 0

df["IPV4_SRC_ADDR"] = df["IPV4_SRC_ADDR"].apply(lambda x: int(ipaddress.IPv4Address(x)))
df["IPV4_DST_ADDR"] = df["IPV4_DST_ADDR"].apply(lambda x: int(ipaddress.IPv4Address(x)))
# -------------------------
# 2. Normalize Features
# -------------------------
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# -------------------------
# 3. Train/Val/Test Split
# -------------------------
X_train_full, X_temp, y_train_full, y_temp = train_test_split(
    data_scaled, labels, test_size=0.3, stratify=labels, random_state=42)

# Use only normal data (label=0) for training
X_train = X_train_full[y_train_full == 0]

# Split the rest into val and test sets
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Convert to torch tensors
train_tensor = torch.tensor(X_train, dtype=torch.float32)
val_tensor = torch.tensor(X_val, dtype=torch.float32)
test_tensor = torch.tensor(X_test, dtype=torch.float32)



latent_dims = [64, 32, 16, 8, 4]
all_results = []

for latent_dim in latent_dims:
    print(f"\n🚀 Training Autoencoder with latent_dim = {latent_dim}")

    # Define model
    class Autoencoder(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, latent_dim),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, input_dim)
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=128, shuffle=True)
    epochs = 20
    start_fit = time.time()

    # Training
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            output = model(batch)
            loss = criterion(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    time_fit = time.time() - start_fit

    # Inference
    model.eval()
    with torch.no_grad():
        train_recon = model(train_tensor)
        train_errors = torch.mean((train_tensor - train_recon) ** 2, dim=1).numpy()
        threshold = np.percentile(train_errors, 97)

        val_recon = model(val_tensor)
        val_errors = torch.mean((val_tensor - val_recon) ** 2, dim=1).numpy()
        val_preds = (val_errors > threshold).astype(int)

        test_recon = model(test_tensor)
        test_errors = torch.mean((test_tensor - test_recon) ** 2, dim=1).numpy()
        test_preds = (test_errors > threshold).astype(int)

    time_inference = time.time() - start_fit - time_fit

    # Evaluation
    result = {
        'val_aucroc': roc_auc_score(y_val, val_errors),
        'val_aucpr': average_precision_score(y_val, val_errors),
        'val_accuracy': accuracy_score(y_val, val_preds),
        'val_precision': precision_score(y_val, val_preds),
        'val_recall': recall_score(y_val, val_preds),
        'val_f1': f1_score(y_val, val_preds),
        'val_f1_macro': f1_score(y_val, val_preds, average='macro'),
        'val_f1_micro': f1_score(y_val, val_preds, average='micro'),
        'val_confusion_matrix': confusion_matrix(y_val, val_preds).tolist(),

        'test_aucroc': roc_auc_score(y_test, test_errors),
        'test_aucpr': average_precision_score(y_test, test_errors),
        'test_accuracy': accuracy_score(y_test, test_preds),
        'test_precision': precision_score(y_test, test_preds),
        'test_recall': recall_score(y_test, test_preds),
        'test_f1': f1_score(y_test, test_preds),
        'test_f1_macro': f1_score(y_test, test_preds, average='macro'),
        'test_f1_micro': f1_score(y_test, test_preds, average='micro'),
        'test_confusion_matrix': confusion_matrix(y_test, test_preds).tolist(),

        'latent_dim': latent_dim,
        'time_fit': time_fit,
        'time_inference': time_inference
    }

    all_results.append(result)

# Create a DataFrame
df_AUCROC = pd.DataFrame([{
    'model_name': f"Autoencoder{res['latent_dim']}",
    'params': {'latent_dim': res['latent_dim']},
    'config': {'threshold_method': 'percentile_97'},

    'AUCROC-Val': res['val_aucroc'],
    'AUCPR-Val': res['val_aucpr'],
    'Accuracy-Val': res['val_accuracy'],
    'Precision-Val': res['val_precision'],
    'Recall-Val': res['val_recall'],
    'F1-Val': res['val_f1'],
    'F1-Macro-Val': res['val_f1_macro'],
    'F1-Micro-Val': res['val_f1_micro'],
    'Confusion-Matrix-Val': res['val_confusion_matrix'],

    'AUCROC-Test': res['test_aucroc'],
    'AUCPR-Test': res['test_aucpr'],
    'Accuracy-Test': res['test_accuracy'],
    'Precision-Test': res['test_precision'],
    'Recall-Test': res['test_recall'],
    'F1-Test': res['test_f1'],
    'F1-Macro-Test': res['test_f1_macro'],
    'F1-Micro-Test': res['test_f1_micro'],
    'Confusion-Matrix-Test': res['test_confusion_matrix'],

    'TIME_FIT': res['time_fit'],
    'TIME_INFER': res['time_inference']
} for res in all_results])

# Save to CSV
df_AUCROC.to_csv("autoencoder_results_l.csv", index=False)
print("\n✅ Results for all latent sizes saved to netflow_autoencoder_latent_results.csv")
