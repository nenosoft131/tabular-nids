import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
import warnings

from model import Neural_Net
from dataset_samplers import SupervisedSampler, ClassCorruptSampler
from corruption_mask_generators import RandomMaskGenerator
from training import train_contrastive_loss, train_classification
from utils import fix_seed, get_bootstrapped_targets, preprocess_datasets, fit_one_hot_encoder, FRACTION_LABELED

warnings.filterwarnings('ignore')
print("🔕 Disabled warnings!")

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using DEVICE: {DEVICE}")

METHODS = ['No Pre-train', 'Class-Conditioned (Ours)']
seed = 614579
fix_seed(seed)


def save_embeddings_in_batches(model, data, labels, batch_size, prefix, encoder, device, index):
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    num_samples = data.shape[0]
    total_batches = (num_samples + batch_size - 1) // batch_size

    model.eval()

    for batch_idx in range(total_batches):
        emb_path = f"{prefix}_embeddings_batch{index}.npy"
        label_path = f"{prefix}_labels_batch{index}.npy"

        if os.path.exists(emb_path) and os.path.exists(label_path):
            print(f"⏭️ Skipping batch {batch_idx} (already saved)")
            continue

        start = batch_idx * batch_size
        end = min(start + batch_size, num_samples)

        batch_data = data[start:end]
        batch_labels = labels[start:end]

        with torch.no_grad():
            batch_input = encoder.transform(batch_data)
            batch_input_tensor = torch.tensor(batch_input, dtype=torch.float32).to(device)
            batch_embeddings = model.module.get_middle_embedding(batch_input_tensor).cpu().numpy()

        np.save(emb_path, batch_embeddings)
        np.save(label_path, batch_labels)
        print(f"✅ Saved batch {batch_idx + 1}/{total_batches}")



# Load and prepare dataset
# df = pd.read_csv("dataset/CIDDS/CIDDS-001-internal-week1.csv")


file_path = "dataset/NF-UNSW-NB15-v2/data/NF-UNSW-NB15-v2.csv"

# Step 1: Count total rows (excluding header)
# total_rows = sum(1 for _ in open(file_path)) - 1  # subtract header
# chunk_size = total_rows // 40

# chunks = []
# for i in range(4):
#     skip = 1 + i * chunk_size  # +1 to skip header
#     nrows = chunk_size if i < 3 else total_rows - skip + 1  # last chunk gets remainder
#     chunk = pd.read_csv(file_path, skiprows=range(1, skip), nrows=nrows)
#     chunks.append(chunk)
#     print(f"✅ Loaded chunk {i+1}: Rows {skip} to {skip + nrows - 1}")

# # Example: access chunk 1
# df = chunks[0]


chunk_size = 60000

chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)

for i, df in enumerate(chunk_iter):
    if i < 18 :
        print(f"Skipping chunk {i} (already processed)")
        continue
    
    target = df["Attack"]
    drop_cols = ["Label", "Attack"]
    data = df.drop(columns=[col for col in drop_cols if col in df.columns])
    

    # Convert categorical to numeric
    for col in data.columns:
        if data[col].dtype == object:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    X = data.fillna(0)

    # Split dataset
    train_data, test_data, train_targets, test_targets = train_test_split(
        data, target, test_size=0.2, stratify=target, random_state=seed
    )
    original_train_labels = df.loc[train_data.index, "Attack"].reset_index(drop=True)
    original_test_labels = df.loc[test_data.index, "Attack"].reset_index(drop=True)
    # Encode labels
    label_encoder = LabelEncoder()
    train_targets = label_encoder.fit_transform(train_targets)
    test_targets = label_encoder.transform(test_targets)
    n_classes = len(np.unique(train_targets))

    # Normalize data and fit one-hot encoder
    train_data, test_data = preprocess_datasets(train_data, test_data, normalize_numerical_features=True)
    one_hot_encoder = fit_one_hot_encoder(preprocessing.OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False), train_data)

    # Sample labeled data
    n_train_samples_labeled = int(len(train_data) * FRACTION_LABELED)
    idxes_tmp = np.random.permutation(len(train_data))[:n_train_samples_labeled]
    mask_train_labeled = np.zeros(len(train_data), dtype=bool)
    mask_train_labeled[idxes_tmp] = True

    supervised_sampler = SupervisedSampler(data=train_data[mask_train_labeled], target=train_targets[mask_train_labeled])

    # Initialize models
    models, contrastive_loss_histories, supervised_loss_histories = {}, {}, {}
    for method in METHODS:
        models[method] = nn.DataParallel(Neural_Net(
            input_dim=train_data.shape[1],
            emb_dim=64,
            output_dim=n_classes
        )).to(DEVICE)

    # Supervised training (No Pre-train)
    models['No Pre-train'].module.freeze_encoder()
    print("🔁 Supervised training (No Pre-train)...")
    print("HELLLLOX")
    train_losses = train_classification(models['No Pre-train'], supervised_sampler, one_hot_encoder)
    supervised_loss_histories['No Pre-train'] = train_losses

    print("ONE")
    # Class-conditioned sampling
    bootstrapped_train_targets = get_bootstrapped_targets(
        train_data, train_targets, models['No Pre-train'], mask_train_labeled, one_hot_encoder
    )
    print("TWO")
    contrastive_samplers = {
        'Class-Conditioned (Ours)': ClassCorruptSampler(train_data, bootstrapped_train_targets)
    }

    mask_generator = RandomMaskGenerator(train_data.shape[1])
    print("Three")
    # Contrastive training
    for method in METHODS:
        if method == "No Pre-train":
            continue
        train_losses = train_contrastive_loss(models[method],
                                            method,
                                            contrastive_samplers[method],
                                            supervised_sampler,
                                            mask_generator,
                                            mask_train_labeled,
                                            one_hot_encoder)
        contrastive_loss_histories[method] = train_losses

    # ---------------------------------------
    # 🔄 Batch Embedding Saver with Checkpoint
    # ---------------------------------------
    # Save train and test embeddings
    batch_size = 60000

    print("💾 Saving train embeddings in batches...")
    save_embeddings_in_batches(
        model=models['Class-Conditioned (Ours)'],
        data=train_data.reset_index(drop=True),
        labels=original_train_labels,
        batch_size=batch_size,
        prefix="CC/class_conditioned_embeddings/train",
        encoder=one_hot_encoder,
        device=DEVICE,
        index = i
    )

    print("💾 Saving test embeddings in batches...")
    save_embeddings_in_batches(
        model=models['Class-Conditioned (Ours)'],
        data=test_data.reset_index(drop=True),
        labels=original_test_labels,
        batch_size=batch_size,
        prefix="CC/class_conditioned_embeddings/test",
        encoder=one_hot_encoder,
        device=DEVICE,
        index = i
    )

    print("🎉 Done saving all batches.")


