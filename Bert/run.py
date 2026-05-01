import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

def row_to_full_sentence(row):
    try:
        return (
            f"{row['PROTOCOL']} flow from {row['IPV4_SRC_ADDR']}:{row['L4_SRC_PORT']} "
            f"to {row['IPV4_DST_ADDR']}:{row['L4_DST_PORT']}, using L7 protocol {row['L7_PROTO']}. "
            f"Flow transferred {row['IN_BYTES']} bytes in ({row['IN_PKTS']} packets) and "
            f"{row['OUT_BYTES']} bytes out ({row['OUT_PKTS']} packets), lasting {row['FLOW_DURATION_MILLISECONDS']} ms. "
            f"Client duration: {row['DURATION_IN']} ms, Server duration: {row['DURATION_OUT']} ms. "
            f"TCP flags: {row['TCP_FLAGS']} (client: {row['CLIENT_TCP_FLAGS']}, server: {row['SERVER_TCP_FLAGS']}). "
            f"TTL range: {row['MIN_TTL']}–{row['MAX_TTL']}. Packet size range: {row['SHORTEST_FLOW_PKT']}–{row['LONGEST_FLOW_PKT']} bytes. "
            f"IP packet length range: {row['MIN_IP_PKT_LEN']}–{row['MAX_IP_PKT_LEN']} bytes. "
            f"Throughput: {row['SRC_TO_DST_SECOND_BYTES']} Bps src→dst, {row['DST_TO_SRC_SECOND_BYTES']} Bps dst→src. "
            f"Retransmissions: {row['RETRANSMITTED_IN_PKTS']} packets ({row['RETRANSMITTED_IN_BYTES']} bytes) src→dst, "
            f"{row['RETRANSMITTED_OUT_PKTS']} packets ({row['RETRANSMITTED_OUT_BYTES']} bytes) dst→src. "
            f"Avg throughput: {row['SRC_TO_DST_AVG_THROUGHPUT']} bps src→dst, {row['DST_TO_SRC_AVG_THROUGHPUT']} bps dst→src. "
            f"Packet size categories: ≤128: {row['NUM_PKTS_UP_TO_128_BYTES']}, 128–256: {row['NUM_PKTS_128_TO_256_BYTES']}, "
            f"256–512: {row['NUM_PKTS_256_TO_512_BYTES']}, 512–1024: {row['NUM_PKTS_512_TO_1024_BYTES']}, "
            f"1024–1514: {row['NUM_PKTS_1024_TO_1514_BYTES']}. "
            f"TCP win size: in={row['TCP_WIN_MAX_IN']}, out={row['TCP_WIN_MAX_OUT']}. "
            f"ICMP type: {row['ICMP_TYPE']} (IPv4: {row['ICMP_IPV4_TYPE']}). "
            f"DNS query ID: {row['DNS_QUERY_ID']}, type: {row['DNS_QUERY_TYPE']}, TTL: {row['DNS_TTL_ANSWER']}. "
            f"FTP return code: {row['FTP_COMMAND_RET_CODE']}."
        )
    except Exception as e:
        return f"Error formatting row: {e}"


import pandas as pd

# Load NetFlow CSV
df = pd.read_csv("dataset/data.csv")  
df['text'] = df.apply(row_to_full_sentence, axis=1)

# ✅ Final format for BERT
df_bert = df[['text', 'Label']]

print(df_bert.head())

df_train_val, df_test = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)
df_train, df_val = train_test_split(df_train_val, test_size=0.25, stratify=df_train_val['Label'], random_state=42)


df_train = df_train.rename(columns={'Label': 'labels'})
df_val = df_val.rename(columns={'Label': 'labels'})
df_test = df_test.rename(columns={'Label': 'labels'})


# ✅ Convert to Hugging Face Datasets
train_ds = Dataset.from_pandas(df_train)
val_ds = Dataset.from_pandas(df_val)
test_ds = Dataset.from_pandas(df_test)

# ✅ Tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

train_ds = train_ds.map(tokenize_function, batched=True)
val_ds = val_ds.map(tokenize_function, batched=True)
test_ds = test_ds.map(tokenize_function, batched=True)

# ✅ Set format for PyTorch
cols = ['input_ids', 'attention_mask', 'labels']
train_ds.set_format(type='torch', columns=cols)
val_ds.set_format(type='torch', columns=cols)
test_ds.set_format(type='torch', columns=cols)

# ✅ Load BERT model
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# ✅ Training arguments
training_args = TrainingArguments(
    output_dir="./bert-binary-output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_dir="./logs",
    seed=42
)

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer
)

# ✅ Train model
trainer.train()

# ✅ Predict on test set
predictions = trainer.predict(test_ds)
logits = predictions.predictions
labels = predictions.label_ids
probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()  # probability for class 1
preds = (probs > 0.5).astype(int)

# ✅ Evaluation Metrics
print("🔍 Classification Report:\n")
print(classification_report(labels, preds))

print("🔍 Confusion Matrix:")
cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
disp.plot(cmap=plt.cm.Blues)
plt.show()

# ✅ ROC AUC Score
roc_auc = roc_auc_score(labels, probs)
print(f"\n🔍 ROC AUC Score: {roc_auc:.4f}")

# ✅ ROC Curve
fpr, tpr, _ = roc_curve(labels, probs)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


