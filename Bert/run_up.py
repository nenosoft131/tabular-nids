import time
import pandas as pd
import torch
import os
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, precision_recall_curve, auc, accuracy_score,
    precision_score, recall_score, f1_score
)
from transformers import BertForSequenceClassification,BertTokenizer, Trainer, TrainingArguments
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datasets import Dataset


results = []

# 🔁 Define list of hyperparameter configurations to try
hyperparameter_sets = [
    # {"learning_rate": 1e-5, "batch_size": 16, "epochs": 3},
    {"learning_rate": 1e-5, "batch_size": 32, "epochs": 4},
    
    # {"learning_rate": 2e-5, "batch_size": 16, "epochs": 3},
    # {"learning_rate": 3e-5, "batch_size": 32, "epochs": 4},

    # {"learning_rate": 5e-5, "batch_size": 16, "epochs": 3},
    # {"learning_rate": 5e-5, "batch_size": 32, "epochs": 4},
]





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
df = pd.read_csv("dataset/NF-CSE-CIC-IDS2018-v2/data/NF-CSE-CIC-IDS2018-v2.csv")  
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



for idx, hp in enumerate(hyperparameter_sets):
    print(f"🚀 Starting run {idx+1} with hyperparameters: {hp}")

    # ✅ Load model
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # ✅ Define training arguments
    training_args = TrainingArguments(
        output_dir=f"./bert-output-run{idx+1}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=hp["learning_rate"],
        per_device_train_batch_size=hp["batch_size"],
        per_device_eval_batch_size=hp["batch_size"],
        num_train_epochs=hp["epochs"],
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

    # ⏱️ Train
    start_fit = time.time()
    trainer.train()
    end_fit = time.time()
    time_fit = round(end_fit - start_fit, 2)

    # ⏱️ Predict
    start_infer = time.time()
    val_pred = trainer.predict(val_ds)
    test_pred = trainer.predict(test_ds)
    end_infer = time.time()
    time_inference = round(end_infer - start_infer, 2)

    def evaluate(predictions):
        logits = predictions.predictions
        labels = predictions.label_ids
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
        preds = (probs > 0.5).astype(int)

        # Basic Metrics
        aucroc = roc_auc_score(labels, probs)
        precision, recall, _ = precision_recall_curve(labels, probs)
        aucpr = auc(recall, precision)
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds)
        rec = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        f1_macro = f1_score(labels, preds, average='macro')
        f1_micro = f1_score(labels, preds, average='micro')
        cm = confusion_matrix(labels, preds)

        return {
            "aucroc": aucroc,
            "aucpr": aucpr,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "confusion_matrix": cm.tolist()  # Store as list for JSON compatibility
        }

    print(val_pred)
    print(test_pred)
    # val_result = evaluate(val_pred)
    test_result = evaluate(test_pred)

    # 📦 Save results
    result = {
        'model_name': model_name,
        'params': str(hp),
        'config': str(model.config.to_dict()),

        # 'AUCROC-Val': val_result["aucroc"],
        # 'AUCPR-Val': val_result["aucpr"],
        # 'Accuracy-Val': val_result["accuracy"],
        # 'Precision-Val': val_result["precision"],
        # 'Recall-Val': val_result["recall"],
        # 'F1-Val': val_result["f1"],
        # 'F1-Macro-Val': val_result["f1_macro"],
        # 'F1-Micro-Val': val_result["f1_micro"],
        # 'Confusion-Matrix-Val': val_result["confusion_matrix"],

        'AUCROC-Test': test_result["aucroc"],
        'AUCPR-Test': test_result["aucpr"],
        'Accuracy-Test': test_result["accuracy"],
        'Precision-Test': test_result["precision"],
        'Recall-Test': test_result["recall"],
        'F1-Test': test_result["f1"],
        'F1-Macro-Test': test_result["f1_macro"],
        'F1-Micro-Test': test_result["f1_micro"],
        'Confusion-Matrix-Test': test_result["confusion_matrix"],

        'TIME_FIT': time_fit,
        'TIME_INFER': time_inference
    }

    results.append(result)

# ✅ Save to CSV
# df_AUCROC = pd.DataFrame(results)
# df_AUCROC.to_csv("bert_hp_results.csv", index=False)
file_path='bert_hp_results.csv'

write_header = not os.path.exists(file_path)

df_AUCROC = pd.DataFrame(results)
df_AUCROC.to_csv(file_path, mode='a', index=False, header=write_header)
print("📁 Results saved to bert_hp_results.csv")
