import time
import pandas as pd
import torch
import os
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, precision_recall_curve, auc, accuracy_score,
    precision_score, recall_score, f1_score
)
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datasets import Dataset


# 🔁 Define hyperparameter configurations
hyperparameter_sets = [
    # {"learning_rate": 1e-5, "batch_size": 32, "epochs": 4},
    # {"learning_rate": 3e-5, "batch_size": 32, "epochs": 4}
    {"learning_rate": 5e-05, "batch_size": 32, 'epochs': 4}
]

results = []

# 🧠 Sentence builder from NetFlow-like data
def row_to_full_sentence(row):
    try:
        return (
            f"Flow on {row['Date first seen']} with duration {row['Duration']} seconds. "
            f"Protocol: {row['Proto']}, from {row['Src IP Addr']}:{row['Src Pt']} "
            f"to {row['Dst IP Addr']}:{row['Dst Pt']}. "
            f"{row['Packets']} packets sent with {row['Bytes']} bytes. "
            f"Flow flags: {row['Flags']}, TOS: {row['Tos']}."
        )
    except Exception as e:
        return f"Error formatting row: {e}"

# 🧼 Clean for PyArrow
def clean_dataframe(df):
    for col in df.columns:
        if col not in ['text', 'labels']:
            df[col] = df[col].astype(str)
    return df

# 📥 Load dataset
df = pd.read_csv("dataset/CIDDS/CIDDS-001-internal-week1.csv", dtype=str, low_memory=False)

# 🏷️ Convert 'class' to binary label
df['labels'] = df['class'].apply(lambda x: 0 if str(x).strip().lower() == 'normal' else 1)

# ✍️ Create textual representation
df['text'] = df.apply(row_to_full_sentence, axis=1)

# 🧪 Train/Val/Test split
df_train_val, df_test = train_test_split(df, test_size=0.2, stratify=df['labels'], random_state=42)
df_train, df_val = train_test_split(df_train_val, test_size=0.25, stratify=df_train_val['labels'], random_state=42)

# ✅ Ensure proper types
df_train = df_train[['text', 'labels']].copy()
df_val = df_val[['text', 'labels']].copy()
df_test = df_test[['text', 'labels']].copy()

df_train['text'] = df_train['text'].astype(str)
df_val['text'] = df_val['text'].astype(str)
df_test['text'] = df_test['text'].astype(str)

df_train['labels'] = df_train['labels'].astype(int)
df_val['labels'] = df_val['labels'].astype(int)
df_test['labels'] = df_test['labels'].astype(int)

# ✅ Now convert
train_ds = Dataset.from_pandas(df_train.reset_index(drop=True))
val_ds = Dataset.from_pandas(df_val.reset_index(drop=True))
test_ds = Dataset.from_pandas(df_test.reset_index(drop=True))


# 🧠 Tokenization
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

train_ds = train_ds.map(tokenize_function, batched=True)
val_ds = val_ds.map(tokenize_function, batched=True)
test_ds = test_ds.map(tokenize_function, batched=True)

# 🧱 Set format for PyTorch
train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# 🚀 Training loop
for idx, hp in enumerate(hyperparameter_sets):
    print(f"\n🚀 Starting run {idx+1} with hyperparameters: {hp}")

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer
    )

    # ⏱️ Training
    start_fit = time.time()
    trainer.train()
    end_fit = time.time()
    time_fit = round(end_fit - start_fit, 2)

    # ⏱️ Inference
    start_infer = time.time()
    val_pred = trainer.predict(val_ds)
    test_pred = trainer.predict(test_ds)
    end_infer = time.time()
    time_inference = round(end_infer - start_infer, 2)

    # 📊 Evaluation
    def evaluate(predictions):
        logits = predictions.predictions
        labels = predictions.label_ids
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
        preds = (probs > 0.5).astype(int)

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
            "confusion_matrix": cm.tolist()
        }

    test_result = evaluate(test_pred)

    result = {
        'model_name': model_name,
        'params': str(hp),
        'config': str(model.config.to_dict()),

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

# 🧾 Save results
file_path = 'bert_hp_results.csv'
write_header = not os.path.exists(file_path)

df_AUCROC = pd.DataFrame(results)
df_AUCROC.to_csv(file_path, mode='a', index=False, header=write_header)
print("📁 Results saved to bert_hp_results.csv")
