import numpy as np
from logger_config import get_logger
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

def tokenize_function(example, tokenizer=None):    
    return tokenizer(
        example["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=128
    )
class BOD:
    def __init__(self, seed, model_name, device=None):
        self.seed = seed
        self.model_name = "bert-base-uncased"
        # self.model_dict = {'Bert':NearestNeighbors} # default value; will be overridden by config
        self.logger = get_logger(__name__)
        self.trainer = None
        print(f"Using model: {self.model_name}")

    def fit(self, train_ds, val_ds, **kwargs):
        # model_name = "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(self.model_name)
        
        train_ds = train_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
        val_ds = val_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
        
        cols = ['input_ids', 'attention_mask', 'labels']
        train_ds.set_format(type='torch', columns=cols)
        val_ds.set_format(type='torch', columns=cols)
        
        # ✅ Load BERT model
        self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=2)

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
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer
        )
        self.trainer.train()
        return self

    def predict_score(self, test_ds):
        predictions = self.trainer.predict(test_ds)
        logits = predictions.predictions
        labels = predictions.label_ids
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()  # probability for class 1
        preds = (probs > 0.5).astype(int)
        return preds, probs
