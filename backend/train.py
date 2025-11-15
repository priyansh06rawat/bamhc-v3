import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="../data/labeled_data.csv")
parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
parser.add_argument("--output_dir", type=str, default="saved_model")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--max_length", type=int, default=128)
args = parser.parse_args()

df = pd.read_csv(args.data)
df = df.rename(columns={col: col.strip() for col in df.columns})
if "tweet" not in df.columns or "class" not in df.columns:
    raise SystemExit("CSV must contain 'tweet' and 'class' columns")
df = df[["tweet", "class"]].dropna()
df["class"] = df["class"].astype(int)
train_df, val_df = train_test_split(df, test_size=0.15, stratify=df["class"], random_state=42)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
def tokenize_function(ex):
    return tokenizer(ex["tweet"], padding="max_length", truncation=True, max_length=args.max_length)

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))
train_ds = train_ds.map(tokenize_function, batched=True)
val_ds = val_ds.map(tokenize_function, batched=True)
if "label" not in train_ds.column_names:
    train_ds = train_ds.rename_column("class", "label")
    val_ds = val_ds.rename_column("class", "label")
train_ds.set_format(type="torch", columns=["input_ids","attention_mask","label"])
val_ds.set_format(type="torch", columns=["input_ids","attention_mask","label"])

num_labels = int(df["class"].nunique())
model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_score(p.label_ids, preds)
    report = classification_report(p.label_ids, preds, output_dict=True)
    return {"accuracy": acc, "macro_f1": report["macro avg"]["f1-score"]}

training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=args.lr,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    save_total_limit=3
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
metrics = trainer.evaluate()
print(metrics)
trainer.save_model(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
