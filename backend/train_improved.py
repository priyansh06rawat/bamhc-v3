import argparse
import os
import random
import json
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="../data/labeled_data.csv")
parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
parser.add_argument("--output_dir", type=str, default="saved_model")
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--max_length", type=int, default=128)
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def clean_text(text):
    text = str(text)
    text = text.replace("RT @", "")
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9\s,.!?\'\"-]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

df = pd.read_csv(args.data)
df = df.rename(columns={c: c.strip() for c in df.columns})
if "tweet" not in df.columns or "class" not in df.columns:
    raise SystemExit("CSV must contain 'tweet' and 'class' columns")
df = df[["tweet","class"]].dropna()
df["class"] = df["class"].astype(int)
df["tweet"] = df["tweet"].apply(clean_text)

train_df, val_df = train_test_split(df, test_size=0.15, stratify=df["class"], random_state=args.seed)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        t = str(self.texts[idx])
        enc = self.tokenizer(t, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item

train_dataset = TextDataset(train_df["tweet"].tolist(), train_df["class"].tolist(), tokenizer, args.max_length)
val_dataset = TextDataset(val_df["tweet"].tolist(), val_df["class"].tolist(), tokenizer, args.max_length)

class_counts = train_df["class"].value_counts().to_dict()
num_samples = len(train_dataset)
weights = [1.0 / class_counts[label] for label in train_df["class"].tolist()]
weights = torch.DoubleTensor(weights)
sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = int(df["class"].nunique())

model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)
model.to(device)

optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
total_steps = len(train_loader) * args.epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

best_f1 = 0.0
os.makedirs(args.output_dir, exist_ok=True)

for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(train_loader, 1):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        if step % 100 == 0:
            print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")
    avg_loss = total_loss / max(1, len(train_loader))
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f} Val Acc: {acc:.4f} Val Macro F1: {macro_f1:.4f}")
    if macro_f1 > best_f1:
        best_f1 = macro_f1
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        with open(os.path.join(args.output_dir, "best_metrics.json"), "w") as f:
            json.dump({"epoch": epoch, "val_acc": acc, "val_macro_f1": macro_f1, "avg_loss": avg_loss}, f)
        print(f"New best model saved to {args.output_dir} (f1={best_f1:.4f})")
print("Training complete")
