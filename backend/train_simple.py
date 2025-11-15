import argparse
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str)
parser.add_argument("--model_name", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--epochs", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--max_length", type=int, default=128)
args = parser.parse_args()

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

df = pd.read_csv(args.data)
df = df.rename(columns={c: c.strip() for c in df.columns})
df = df[["tweet", "class"]].dropna()
df["class"] = df["class"].astype(int)

train_df, val_df = train_test_split(df, test_size=0.15, stratify=df["class"], random_state=seed)

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
        text = str(self.texts[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item

train_dataset = TextDataset(train_df["tweet"].tolist(), train_df["class"].tolist(), tokenizer, args.max_length)
val_dataset = TextDataset(val_df["tweet"].tolist(), val_df["class"].tolist(), tokenizer, args.max_length)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = df["class"].nunique()

model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=num_labels
).to(device)

optimizer = AdamW(model.parameters(), lr=args.lr)
loss_fn = CrossEntropyLoss()

print("Starting training...")

for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss = 0

    print(f"Epoch {epoch} started...")

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}/{len(train_loader)}  Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}")

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")

os.makedirs(args.output_dir, exist_ok=True)
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

print(f"Model saved to: {args.output_dir}")
