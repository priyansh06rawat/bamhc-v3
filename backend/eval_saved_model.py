import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

df = pd.read_csv("../data/labeled_data.csv")
df = df.rename(columns={c: c.strip() for c in df.columns})
df = df[["tweet","class"]].dropna()
df["class"] = df["class"].astype(int)
train_df, val_df = train_test_split(df, test_size=0.15, stratify=df["class"], random_state=42)

tokenizer = AutoTokenizer.from_pretrained("saved_model")
model = AutoModelForSequenceClassification.from_pretrained("saved_model")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

y_true = []
y_pred = []

for text, label in zip(val_df['tweet'].tolist(), val_df['class'].tolist()):
    enc = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=128).to(device)
    with torch.no_grad():
        out = model(**enc)
    logits = out.logits.cpu().numpy()[0]
    pred = int(np.argmax(logits))
    y_true.append(int(label))
    y_pred.append(pred)

report = classification_report(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
print(report)
print("Confusion matrix:")
print(cm)
with open("eval_report.txt", "w") as f:
    f.write(report)
    f.write("\nConfusion matrix:\n")
    f.write(str(cm))
print("Saved eval_report.txt")
