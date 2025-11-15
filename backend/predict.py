import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="saved_model")
parser.add_argument("--text", type=str, default=None)
parser.add_argument("--file", type=str, default=None)
args = parser.parse_args()

label_map = {0: "hate", 1: "offensive", 2: "neither"}

tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_batch(texts):
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
    enc = {k:v.to(device) for k,v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits.cpu().numpy()
    preds = np.argmax(logits, axis=1)
    return [label_map[int(p)] for p in preds]

if args.text:
    result = predict_batch([args.text])[0]
    print(result)
elif args.file:
    df = pd.read_csv(args.file)
    if "tweet" not in df.columns:
        raise SystemExit("CSV must contain a 'tweet' column")
    texts = df["tweet"].astype(str).tolist()
    preds = predict_batch(texts)
    out = df.copy()
    out["prediction"] = preds
    out.to_csv("predictions.csv", index=False)
    print("Saved predictions.csv")
else:
    print("Provide --text or --file")
