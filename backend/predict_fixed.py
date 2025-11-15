import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="saved_model")
parser.add_argument("--text", type=str, default=None)
args = parser.parse_args()

label_map = {0: "hate", 1: "offensive", 2: "neither"}

tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_one(text):
    enc = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=128)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits.cpu().numpy()[0]
    probs = np.exp(logits) / np.sum(np.exp(logits))
    pred = int(np.argmax(logits))
    return {"label": label_map[pred], "score": float(np.max(probs)), "probs": probs.tolist()}

if args.text:
    print(predict_one(args.text))
else:
    print("provide --text")
