from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from detoxify import Detoxify
import torch

class Input(BaseModel):
    text: str

print("Loading models...")
device = 0 if torch.cuda.is_available() else -1

# Model 1: Detoxify (for general toxicity)
detoxify_model = Detoxify('original')

# Model 2: RoBERTa hate speech (for hate/offensive content)
hate_classifier = pipeline(
    "text-classification",
    model="facebook/roberta-hate-speech-dynabench-r4-target",
    device=device
)

print("All models loaded!")

app = FastAPI()

def predict_one(text):
    """
    Use ensemble of models for better accuracy
    """
    # Get Detoxify scores
    detox_results = detoxify_model.predict(text)
    
    # Get RoBERTa hate speech score
    hate_results = hate_classifier(text)[0]
    hate_label = hate_results['label'].lower()
    hate_score = hate_results['score'] if 'hate' in hate_label and 'not' not in hate_label else 0
    
    # Combine scores
    toxicity = detox_results['toxicity']
    severe_toxicity = detox_results['severe_toxicity']
    threat = detox_results['threat']
    insult = detox_results['insult']
    identity_attack = detox_results['identity_attack']
    obscene = detox_results['obscene']
    
    # Calculate combined score
    max_detox = max(toxicity, severe_toxicity, threat, identity_attack)
    combined_score = max(hate_score, max_detox)
    
    # Classification with ensemble logic
    if hate_score > 0.6 or threat > 0.5 or severe_toxicity > 0.5 or combined_score > 0.7:
        label = "hate"
        score = combined_score
    elif hate_score > 0.3 or obscene > 0.4 or insult > 0.4 or identity_attack > 0.3 or toxicity > 0.4:
        label = "offensive"
        score = max(hate_score, obscene, insult, identity_attack, toxicity)
    else:
        label = "neither"
        score = 1.0 - max(hate_score, toxicity)
    
    return {
        "label": label,
        "score": float(score)
    }

@app.post("/predict")
def predict(payload: Input):
    return predict_one(payload.text)

@app.get("/")
def root():
    return {
        "message": "Hate Speech Classifier API",
        "model": "Ensemble (Detoxify + RoBERTa)",
        "status": "running"
    }