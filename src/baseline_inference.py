import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Setup Device & Model
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

print(f"Loading {MODEL_NAME} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(
    device
)

# Label Mapping for RoBERTa (0: Negative, 1: Neutral, 2: Positive)
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

# 2. Load Dataset
df = pd.read_csv("samsung_preprocessed_sentiment_data.csv")
texts = df["cleaned_text"].tolist()

labels = []
scores = []
batch_size = 32

print(f"Running baseline sentiment inference on {len(df)} records...")

# 3. Batch Inference Loop
for i in tqdm(range(0, len(texts), batch_size)):
    batch_texts = texts[i : i + batch_size]
    inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        max_probs, preds = torch.max(probs, dim=-1)

    labels.extend([LABEL_MAP[p.item()] for p in preds])
    scores.extend([p.item() for p in max_probs])

# 4. Save Labeled Dataset
df["predicted_sentiment"] = labels
df["confidence_score"] = scores

df.to_csv("samsung_baseline_labeled_data.csv", index=False)
print("\nInference Complete! Saved to 'samsung_baseline_labeled_data.csv'.")
print("\n--- Sentiment Label Distribution ---")
print(df["predicted_sentiment"].value_counts(normalize=True) * 100)