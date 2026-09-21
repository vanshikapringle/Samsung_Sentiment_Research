import os
import sys
import torch
import pandas as pd
from transformers import pipeline

INPUT_CSV = "samsung_preprocessed_sentiment_data.csv"
OUTPUT_CSV = "samsung_absa_enriched.csv"

if not os.path.exists(INPUT_CSV):
    print(f"ERROR: Could not find '{INPUT_CSV}'.")
    sys.exit(1)

print(f"Loading dataset from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

TEXT_COL = None
for col in ["cleaned_text", "text", "comment_text", "post_text", "full_text"]:
    if col in df.columns:
        TEXT_COL = col
        break

if not TEXT_COL:
    print(f"ERROR: Text column not found.")
    sys.exit(1)

print(f"Using text column: '{TEXT_COL}' ({len(df):,} rows).")

# Detect GPU / CUDA acceleration
device = 0 if torch.cuda.is_available() else -1
print(f"Using Device: {'GPU (CUDA)' if device == 0 else 'CPU'}")

print("\nLoading Hugging Face DeBERTa ABSA Pipeline...")
absa_classifier = pipeline(
    "text-classification",
    model="yangheng/deberta-v3-base-absa-v1.1",
    tokenizer="yangheng/deberta-v3-base-absa-v1.1",
    device=device,
    batch_size=32 if device == 0 else 16,
)

ASPECT_KEYWORDS = {
    "battery": ["battery", "charge", "charging", "drain", "sot", "power"],
    "camera": ["camera", "photo", "video", "lens", "zoom", "sensor", "picture"],
    "display": ["display", "screen", "crease", "panel", "hz", "bright", "oled"],
    "processor": ["exynos", "snapdragon", "chip", "processor", "heat", "hot", "thermal", "throttle", "fps", "lag"],
    "build quality": ["build", "hinge", "durability", "crack", "scratch", "glass", "stain", "peeling"],
    "price": ["price", "cost", "expensive", "cheap", "value", "deal", "discount", "worth"],
}

print("\n--- Starting Fast Aspect Extraction ---")

for aspect, keywords in ASPECT_KEYWORDS.items():
    col_name = f"aspect_{aspect.replace(' ', '_')}"
    print(f"Processing aspect: '{aspect}'...")

    # Default all rows to Neutral
    df[col_name] = "Neutral"

    # Find rows containing aspect keywords
    pattern = "|".join(keywords)
    matching_mask = df[TEXT_COL].str.contains(pattern, case=False, na=False)
    matching_indices = df[matching_mask].index

    if len(matching_indices) > 0:
        matching_texts = df.loc[matching_indices, TEXT_COL].astype(str).str[:512].tolist()
        pairs = [{"text": t, "text_pair": aspect} for t in matching_texts]

        # Run batch prediction on relevant subset
        results = absa_classifier(pairs)
        predictions = [res["label"] for res in results]

        df.loc[matching_indices, col_name] = predictions
        print(f"  -> Evaluated {len(matching_indices):,} relevant comments.")
    else:
        print("  -> No matching comments found.")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSUCCESS! Saved to: {OUTPUT_CSV}")