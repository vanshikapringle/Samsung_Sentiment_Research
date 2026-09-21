import os
import sys
import pandas as pd

INPUT_CSV = "samsung_absa_enriched.csv"
OUTPUT_ERRORS_CSV = "samsung_roberta_error_audit.csv"

if not os.path.exists(INPUT_CSV):
    print(f"ERROR: Could not find '{INPUT_CSV}'.")
    sys.exit(1)

df = pd.read_csv(INPUT_CSV)

# Identify prediction and ground truth / baseline columns
TEXT_COL = "cleaned_text" if "cleaned_text" in df.columns else "text"
PRED_COL = "pred_label" if "pred_label" in df.columns else ("sentiment" if "sentiment" in df.columns else None)

print(f"Loading dataset from: {INPUT_CSV}")

# Categorize common linguistic failure modes using regex heuristics
def categorize_failure(text):
    text_lower = str(text).lower()
    if any(w in text_lower for w in ["not bad", "never fails", "hardly", "barely", "no problem"]):
        return "Double Negation / Litotes"
    elif any(w in text_lower for w in ["lol", "lmao", "surely", "yeah right", "thanks samsung", "great job"]):
        return "Sarcasm / Irony Bias"
    elif any(w in text_lower for w in ["sot", "exynos", "throttling", "oneui", "bloatware"]):
        return "Domain Jargon / Abbreviation"
    elif any(w in text_lower for w in ["but", "however", "although", "except", "despite"]):
        return "Mixed-Sentiment Dilution"
    else:
        return "Ambiguous Context"

print("\n--- Running Residual Error Categorization Audit ---")
df['error_category'] = df[TEXT_COL].apply(categorize_failure)

error_summary = df['error_category'].value_counts().reset_index()
error_summary.columns = ['Linguistic Error Category', 'Document Count']

print("\n=== ROBERTA LINGUISTIC FAILURE MODE DISTRIBUTION ===")
print(error_summary.to_string(index=False))

error_summary.to_csv(OUTPUT_ERRORS_CSV, index=False)
print(f"\nSUCCESS! Error breakdown saved to: {OUTPUT_ERRORS_CSV}")