import re
import pandas as pd


def clean_social_text(text):
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Normalize whitespaces and newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Load master dataset
df = pd.read_csv("samsung_master_sentiment_dataset.csv")

# Apply cleaning
df["cleaned_text"] = df["text"].apply(clean_social_text)

# Drop rows with text shorter than 3 characters post-cleaning
df = df[df["cleaned_text"].str.len() > 3].reset_index(drop=True)

# Save cleaned CSV
df.to_csv("samsung_preprocessed_sentiment_data.csv", index=False)
print(
    f"Preprocessing complete. {len(df)} cleaned rows ready for model inference & training!"
)