import os
import sys
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

INPUT_CSV = "samsung_preprocessed_sentiment_data.csv"
OUTPUT_TOPICS_CSV = "samsung_issue_clusters.csv"

if not os.path.exists(INPUT_CSV):
    print(f"ERROR: Could not find '{INPUT_CSV}'.")
    sys.exit(1)

print(f"Loading dataset: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

TEXT_COL = "cleaned_text" if "cleaned_text" in df.columns else "text"

# 1. Clean English text & remove standard stop words + generic YouTube noise
print("Filtering English text and removing generic noise words...")
custom_stop_words = [
    "the", "and", "to", "is", "this", "it", "of", "in", "for", "on", "you", "that", "my", "with",
    "video", "channel", "subscribe", "link", "intro", "hai", "bhai", "ka", "ke", "ki", "ko", "par"
]

vectorizer_model = CountVectorizer(stop_words=custom_stop_words, min_df=5, ngram_range=(1, 2))

# 2. Extract documents
docs = df[TEXT_COL].dropna().astype(str).tolist()

print("\n--- Running Refined BERTopic Topic Extraction ---")
topic_model = BERTopic(
    vectorizer_model=vectorizer_model,
    nr_topics=10,
    language="english",
    calculate_probabilities=False
)

topics, _ = topic_model.fit_transform(docs)

# 3. Retrieve and inspect refined topic metadata
topic_info = topic_model.get_topic_info()

print("\n=== REFINED COMPLAINT & DISCUSSION CLUSTERS ===")
print(topic_info[["Topic", "Count", "Name"]].to_string(index=False))

topic_info.to_csv(OUTPUT_TOPICS_CSV, index=False)
print(f"\nSUCCESS! Refined topic clusters saved to: {OUTPUT_TOPICS_CSV}")