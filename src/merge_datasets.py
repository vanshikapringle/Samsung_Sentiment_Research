import pandas as pd

# 1. Load datasets
yt_df = pd.read_csv("youtube_samsung_sentiment_data.csv")
bsky_df = pd.read_csv("bluesky_samsung_sentiment_data.csv")

# 2. Align and merge
master_df = pd.concat([yt_df, bsky_df], ignore_index=True)

# 3. Standardize timestamps & deduplicate by ID
master_df["created_at"] = pd.to_datetime(master_df["created_at"])
master_df = master_df.sort_values(by="created_at").reset_index(drop=True)
master_df = master_df.drop_duplicates(subset=["id"]).reset_index(drop=True)

# 4. Save Master Dataset
master_df.to_csv("samsung_master_sentiment_dataset.csv", index=False)

print("--- Master Dataset Ready ---")
print(f"Total Rows Saved: {len(master_df)}")
print("\nBreakdown by Platform:")
print(master_df["platform"].value_counts())