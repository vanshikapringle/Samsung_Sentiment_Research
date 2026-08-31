import pandas as pd

# Load generated dataset
df = pd.read_csv("youtube_samsung_sentiment_data.csv")

# 1. Inspect shape and missing values
print("--- Data Shape ---")
print(df.shape)

print("\n--- Missing Values ---")
print(df.isnull().sum())

# 2. Check temporal distribution
df["created_at"] = pd.to_datetime(df["created_at"])
print(
    f"\n--- Date Range ---\n{df['created_at'].min()} to {df['created_at'].max()}"
)

# 3. View comment counts per video source
print("\n--- Record Count per Video ---")
print(df["source_title"].value_counts())

# 4. Preview sample rows
print("\n--- Sample Rows ---")
print(df[["created_at", "text", "engagement_score"]].head())