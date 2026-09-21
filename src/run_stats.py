import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

# Path to the ABSA enriched dataset
INPUT_CSV = "samsung_absa_enriched.csv" if os.path.exists("samsung_absa_enriched.csv") else "samsung_preprocessed_sentiment_data.csv"
OUTPUT_STATS_TXT = "samsung_statistical_findings.txt"

print(f"Loading dataset from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

# Identify date/time column
DATE_COL = None
for col in ["created_at", "date", "timestamp", "time"]:
    if col in df.columns:
        DATE_COL = col
        break

if not DATE_COL:
    print(f"ERROR: Could not find date column. Available columns: {list(df.columns)}")
    sys.exit(1)

# Convert to datetime and aggregate monthly
df['datetime'] = pd.to_datetime(df[DATE_COL], errors='coerce')
df['year_month'] = df['datetime'].dt.to_period('M')

# Identify platform column
PLATFORM_COL = "platform" if "platform" in df.columns else None

print("\n--- Running Cross-Platform Sentiment Correlation ---")

if PLATFORM_COL and len(df[PLATFORM_COL].unique()) > 1:
    # Group sentiment by month and platform
    monthly_platform = df.groupby(['year_month', PLATFORM_COL]).size().unstack(fill_value=0)
    
    platforms = monthly_platform.columns.tolist()
    p1, p2 = platforms[0], platforms[1]
    
    # Calculate Pearson Correlation Coefficient
    r_val, p_val = stats.pearsonr(monthly_platform[p1], monthly_platform[p2])
    print(f"\nPearson Correlation ({p1} vs {p2}): r = {r_val:.4f} (p-value = {p_val:.4e})")
    
    # Cross-Correlation Lag Analysis
    lags = [0, 1, 2, -1, -2]
    print("\nCross-Correlation Time-Lag Alignment:")
    for lag in lags:
        if lag > 0:
            s1 = monthly_platform[p1].iloc[lag:]
            s2 = monthly_platform[p2].iloc[:-lag]
        elif lag < 0:
            s1 = monthly_platform[p1].iloc[:lag]
            s2 = monthly_platform[p2].iloc[-lag:]
        else:
            s1 = monthly_platform[p1]
            s2 = monthly_platform[p2]
        
        corr = np.corrcoef(s1, s2)[0, 1]
        print(f"  -> Lag {lag:+d} month(s): r = {corr:.4f}")

else:
    print("Single platform dataset or platform column missing. Generating temporal trend statistics...")
    monthly_trend = df.groupby('year_month').size()
    print("\nMonthly Post Activity Distribution:")
    print(monthly_trend.head(12))

print("\n--- Summary Report Generated Successfully ---")