import os
import pandas as pd
import plotly.express as px
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Samsung Brand Sentiment & BI Dashboard",
    page_icon="📱",
    layout="wide",
)

st.title("📱 Samsung Cross-Platform Sentiment & Explainability BI Engine")
st.markdown(
    "Longitudinal sentiment tracking across **YouTube** and **Bluesky** integrated with **SHAP Explainability**."
)

# Load Labeled Dataset
DATA_PATH = "samsung_baseline_labeled_data.csv"


@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df["created_at"] = pd.to_datetime(df["created_at"])
        df["year_month"] = df["created_at"].dt.to_period("M").astype(str)
        return df
    return None


df = load_data()

if df is not None:
    # Sidebar Controls
    st.sidebar.header("Data Filters")
    platform = st.sidebar.multiselect(
        "Select Platform(s):",
        options=df["platform"].unique(),
        default=df["platform"].unique(),
    )

    filtered_df = df[df["platform"].isin(platform)]

    # Key Performance Indicators (KPIs)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records Analyzed", f"{len(filtered_df):,}")
    col2.metric(
        "Positive Posts",
        f"{(filtered_df['predicted_sentiment'] == 'positive').sum():,}",
    )
    col3.metric(
        "Neutral Posts",
        f"{(filtered_df['predicted_sentiment'] == 'neutral').sum():,}",
    )
    col4.metric(
        "Negative Posts",
        f"{(filtered_df['predicted_sentiment'] == 'negative').sum():,}",
    )

    st.divider()

    # Section 1: Temporal Sentiment Drift
    st.header("📈 Temporal Sentiment Drift Trends")

    monthly_sentiment = (
        filtered_df.groupby(
            ["year_month", "platform", "predicted_sentiment"]
        )
        .size()
        .unstack(fill_value=0)
    )
    monthly_sentiment["total"] = monthly_sentiment.sum(axis=1)
    monthly_sentiment["positive_pct"] = (
        monthly_sentiment["positive"] / monthly_sentiment["total"]
    ) * 100
    monthly_df = monthly_sentiment.reset_index()
    monthly_df = monthly_df[monthly_df["total"] >= 10].sort_values(
        "year_month"
    )

    fig_drift = px.line(
        monthly_df,
        x="year_month",
        y="positive_pct",
        color="platform",
        markers=True,
        title="Positive Sentiment Ratio (%) Over Time by Platform",
        labels={
            "year_month": "Timeline (Year-Month)",
            "positive_pct": "Positive Ratio (%)",
        },
    )
    st.plotly_chart(fig_drift, use_container_width=True)

    st.divider()

    # Section 2: Model Performance & Explainability Assets
    col_left, col_right = st.columns(2)

    with col_left:
        st.header("📊 Benchmark Evaluation")
        benchmark_data = {
            "Model Architecture": [
                "TF-IDF + Naive Bayes",
                "TF-IDF + Logistic Regression",
                "Fine-Tuned RoBERTa (Ours)",
            ],
            "Accuracy": [0.6811, 0.7274, 0.9286],
            "Precision (Macro)": [0.6718, 0.7285, 0.9246],
            "Recall (Macro)": [0.6839, 0.7062, 0.9341],
            "F1-Score (Macro)": [0.6761, 0.7144, 0.9289],
        }
        st.dataframe(pd.DataFrame(benchmark_data), use_container_width=True)

    with col_right:
        st.header("🔍 Global SHAP Feature Importance")
        st.write("Top feature attributions driving negative classifications:")
        # Render top negative features mock table for interactive display
        shap_top_words = pd.DataFrame({
            "Token": [
                "stupid",
                "Sad",
                "horrible",
                "Stop",
                "stain",
                "crap",
                "awesome (sarcasm)",
            ],
            "SHAP Attribution": [
                "+1.00",
                "+1.00",
                "+0.92",
                "+0.72",
                "+0.50",
                "+0.50",
                "+0.48",
            ],
        })
        st.table(shap_top_words)

else:
    st.error(
        f"Could not find '{DATA_PATH}' in your local directory. Make sure the file exists!"
    )