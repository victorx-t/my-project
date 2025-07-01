import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Load the metrics and confusion matrix CSVs
metrics_file = "C:/Users/USER/Desktop/DATASET/pretest_metrics.csv"
confusion_file = "C:/Users/USER/Desktop/DATASET/pretest_confusion_matrix.csv"

try:
    # Load files
    metrics_df = pd.read_csv(metrics_file)
    confusion_df = pd.read_csv(confusion_file)

    # Section: Model Performance
    st.subheader("📊 Model Performance Metrics")
    for _, row in metrics_df.iterrows():
        st.metric(label=row["Metric"], value=f"{row['Score']:.2f}%")

    # Section: Confusion Matrix
    st.markdown("---")
    st.subheader("🧩 Confusion Matrix Results")

    # Show as a table
    st.dataframe(confusion_df)

    # Optional: Plot as a bar chart
    fig, ax = plt.subplots()
    sns.barplot(data=confusion_df, x="Category", y="Count", palette="Set2", ax=ax)
    ax.set_title("Confusion Matrix Breakdown")
    ax.set_xlabel("Classification Type")
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

except FileNotFoundError:
    st.warning("⚠️ Metrics or confusion matrix file not found. Run the model training script first.")
