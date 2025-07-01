import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Load dataset
df = pd.read_csv("Modified_SQL_Dataset.csv")

# Clean SQL queries
def clean_query(query):
    query = str(query).lower()
    query = re.sub(r"\d+", "0", query)
    query = re.sub(r"[^a-z0-9_\'\"= ]", " ", query)
    query = re.sub(r"\s+", " ", query).strip()
    return query

df["cleaned_query"] = df["Query"].apply(clean_query)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df["cleaned_query"]).toarray()
y = df["Label"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred) * 100
recall = recall_score(y_test, y_pred) * 100
f1 = f1_score(y_test, y_pred) * 100

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Print classification report
print("\nPOST-TEST EVALUATION RESULTS")
print(f"Accuracy : {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall   : {recall:.2f}%")
print(f"F1-Score : {f1:.2f}%\n")
print(classification_report(y_test, y_pred))

# Save metrics to CSV
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Score": [accuracy, precision, recall, f1]
})
metrics_df.to_csv("posttest_metrics.csv", index=False)

# Save confusion matrix to CSV
confusion_df = pd.DataFrame({
    "Category": ["True Positives", "True Negatives", "False Positives", "False Negatives"],
    "Count": [tp, tn, fp, fn]
})
confusion_df.to_csv("posttest_confusion_matrix.csv", index=False)

# === Visuals ===

# 1. Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malicious"], yticklabels=["Benign", "Malicious"])
plt.title("Confusion Matrix - SQL Injection Detection")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig("confusion_matrix_heatmap.png")  # optional
plt.show()

# 2. Bar Chart of Metrics
plt.figure(figsize=(7, 5))
sns.barplot(x=metrics_df["Metric"], y=metrics_df["Score"], palette="viridis")
plt.ylim(0, 105)
plt.title("Post-Test Evaluation Metrics")
plt.ylabel("Score (%)")
for i, v in enumerate(metrics_df["Score"]):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center')
plt.tight_layout()
plt.savefig("evaluation_metrics_barplot.png")  # optional
plt.show()
