import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load dataset
df = pd.read_csv('Modified_SQL_Dataset.csv')

# Step 2: Clean queries
def clean_query(query):
    query = str(query).lower()
    query = re.sub(r'\d+', '0', query)
    query = re.sub(r'[^a-z0-9_\'\"= ]', ' ', query)
    query = re.sub(r'\s+', ' ', query).strip()
    return query

df['cleaned_query'] = df['Query'].apply(clean_query)

# Step 3: TF-IDF Feature Extraction
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['cleaned_query']).toarray()
y = df['Label'].values

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

import joblib

# Save model and vectorizer
joblib.dump(model, 'sql_injection_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')


# Step 8: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SQL Injection Detection")
plt.tight_layout()
plt.show()
