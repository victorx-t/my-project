import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (replace 'path_to_csv' with your local file path)
df = pd.read_csv('Modified_SQL_Dataset.csv')

# Display the first few rows to verify the data loaded correctly
print(df.head())

# Tokenization function: split SQL queries into tokenspython
def tokenize_sql(query):
    tokens = re.findall(r"\b\w+\b", str(query).lower())
    return tokens

df['tokens'] = df['Query'].apply(tokenize_sql)

# Cleaning function
def clean_query(query):
    query = str(query).lower()
    query = re.sub(r'\d+', '0', query)  # Replace digits with 0
    query = re.sub(r'[^a-z0-9_ ]', ' ', query)  # Remove special chars except underscore
    query = re.sub(r'\s+', ' ', query).strip()  # Remove extra spaces
    return query

df['cleaned_query'] = df['Query'].apply(clean_query)

# Feature extraction: example using query length (number of tokens)
df['query_length'] = df['cleaned_query'].apply(lambda x: len(x.split()))

# Normalize numerical feature
scaler = MinMaxScaler()
df['query_length_norm'] = scaler.fit_transform(df[['query_length']])

# Prepare feature matrix X and label vector y
X = df['query_length_norm'].values.reshape(-1, 1)
y = df['Label'].values

# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training: Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - SQL Injection Detection')
plt.show()
