import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

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

# Step 3: Tokenization (Optional for debugging)
def tokenize_sql(query):
    return re.findall(r"\b\w+\b", str(query).lower())

df['tokens'] = df['cleaned_query'].apply(tokenize_sql)

# Step 4: TF-IDF Feature Extraction
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['cleaned_query']).toarray()

# Step 5: Extract labels
y = df['Label'].values

# Output shape
print(f"TF-IDF Feature Matrix Shape: {X.shape}")
print(f"Label Vector Shape: {y.shape}")
