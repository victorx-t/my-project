import pandas as pd
import re

# Load your dataset (replace with your actual CSV path)
df = pd.read_csv('Modified_SQL_Dataset.csv')  # Ensure it has 'Query' and 'Label' columns

# Preview the first few rows
print("Original Data Sample:\n", df.head())

# Step 1: Lowercase conversion
def to_lowercase(query):
    return str(query).lower()

# Step 2: Remove digits and special characters (basic cleaning)
def clean_query(query):
    query = str(query).lower()
    query = re.sub(r'\d+', '0', query)                     # Replace digits with 0
    query = re.sub(r'[^a-z0-9_\'\" ]+', ' ', query)        # Keep alphanum, underscore, quotes
    query = re.sub(r'\s+', ' ', query).strip()             # Remove multiple spaces
    return query

# Step 3: Tokenization
def tokenize_query(query):
    return re.findall(r'\b\w+\b', query)  # Extract words (alphanumerics)

# Apply preprocessing functions
df['lowercase_query'] = df['Query'].apply(to_lowercase)
df['cleaned_query'] = df['Query'].apply(clean_query)
df['tokens'] = df['cleaned_query'].apply(tokenize_query)

# Print sample results
print("\nPreprocessed Data Sample:")
print(df[['Query', 'cleaned_query', 'tokens']].head())
