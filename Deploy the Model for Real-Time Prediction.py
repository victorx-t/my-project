import streamlit as st
import joblib
import re

# Load model and vectorizer
model = joblib.load('sql_injection_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Query cleaning function
def clean_query(query):
    query = str(query).lower()
    query = re.sub(r'\d+', '0', query)
    query = re.sub(r'[^a-z0-9_\'\"= ]', ' ', query)
    query = re.sub(r'\s+', ' ', query).strip()
    return query

# Streamlit UI
st.title("🔐 SQL Injection Detector")
st.write("Enter a SQL query below to check if it's malicious or safe.")

user_input = st.text_area("SQL Query")

if st.button("Check"):
    cleaned = clean_query(user_input)
    transformed = vectorizer.transform([cleaned])
    prediction = model.predict(transformed)[0]

    if prediction == 1:
        st.error("⚠️ This query is likely a **SQL Injection Attack**.")
    else:
        st.success("✅ This query appears to be **safe**.")
