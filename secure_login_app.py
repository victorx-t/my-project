import streamlit as st
import joblib
import re

# Load trained SQLi detection model and vectorizer
model = joblib.load('sql_injection_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Simulated database
fake_db = {
    "admin": "admin123",
    "user1": "pass123",
    "jane": "qwerty",
}

# Query cleaning function (same as training phase)
def clean_query(query):
    query = str(query).lower()
    query = re.sub(r'\d+', '0', query)
    query = re.sub(r'[^a-z0-9_\'\"= ]', ' ', query)
    query = re.sub(r'\s+', ' ', query).strip()
    return query

# SQLi Detection function
def is_sql_injection(query_text):
    cleaned = clean_query(query_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return prediction == 1  # True if it's an injection

# Streamlit UI
st.title("🔐 Secure Login System with SQL Injection Detection")

username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login"):
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"

    # Check for SQL injection
    if is_sql_injection(query):
        st.error("🚨 SQL Injection Detected! Access Denied.")
    else:
        # Simulate DB check
        if username in fake_db and fake_db[username] == password:
            st.success(f"✅ Welcome, {username}!")
        else:
            st.warning("❌ Invalid username or password.")
