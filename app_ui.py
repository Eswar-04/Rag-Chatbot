import streamlit as st
import requests
import os

st.set_page_config(page_title="AI PDF Assistant", page_icon="📘", layout="centered")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.markdown("## Chatbot")
st.markdown("Ask any question related to **Artificial Intelligence.pdf**")

question = st.text_input("Ask a question (text or image related)", "")
mode = st.radio("Select question type:", ["Text", "Image"], horizontal=True)

if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/ask",
                json={"query": question, "mode": mode.lower()},
                timeout=300
            )
            if response.ok:
                st.success(response.json()["answer"])
            else:
                st.error(" Failed to get answer from backend.")
        except Exception as e:
            st.error(f"Request failed: {e}")
