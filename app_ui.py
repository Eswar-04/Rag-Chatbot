import streamlit as st
import requests

# Set Streamlit page configuration
st.set_page_config(page_title="Chatbot", page_icon="📘", layout="centered")

# Display chatbot title and instructions
st.markdown("## Chatbot")
st.markdown("Ask any question related to **Artificial Intelligence.pdf**")

# Input field for user to type their question
question = st.text_input("Ask a question (text or image related)", "")

# Radio buttons to choose between text-based or image-based question
mode = st.radio("Select question type:", ["Text", "Image"], horizontal=True)

# When the "Ask" button is clicked and a question is entered
if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        try:
            # Send POST request to FastAPI backend with question and mode
            response = requests.post(
                "http://localhost:8000/ask",  # Backend endpoint
                json={"query": question, "mode": mode.lower()},  # Payload
                timeout=300  # Timeout in seconds
            )

            # If request is successful, show the answer
            if response.ok:
                st.success(response.json()["answer"])
            else:
                # Show error if backend returns a failed response
                st.error(" Failed to get answer from backend.")
        except Exception as e:
            # Handle and display request failure or connection error
            st.error(f"Request failed: {e}")
