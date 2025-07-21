import streamlit as st  # Streamlit: main library to build the UI
from app.qa import ask_question  # Import backend function to get answers

# Set the page title and center the layout
st.set_page_config(page_title="RAG Chatbot", layout="centered")

# Display the main heading for the chatbot
st.title(" RAG Chatbot")

# Input box where the user types their question
query = st.text_input("Ask your question below ")

# When the user presses the "Ask" button, run the following.
if st.button("Ask"):

    # Only proceed if the user typed something (not empty or blank)
    if query.strip() != "":

        # Show a spinner while the chatbot processes the question
        with st.spinner("Thinking... "):

            try:
                # Call the ask_question function to retrieve an answer
                response = ask_question(query)

                # - Display the answer heading
                st.success(" Answer:")

                # - Show the actual answer text
                st.write(response["answer"])

                # - Show the similarity score from the search
                st.write(f"Best match score: {response['score']:.4f}")

            except Exception as e:
                # If something goes wrong, show an error message
                st.error(f" Error: {e}")

    else:
        # If the input is empty, prompt the user to type something
        st.warning("Please type something!")
