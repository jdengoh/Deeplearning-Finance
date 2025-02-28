import streamlit as st
import requests

st.title("AI-Powered Trading Assistant")

# User input
question = st.sidebar.text_area("Ask DeepSeek-R1 a question:", "")

if st.sidebar.button("Generate Response"):
    if question:
        with st.spinner("Generating response..."):
            response = requests.post(
                "http://127.0.0.1:8000/deepseek",
                json={"question": question}
            ).json()
        
        st.success("Response Generated!")
        st.write(response["response"])
    else:
        st.warning("Please enter a question.")
