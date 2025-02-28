import asyncio
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from InstructorEmbedding import INSTRUCTOR
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from html_templates import css, bot_template, user_template

from os import getenv


def get_pdf_text(pdf_docs):
    """Extract text from PDFs and transforms into a single string"""
    text = ""
    for pdf in pdf_docs: # Loop through each PDF
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages: # Loop through each page
            text += page.extract_text()
    return text # single string of text

def get_text_chunks(text):
    """Split text into chunks of 1000 characters"""
    text_chunks = []
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000, # Number of characters per chunk
        chunk_overlap=200, # Prevents words from cutting off
        length_function = len # Function to get total length of text
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """
    Using InstructorEmbedding to create a vector store.
    Runs locally, hence, requires large computing power to 
    and store the embeddings.
    OpenAI's ada model uses the cloud, hence, it is faster and
    requires lower computing power.
    """

    instructembeddings= HuggingFaceInstructEmbeddings(
        model_name = "hkunlp/instructor-xl",
        model_kwargs = {'device': 'cpu'}, 
        encode_kwargs = {'normalize_embeddings': True}
        )

    # Create a vector store with FAISS
    vectorstore = FAISS.from_texts(text_chunks, instructembeddings)
    
    return vectorstore

def get_openrouter(model: str = "deepseek-r1:1.5b") -> ChatOpenAI:
    # alternatively you can use OpenAI directly via ChatOpenAI(model="gpt-4o") via -> from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model,
        openai_api_key=getenv("OPENROUTER_API_KEY"),
        openai_api_base=getenv("https://openrouter.ai/api/v1"))

def get_conversation_chain(vectorstore):
    """
    Create a conversation chain by prompting the model with a question.
    """
    # Initialize an instance of the model
    llm = get_openrouter(
        model="deepseek-r1:1.5b"
    )  

    # Initialize an instance of memory using ConversationBufferMemory for the new memory architecture
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Initialize session using ConversationalRetrievalChain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=vectorstore.as_retriever()
    )
    
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.write(response)

def main():
    load_dotenv()
    st.set_page_config(page_title="PDF App", page_icon=":shark:", layout="wide")

    # Add CSS on top
    st.write(css, unsafe_allow_html=True)
    
    # Checks if conversation is in session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

    # Header
    st.header("PDF App: Sharks")

    # User Input
    user_question = st.text_input("Ask a question about sharks")
    if user_question:
        handle_user_input(user_question)

    st.write(user_template.replace("{{MSG}}", "hello bot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "hello human"), unsafe_allow_html=True)

    with st.sidebar: 
        st.subheader("Navigation")
        pdf_docs = st.file_uploader(
            "Upload a PDF and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):  # Adds spinning wheel (shows not frozen)
                # Get PDF text
                raw_text = get_pdf_text(pdf_docs) # Return single string of text from each pdf
                st.write(raw_text) 

                # Get Text Chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

                # Create vector store from the embeddings
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain (prompt memory)
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    main()

#"""
# st.session_state.[variable] - stores the variable in the session state 
# (doesn't re-initialize when button are pressed fro streamlit)
# makes variables persistent as long as the application is active
#"""