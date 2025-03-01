from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from uuid import uuid4
from langchain_core.documents import Document
import os

load_dotenv()

def load_database(doc_list): 
    
    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large",
    )

    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory=os.getenv("DB_DUMP_DIRECTORY")
    )

    uuids = [str(uuid4()) for _ in range(len(doc_list))]     

    vector_store.add_documents(documents=doc_list, ids=uuids)

    return doc_list[2]