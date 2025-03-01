from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os

def load_database(csv_file):
    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large"
    )

    embeddings = OllamaEmbeddings()
    index = faiss.IndexFlatL2(len(OllamaEmbeddings().embed_query(" ")))
    vector_store = FAISS(
        embedding_function=OllamaEmbeddings(),
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    vector_store.add_documents(documents=csv_file)
    return vector_store
