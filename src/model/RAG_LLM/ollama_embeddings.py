from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os

def load_database(documents): 
    if not isinstance(documents, list):
        raise ValueError("Expected a list of LangChain `Document` objects")

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    print("🔹 Testing Ollama Embeddings...")
    test_embedding = embeddings.embed_query("test query")
    print(f"Embedding Dimension: {len(test_embedding)}")

    index = faiss.IndexFlatL2(len(test_embedding))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    vector_store.add_documents(documents=documents)

    print(f"Successfully added {len(documents)} documents to FAISS")
    return vector_store