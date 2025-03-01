from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader  # CSV
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

# Headers.csv use csvloader()
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
    
    # Add csv documents into vectorstore
    vector_store.add_documents(documents=csv_file)

    return vector_store

