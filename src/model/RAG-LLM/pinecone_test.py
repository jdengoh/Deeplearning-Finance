import getpass
import os
import time
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from uuid import uuid4
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

def load_pinecone(doc_list):  
    
    load_dotenv()

    if not os.getenv("PINECONE_API_KEY"):
        os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")

    pinecone_api_key = os.environ.get("PINECONE_API_KEY")

    pc = Pinecone(api_key=pinecone_api_key)

    index_name = "langchain-test-index"  # change if desired

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-west-2"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    index = pc.Index(index_name)


    embeddings = OllamaEmbeddings(model="llama3")

    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    uuids = [str(uuid4()) for _ in range(len(doc_list))]

    vector_store.add_documents(documents=doc_list, ids=uuids)

    return vector_store