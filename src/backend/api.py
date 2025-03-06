from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import DataFrameLoader
from dotenv import load_dotenv
import sys
import os
import re
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "headlines.csv")

from model.RAG_LLM.ollama_embeddings import load_database


# load_dotenv()

app = FastAPI()


df = pd.read_csv(csv_path)
df['ticker'] = df['ticker'].fillna("NA")
df = df.dropna().head(5) # DEMO PURPOSES: 5 processes
print(len(df))
loader = DataFrameLoader(df, page_content_column="headline")
data = loader.load()

# Initialize **DeepSeek-R1** Locally via Ollama
llm = OllamaLLM(model="deepseek-r1:1.5b") 

vector_data = load_database(data)
vector_retriever = vector_data.as_retriever()

system_prompt = (
    "You are an assistant for financial question-answering tasks. "
    "Use the following retrieved stock market headlines to answer "
    "the user's question."
    "Ignore headlines that are not relevant to the user's question. "
    "If you can't find the answer, just provide their information like an expert in the field."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vector_retriever, question_answer_chain)


class QueryRequest(BaseModel):
    question: str


@app.post("/deepseek")  
def generate_trade_signal(request: QueryRequest):
    print(f"🔍 Received query: {request.question}")

    # Retrieve relevant context
    response = rag_chain.invoke({"input": request.question})
    response = response["answer"]
    return {"response": response}
