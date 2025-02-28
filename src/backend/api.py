from fastapi import FastAPI
from pydantic import BaseModel
import subprocess

app = FastAPI()

# Request Model
class QueryRequest(BaseModel):
    question: str

# DeepSeek-R1 Inference
def ask_deepseek(question):
    result = subprocess.run(
        ["ollama", "run", "deepseek-r1:1.5b", question],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

# API Endpoint for DeepSeek-R1
@app.post("/deepseek")
def generate_trade_signal(request: QueryRequest):
    response = ask_deepseek(request.question)
    return {"response": response}