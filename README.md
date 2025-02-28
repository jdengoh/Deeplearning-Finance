# Deeplearning-Finance
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

## Project Structure

*work in progress*

## Getting Started

### 1. Clone the Repo

```bash
git clone git@github.com:jdengoh/Deeplearning-Finance.git
cd Deeplearning-Finance/src
```

### 2. Set up Virtual Env

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```
### 3. Install Dependencies

```bash
pip install -r requirements.txt
```
## Running the Application

### 1. Start the Ollama Server
DeepSeek-R1 requires Ollama to be running. First, install Ollama if you haven’t already:

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```
<!-- Then, start the server:

```bash
ollama serve
``` -->

If you haven't downloaded DeepSeek-R1 yet:

```bash
ollama pull deepseek/deepseek-r1:1.5b
```

### 2. Start the FastAPI Backend

```bash
cd backend
uvicorn api:app --reload
```

- Runs the API at: `http://127.0.0.1:8000`
- You can test it at: `http://127.0.0.1:8000/docs`

### 3. Start the Streamlit Frontend
In a new terminal:

```bash
cd frontend
streamlit run app.py
```
- Access the app at: `http://localhost:8501`
