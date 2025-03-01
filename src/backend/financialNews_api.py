from fastapi import FastAPI, Query
import requests

app = FastAPI()

EODHD_NEWS_API_URL = "https://eodhd.com/api/news"

@app.get("/news/")
def get_news(
    symbol: str = Query("AAPL.US", description="Stock ticker symbol (e.g., AAPL.US)"),
    api_token: str = Query(..., description="Your EODHD API token"),
    limit: int = Query(5, ge=1, le=10, description="Number of news articles (default=5, max=10)"),
):
    """Fetches latest financial news from EODHD API"""
    
    params = {
        "s": symbol,
        "api_token": api_token,
        "limit": limit,
        "fmt": "json"
    }

    response = requests.get(EODHD_NEWS_API_URL, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to fetch data", "status_code": response.status_code}

