from fastapi import FastAPI, Query
import requests

app = FastAPI(
    title="Financial News API",
    description="Fetches market news based on stock tickers or topics using the EODHD API.",
    version="1.0.0"
)

EODHD_NEWS_API_URL = "https://eodhd.com/api/news"

@app.get("/news/", tags=["Market News"])
def get_news(
    symbol: str = Query(None, description="Stock ticker (e.g., AAPL.US). Required if 'topic' is not set."),
    topic: str = Query(None, description="News topic (e.g., economy, crypto). Required if 'symbol' is not set."),
    api_token: str = Query(..., description="Your EODHD API token"),
    from_date: str = Query(None, description="Start date (YYYY-MM-DD)"),
    to_date: str = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(50, ge=1, le=1000, description="Number of results (default=50, max=1000)"),
    offset: int = Query(0, ge=0, description="Pagination offset (default=0)"),
):
    """Fetch financial news from EODHD API."""
    
    if not symbol and not topic:
        return {"error": "Either 'symbol' or 'topic' must be provided."}

    params = {
        "api_token": api_token,
        "limit": limit,
        "offset": offset,
        "fmt": "json"
    }
    
    if symbol:
        params["s"] = symbol
    if topic:
        params["t"] = topic
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date

    response = requests.get(EODHD_NEWS_API_URL, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to fetch data", "status_code": response.status_code}
