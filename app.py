import yfinance as yf
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Depends, status
import uvicorn
import logging
import os


import openai
import instructor
from pydantic import BaseModel

instructor.patch()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment Variables for API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "your-serpapi-key")

app = FastAPI()

# Cache for S&P 500 data
sp500_data = None


# create a pydantic response model
class response_model(BaseModel):
    buy_or_sell: str
    analysis: str
    news_sentiment: str
    current_price_usd: float
    predicted_price_usd_next_week: float
    predicted_price_usd_next_month: float


@app.on_event("startup")
async def load_sp500_data():
    global sp500_data
    sp500 = yf.Ticker("^GSPC")
    sp500_data = sp500.history(period="2w")  # Cache the S&P 500 data at startup


def get_news(ticker: str):
    params = {"q": f"{ticker} stock news", "tbm": "nws", "api_key": SERPAPI_KEY}
    response = requests.get("https://serpapi.com/search", params=params)
    if response.status_code != 200:
        logger.error(f"Failed to fetch news data: {response.text}")
        raise HTTPException(
            status_code=response.status_code, detail="Failed to fetch news data"
        )
    return response.json()


def generate_sentiment_analysis_prompt(stock_data, news_data, ticker):
    news_articles = "\n".join(
        [
            f"Title: {item['title']} - Snippet: {item['snippet']}"
            for item in news_data.get("news_results", [])
        ]
    )
    return f"""
    Analyze the sentiment of the stock {ticker} based on the recent news articles and stock data.
    ive included the snp 500 data for the last 2 weeks as a baseline.

    News Articles:
    {news_articles}

    Stock Data:
    {stock_data.to_string()}
    
    snp 500 data:
    {sp500_data.to_string()}
    """


@app.get("/sentiment/{ticker}")
async def analyze_sentiment(ticker: str):
    stock = yf.Ticker(ticker)
    stock_data = stock.history(period="2w")
    news_data = get_news(ticker)

    if stock_data.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Stock not found"
        )

    prompt = generate_sentiment_analysis_prompt(stock_data, news_data, ticker)

    openai.api_key = OPENAI_API_KEY
    
    response: response_model = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt},
        ],
        response_model=response_model,
    )

    return response


if __name__ == "__main__":
    uvicorn.run(app)
