import yfinance as yf
import requests
from fastapi import FastAPI, HTTPException, status, Path
import uvicorn
import logging
import os
import openai
import instructor
from pydantic import BaseModel, Field

# Apply patches provided by 'instructor' to enhance introspection capabilities
instructor.patch()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment Variables for API Keys (fallback to placeholder if not set)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "your-serpapi-key")

# Initialize FastAPI app instance
app = FastAPI(
    title="Stock Sentiment Analysis App",
    description="This app analyzes the sentiment of a given stock ticker by evaluating recent news articles and historical stock data, providing insights on whether to buy or sell.",
    version="1.0.0",
)

# Initialize cache for S&P 500 data
sp500_data = None


# Define Pydantic model for structured responses
class ResponseModel(BaseModel):
    buy_or_sell: str = Field(
        ..., description="Recommendation to buy or sell the stock", example="Buy"
    )
    analysis: str = Field(
        ...,
        description="Detailed sentiment analysis",
        example="The sentiment is positive based on recent news articles.",
    )
    news_sentiment: str = Field(
        ...,
        description="Sentiment of the news articles",
        example="The sentiment is positive based on recent news articles.",
    )
    current_price_usd: float = Field(
        ...,
        description="Current price of the stock in USD",
        example=100.0,
    )
    predicted_price_usd_next_week: float = Field(
        ...,
        description="Predicted price of the stock in USD for next week",
        example=110.0,
    )
    predicted_price_usd_next_month: float = Field(
        ...,
        description="Predicted price of the stock in USD for next month",
        example=120.0,
    )


# Event handler for application startup
@app.on_event("startup")
async def load_sp500_data():
    """Load and cache S&P 500 data at startup."""
    global sp500_data
    sp500 = yf.Ticker("^GSPC")
    sp500_data = sp500.history(period="4w")


# Utility functions
def get_news(ticker: str):
    """Fetch news articles for a given stock ticker."""
    params = {"q": f"{ticker} stock news", "tbm": "nws", "api_key": SERPAPI_KEY}
    response = requests.get("https://serpapi.com/search", params=params)
    if response.status_code != 200:
        logger.error(f"Failed to fetch news data: {response.text}")
        raise HTTPException(
            status_code=response.status_code, detail="Failed to fetch news data"
        )
    return response.json()


def generate_sentiment_analysis_prompt(stock_data, news_data, ticker):
    """Generate a prompt for sentiment analysis using stock and news data."""
    news_articles = "\n".join(
        [
            f"Title: {item['title']} - Snippet: {item['snippet']}"
            for item in news_data.get("news_results", [])
        ]
    )
    return f"""
    Analyze the sentiment of the stock {ticker} based on the recent news articles and stock data.
    I've included the S&P 500 data for the last month as a baseline.

    News Articles:
    {news_articles}

    Stock Data:
    {stock_data.to_string()}
    
    S&P 500 Data:
    {sp500_data.to_string()}
    """


# API endpoints
@app.get(
    "/sentiment/{ticker}",
    summary="Analyze Stock Sentiment",
    description="Retrieve sentiment analysis and trading recommendation for a given stock ticker based on news articles and stock data.",
    response_model=ResponseModel,
    responses={
        200: {
            "description": "A successful response with analysis and recommendations",
            "content": {
                "application/json": {
                    "example": {
                        "buy_or_sell": "Buy",
                        "analysis": "Positive sentiment observed with a potential increase in stock price",
                        "current_price": 150.42,
                        "predicted_price_usd_next_week": 155.50,
                        "predicted_price_usd_next_month": 160.75,
                    }
                }
            },
        },
        404: {
            "description": "Stock ticker not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Stock ticker 'XYZ' not found"}
                }
            },
        },
    },
)
async def analyze_sentiment(ticker: str = Path(..., example="AAPL")):
    """Endpoint for sentiment analysis of a given stock ticker."""
    stock = yf.Ticker(ticker)
    stock_data = stock.history(period="4w")
    news_data = get_news(ticker)

    # Handle cases where the stock data is not found
    if stock_data.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Stock not found"
        )

    prompt = generate_sentiment_analysis_prompt(stock_data, news_data, ticker)

    # Set API key for OpenAI service
    openai.api_key = OPENAI_API_KEY

    # Create a chat completion using OpenAI's API
    response: ResponseModel = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        response_model=ResponseModel,
    )

    return response


# Entry point for running the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
