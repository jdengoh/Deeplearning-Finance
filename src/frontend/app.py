import streamlit as st
import requests
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import os
import sys

# from model.LSTM.LSTM_model_final import monthly_df, predictions

load_dotenv()

st.title("AI-Powered Trading Assistant")

# Function to fetch market data from Yahoo Finance
def get_market_data(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    data = stock.history(period="5d")
    return data

# Function to plot LSTM predictions
def plot_lstm_predictions(actual_data, predicted_data):
    plt.figure(figsize=(10, 6))
    plt.plot(actual_data.index, actual_data['Close'], label='Actual Prices', color='blue')
    plt.plot(predicted_data.index, predicted_data['Predicted_Close'], label='Predicted Prices', color='red', linestyle='--')
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.xticks(rotation=45)
    plt.title("LSTM Predictions vs Actual Data")
    plt.legend()
    st.pyplot(plt)

# Function to fetch latest market news from FastAPI
def get_latest_market_news(stock_symbol):
    news_api_key = os.getenv("NEWS_API_KEY")
    api_url = f"http://127.0.0.1:8000/news/?symbol={stock_symbol}&api_token={news_api_key}&limit=5"
    response = requests.get(api_url)
    
    if response.status_code == 200:
        return response.json()
    else:
        return [{"headline": "Error fetching news", "link": "#"}]

# User input for stock symbol
stock_symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL")

# Section 1: Display Market Data from Yahoo Finance
st.header(f"Market Data: {stock_symbol}")
if stock_symbol:
    market_data = get_market_data(stock_symbol)
    st.write(market_data)
else:
    st.warning("Please enter a valid stock symbol.")

# # Section 2: LSTM Model Predictions (Plot)
# st.header("LSTM Model Predictions")

# actual_data = monthly_df[['Close']].copy()
# predicted_data = actual_data.copy()
# predicted_data['Predicted_Close'] = predictions.flatten()
# predicted_data.index = monthly_df.index

# plot_lstm_predictions(actual_data, predicted_data)

# Dummy Button for Re-running the Model
# st.subheader("Re-Run Model (Work in Progress)")
# st.button("Re-run the Model")

# # Section 3: Latest Market News
# st.header("Latest Market News")
# news = get_latest_market_news(stock_symbol)
# for article in news:
#     st.markdown(f"[{article['headline']}]({article['link']})")

# Section 4: Predefined LLM Header Queries
st.header("Header Queries for LLM")
header_queries = ["Economic Outlook", "Stock Performance of Tech Companies", "Oil Market Trends"]
selected_query = st.selectbox("Select a query", header_queries)

# Section 5: User Input to Query LLM Response
question = st.sidebar.text_area("Ask DeepSeek-R1 a question:", "")

# Section 6: LLM Response
if st.sidebar.button("Generate Response"):
    if question:
        with st.spinner("Generating response..."):
            response = requests.post(
                "http://127.0.0.1:8000/deepseek",
                json={"question": question}
            ).json()
        
        st.success("Response Generated!")
        st.write(response["response"])
    else:
        st.warning("Please enter a question.")

# Optionally, display the selected predefined header query insights
if selected_query:
    st.header(f"Insights on {selected_query}")
    with st.spinner("Generating response for predefined query..."):
        response = requests.post(
            "http://127.0.0.1:8000/deepseek",
            json={"question": selected_query}
        ).json()
    
    st.success(f"Response for '{selected_query}'")
    st.write(response["response"])
