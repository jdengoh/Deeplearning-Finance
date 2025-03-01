import streamlit as st
import requests
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

st.title("AI-Powered Trading Assistant")

# Function to fetch market data from Yahoo Finance
def get_market_data(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    data = stock.history(period="5d")
    return data

# Sample function to plot LSTM predictions (replace with your actual model predictions)
def plot_lstm_predictions(actual_data, predicted_data):
    plt.figure(figsize=(10, 6))
    plt.plot(actual_data.index, actual_data['Close'], label='Actual Prices', color='blue')
    plt.plot(predicted_data.index, predicted_data['Close'], label='Predicted Prices', color='red', linestyle='--')
    plt.title("LSTM Predictions vs Actual Data")
    plt.legend()
    st.pyplot(plt)

# Sample function to fetch latest market news (placeholder)
def get_latest_market_news():
    # In a real scenario, you could fetch news from a finance API or RSS feed.
    news = [
        {"headline": "Global Markets Rise Amid Economic Recovery", "link": "https://example.com/1"},
        {"headline": "Tech Stocks Lead the Charge on Wall Street", "link": "https://example.com/2"},
        {"headline": "Oil Prices Surge as OPEC Tightens Supply", "link": "https://example.com/3"},
    ]
    return news

# User input for stock symbol
stock_symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL")

# Section 1: Display Market Data from Yahoo Finance
st.header(f"Market Data: {stock_symbol}")
if stock_symbol:
    market_data = get_market_data(stock_symbol)
    st.write(market_data)
else:
    st.warning("Please enter a valid stock symbol.")

# Section 2: LSTM Model Predictions (Plot)
st.header("LSTM Model Predictions")
if stock_symbol:
    predicted_data = market_data.copy()  # Just an example, replace with your LSTM model predictions
    predicted_data['Close'] = predicted_data['Close'] * np.random.uniform(0.98, 1.02, len(predicted_data))  # Fake prediction
    plot_lstm_predictions(market_data, predicted_data)
else:
    st.warning("Please enter a stock symbol to see LSTM predictions.")

# Section 3: Latest Market News
st.header("Latest Market News")
news = get_latest_market_news()
for article in news:
    st.markdown(f"[{article['headline']}]({article['link']})")

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
