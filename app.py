# crypto_dashboard.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import feedparser  # For parsing RSS feeds

# Set page config
st.set_page_config(
    page_title="Crypto Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Cache data functions
@st.cache_data(ttl=3600, show_spinner=False)
def get_crypto_data(ticker, period="1y"):
    try:
        crypto = yf.Ticker(ticker)
        hist = crypto.history(period=period)
        return hist
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def get_multiple_cryptos(tickers, period="1y"):
    data = {}
    for ticker in tickers:
        try:
            crypto = yf.Ticker(ticker)
            hist = crypto.history(period=period)
            data[ticker] = hist
        except Exception as e:
            st.warning(f"Could not fetch data for {ticker}: {str(e)}")
    return data

@st.cache_data(ttl=3600, show_spinner=False)
def get_crypto_news():
    try:
        # Using CoinDesk RSS feed as a news source
        rss_url = "https://www.coindesk.com/feed/"
        feed = feedparser.parse(rss_url)
        return feed.entries[:10]  # Get top 10 news articles
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []

# Forecasting function
def forecast_crypto_price(data, days=30):
    if data.empty:
        return None, None, None
    
    # Prepare data
    df = data[['Close']].copy()
    df['Date'] = df.index
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    df = df[['Days', 'Close']]
    
    # Split data
    X = df[['Days']]
    y = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Future prediction
    last_day = df['Days'].max()
    future_days = pd.DataFrame({'Days': range(last_day + 1, last_day + days + 1)})
    future_prices = model.predict(future_days)
    
    # Create dates for future predictions using pandas date_range
    last_date = data.index.max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=days
    )
    
    return future_dates, future_prices, mae

# ... [Keep all other functions same until news_tab]

# News Tab
def news_tab():
    st.header("Latest Cryptocurrency News")
    
    # Get news
    news_articles = get_crypto_news()
    
    if not news_articles:
        st.warning("Could not fetch news articles. Please try again later.")
        return
    
    # Display news cards
    for article in news_articles:
        with st.container():
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Handle image if available
                if 'media_content' in article:
                    st.image(article.media_content[0]['url'], width=200)
                else:
                    st.image("https://via.placeholder.com/200x150?text=No+Image", width=200)
            
            with col2:
                st.markdown(f"### [{article.title}]({article.link})")
                st.caption(f"Published: {article.published[:16]}")
                st.write(article.description[:200] + "...")
                
                # Add a separator
                st.markdown("---")

# ... [Keep rest of the code same]


# Main App
def main():
    # Sidebar
    selected_cryptos, time_period, forecast_days = sidebar()
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Price", "Crypto Analysis", "Forecast", "News"])
    
    with tab1:
        price_tab(selected_cryptos, time_period)
    
    with tab2:
        analysis_tab(selected_cryptos, time_period)
    
    with tab3:
        forecast_tab(selected_cryptos, time_period, forecast_days)
    
    with tab4:
        news_tab()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")
