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
        # Using Yahoo Finance news feed
        news = yf.Ticker("BTC-USD").news
        return news[:10]  # Get top 10 news articles
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

# Sidebar
def sidebar():
    st.sidebar.title("Crypto Dashboard")
    st.sidebar.markdown("Select cryptocurrencies and time period to analyze")
    
    # Popular cryptocurrencies
    crypto_options = {
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
        "Binance Coin": "BNB-USD",
        "Cardano": "ADA-USD",
        "Solana": "SOL-USD",
        "XRP": "XRP-USD",
        "Polkadot": "DOT-USD",
        "Dogecoin": "DOGE-USD",
        "Shiba Inu": "SHIB-USD",
        "Polygon": "MATIC-USD"
    }
    
    selected_names = st.sidebar.multiselect(
        "Select Cryptocurrencies",
        list(crypto_options.keys()),
        default=["Bitcoin", "Ethereum"]
    )
    
    selected_cryptos = [crypto_options[name] for name in selected_names]
    
    # Time period selection
    time_period = st.sidebar.selectbox(
        "Select Time Period",
        options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
        index=5
    )
    
    # Forecast days
    forecast_days = st.sidebar.slider(
        "Forecast Days",
        min_value=7,
        max_value=90,
        value=30
    )
    
    return selected_cryptos, time_period, forecast_days

# Price Tab
def price_tab(selected_cryptos, time_period):
    st.header("Cryptocurrency Price Analysis")
    
    if not selected_cryptos:
        st.warning("Please select at least one cryptocurrency from the sidebar.")
        return
    
    # Get data for selected cryptos
    crypto_data = get_multiple_cryptos(selected_cryptos, time_period)
    
    if not crypto_data:
        st.error("No data available for the selected cryptocurrencies.")
        return
    
    # Plot price chart
    fig = go.Figure()
    
    for ticker, data in crypto_data.items():
        if not data.empty:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                name=ticker,
                mode='lines'
            ))
    
    fig.update_layout(
        title="Cryptocurrency Price Trends",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show raw data
    if st.checkbox("Show raw data"):
        for ticker, data in crypto_data.items():
            st.subheader(ticker)
            st.dataframe(data)

# Analysis Tab
def analysis_tab(selected_cryptos, time_period):
    st.header("Cryptocurrency Technical Analysis")
    
    if not selected_cryptos:
        st.warning("Please select at least one cryptocurrency from the sidebar.")
        return
    
    # Get data for selected cryptos
    crypto_data = get_multiple_cryptos(selected_cryptos, time_period)
    
    if not crypto_data:
        st.error("No data available for the selected cryptocurrencies.")
        return
    
    # Calculate daily returns
    returns_data = {}
    for ticker, data in crypto_data.items():
        if not data.empty:
            returns_data[ticker] = data['Close'].pct_change().dropna()
    
    # Plot returns distribution
    if len(returns_data) > 0:
        st.subheader("Daily Returns Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for ticker, returns in returns_data.items():
            sns.histplot(returns, kde=True, label=ticker, ax=ax)
        
        ax.set_title("Distribution of Daily Returns")
        ax.set_xlabel("Daily Return")
        ax.set_ylabel("Frequency")
        ax.legend()
        
        st.pyplot(fig)
    
    # Correlation matrix
    if len(selected_cryptos) > 1:
        st.subheader("Correlation Matrix")
        
        # Create a DataFrame with closing prices
        close_prices = pd.DataFrame()
        for ticker, data in crypto_data.items():
            if not data.empty:
                close_prices[ticker] = data['Close']
        
        if not close_prices.empty:
            corr_matrix = close_prices.corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

# Forecast Tab
def forecast_tab(selected_cryptos, time_period, forecast_days):
    st.header("Cryptocurrency Price Forecast")
    
    if not selected_cryptos:
        st.warning("Please select at least one cryptocurrency from the sidebar.")
        return
    
    # Get data for selected cryptos
    crypto_data = get_multiple_cryptos(selected_cryptos, time_period)
    
    if not crypto_data:
        st.error("No data available for the selected cryptocurrencies.")
        return
    
    # Create tabs for each crypto
    tabs = st.tabs([ticker.split("-")[0] for ticker in selected_cryptos])
    
    for i, (ticker, data) in enumerate(crypto_data.items()):
        if data.empty:
            continue
        
        with tabs[i]:
            st.subheader(f"{ticker.split('-')[0]} Price Forecast")
            
            # Get forecast
            future_dates, future_prices, mae = forecast_crypto_price(data, forecast_days)
            
            if future_dates is not None:
                # Plot historical and forecasted data
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    name="Historical",
                    mode='lines'
                ))
                
                # Forecasted data
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_prices,
                    name="Forecast",
                    mode='lines',
                    line=dict(color='orange', dash='dash')
                ))
                
                fig.update_layout(
                    title=f"{ticker.split('-')[0]} Price Forecast",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    hovermode="x unified",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show forecast metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Last Price", f"${data['Close'].iloc[-1]:,.2f}")
                with col2:
                    st.metric("Forecast MAE", f"${mae:,.2f}")
                
                # Show forecast table
                if st.checkbox(f"Show forecast data for {ticker.split('-')[0]}"):
                    forecast_df = pd.DataFrame({
                        "Date": future_dates,
                        "Forecasted Price": future_prices
                    })
                    st.dataframe(forecast_df)

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
                if 'thumbnail' in article and article['thumbnail']['resolutions']:
                    st.image(article['thumbnail']['resolutions'][0]['url'], width=200)
                else:
                    st.image("https://via.placeholder.com/200x150?text=No+Image", width=200)
            
            with col2:
                title = article.get('title', 'No title available')
                link = article.get('link', '#')
                st.markdown(f"### [{title}]({link})")
                
                if 'providerPublishTime' in article:
                    publish_time = datetime.fromtimestamp(article['providerPublishTime'])
                    st.caption(f"Published: {publish_time.strftime('%Y-%m-%d %H:%M')}")
                
                if 'summary' in article:
                    st.write(article['summary'][:200] + "...")
                elif 'description' in article:
                    st.write(article['description'][:200] + "...")
                
                # Add a separator
                st.markdown("---")

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
