import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import streamlit as st

# Function to get crypto data using CoinGecko API
@st.cache_data(ttl=300)  # Cache data for 5 minutes
def get_crypto_data(coin_id, days=30):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": days,
            "interval": "daily"
        }
        headers = {
            "Accept": "application/json",
            "User-Agent": "CryptoTrack Dashboard"
        }
        response = requests.get(url, params=params, headers=headers)
        
        if response.status_code == 429:
            st.warning("API rate limit reached. Using cached data if available or trying again shortly...")
            return pd.DataFrame()
        
        if response.status_code != 200:
            st.warning(f"API error: {response.status_code}. Using cached data if available.")
            return pd.DataFrame()
        
        data = response.json()
        
        # Convert price data to DataFrame
        df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[["date", "price"]]
        
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Function to get top cryptocurrencies
@st.cache_data(ttl=300)
def get_top_cryptos(limit=50):
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1,
            "sparkline": False
        }
        headers = {
            "Accept": "application/json",
            "User-Agent": "CryptoTrack Dashboard"
        }
        response = requests.get(url, params=params, headers=headers)
        
        if response.status_code == 429:
            st.warning("API rate limit reached. Using cached data if available or trying again shortly...")
            return pd.DataFrame()
        
        if response.status_code != 200:
            st.warning(f"API error: {response.status_code}. Using cached data if available.")
            return pd.DataFrame()
        
        return pd.DataFrame(response.json())
    except Exception as e:
        st.error(f"Error fetching top cryptocurrencies: {e}")
        return pd.DataFrame()

# Dictionary of popular cryptocurrencies with their IDs
popular_cryptos = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "Binance Coin": "binancecoin",
    "Cardano": "cardano",
    "Solana": "solana",
    "XRP": "ripple",
    "Polkadot": "polkadot",
    "Dogecoin": "dogecoin",
    "Avalanche": "avalanche-2",
    "Polygon": "matic-network",
    "Litecoin": "litecoin",
    "Chainlink": "chainlink",
    "Stellar": "stellar",
    "Uniswap": "uniswap",
    "Tron": "tron"
}

# Show loading spinner for data fetching
@st.cache_data
def get_data_with_progress(func, *args, **kwargs):
    """Wrapper function to show loading spinner"""
    with st.spinner(f"Fetching data..."):
        return func(*args, **kwargs)

# Function to display metrics in a nice styled card
def display_metric(label, value, delta=None, prefix="", suffix=""):
    html = f"""
    <div class="metric-card">
        <div style="font-size: 0.9rem; color: #666;">{label}</div>
        <div class="stat-number">{prefix}{value}{suffix}</div>
    """
    if delta is not None:
        delta_value = float(str(delta).replace("%", ""))
        color_class = "crypto-positive" if delta_value >= 0 else "crypto-negative"
        html += f'<div class="{color_class}">{delta}</div>'
    
    html += "</div>"
    return html

# Function to calculate technical indicators
def calculate_indicators(df):
    # Create a copy to avoid the SettingWithCopyWarning
    df_indicators = df.copy()
    
    # Calculate Simple Moving Averages
    df_indicators['SMA_7'] = df_indicators['price'].rolling(window=7).mean()
    df_indicators['SMA_25'] = df_indicators['price'].rolling(window=25).mean()
    
    # Calculate Relative Strength Index (RSI)
    delta = df_indicators['price'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    # Avoid division by zero
    avg_loss = avg_loss.replace(0, float('inf'))
    rs = avg_gain / avg_loss
    df_indicators['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    df_indicators['EMA_12'] = df_indicators['price'].ewm(span=12, adjust=False).mean()
    df_indicators['EMA_26'] = df_indicators['price'].ewm(span=26, adjust=False).mean()
    df_indicators['MACD'] = df_indicators['EMA_12'] - df_indicators['EMA_26']
    df_indicators['Signal'] = df_indicators['MACD'].ewm(span=9, adjust=False).mean()
    df_indicators['MACD_Hist'] = df_indicators['MACD'] - df_indicators['Signal']
    
    # Calculate Bollinger Bands
    df_indicators['MA_20'] = df_indicators['price'].rolling(window=20).mean()
    df_indicators['STD_20'] = df_indicators['price'].rolling(window=20).std()
    df_indicators['Upper_Band'] = df_indicators['MA_20'] + (df_indicators['STD_20'] * 2)
    df_indicators['Lower_Band'] = df_indicators['MA_20'] - (df_indicators['STD_20'] * 2)
    
    return df_indicators

# Function to create a sample portfolio for demo purposes
def create_sample_portfolio():
    portfolio = []
    
    # Add Bitcoin
    portfolio.append({
        "crypto_name": "Bitcoin",
        "crypto_id": "bitcoin",
        "quantity": 0.5,
        "purchase_price": 40000,
        "purchase_date": datetime.now() - timedelta(days=30),
        "current_price": 45000,
        "current_value": 22500,
        "profit_loss": 2500,
        "profit_loss_percentage": 12.5
    })
    
    # Add Ethereum
    portfolio.append({
        "crypto_name": "Ethereum",
        "crypto_id": "ethereum",
        "quantity": 3.0,
        "purchase_price": 2800,
        "purchase_date": datetime.now() - timedelta(days=14),
        "current_price": 3000,
        "current_value": 9000,
        "profit_loss": 600,
        "profit_loss_percentage": 7.14
    })
    
    # Add Cardano
    portfolio.append({
        "crypto_name": "Cardano",
        "crypto_id": "cardano",
        "quantity": 1000.0,
        "purchase_price": 1.2,
        "purchase_date": datetime.now() - timedelta(days=7),
        "current_price": 1.1,
        "current_value": 1100,
        "profit_loss": -100,
        "profit_loss_percentage": -8.33
    })
    
    return portfolio

# Create demo data for display purposes when API fails
def create_demo_data():
    demo_data = {
        "name": ["Bitcoin", "Ethereum", "Binance Coin", "Cardano", "Solana", 
                "XRP", "Polkadot", "Dogecoin", "Avalanche", "Polygon"],
        "symbol": ["btc", "eth", "bnb", "ada", "sol", "xrp", "dot", "doge", "avax", "matic"],
        "current_price": [45000, 3200, 450, 1.2, 100, 0.5, 15, 0.1, 30, 0.8],
        "market_cap": [800000000000, 350000000000, 80000000000, 40000000000, 30000000000,
                     20000000000, 15000000000, 10000000000, 8000000000, 7000000000],
        "price_change_percentage_24h": [2.5, -1.3, 0.8, -2.1, 3.7, 1.2, -0.5, 4.2, -1.8, 2.7],
        "total_volume": [30000000000, 20000000000, 5000000000, 2000000000, 3000000000,
                        1500000000, 1000000000, 800000000, 700000000, 500000000],
        "id": ["bitcoin", "ethereum", "binancecoin", "cardano", "solana", 
             "ripple", "polkadot", "dogecoin", "avalanche-2", "matic-network"]
    }
    return pd.DataFrame(demo_data)
