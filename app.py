import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Crypto Dashboard",
    page_icon="💰",
    layout="wide"
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    .subheader {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 2rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #1E88E5;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .crypto-positive {
        color: #4CAF50 !important;
        font-weight: bold;
    }
    
    .crypto-negative {
        color: #F44336 !important;
        font-weight: bold;
    }
    
    .stat-number {
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .sidebar-content {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 10px;
    }
    
    /* Theme selector */
    .stRadio > div {
        display: flex;
        flex-direction: row;
        gap: 10px;
    }
    
    /* Customize DataFrames */
    .dataframe-container {
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
        color: #666666;
    }
    
    /* Make the page more readable on mobile */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .subheader {
            font-size: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown("<h1 class='main-header'>📊 Cryptocurrency Analysis Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem;'>A simple dashboard to analyze cryptocurrency data</p>", unsafe_allow_html=True)

# Sidebar for navigation
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>Navigation</h2>", unsafe_allow_html=True)
    
    # Add logo placeholder
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <svg width="100" height="100" viewBox="0 0 200 200">
            <circle cx="100" cy="100" r="80" fill="#1E88E5" />
            <text x="100" y="115" font-size="50" text-anchor="middle" fill="white">₿</text>
            <path d="M100,20 a80,80 0 0,1 0,160" fill="none" stroke="#64B5F6" stroke-width="12" />
        </svg>
        <h3>CryptoTrack</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
    page = st.radio("", ["📈 Overview", "💹 Price Analysis", "🔄 Comparison", "💼 Portfolio Tracker"])
    page = page.split(" ")[1]  # Extract the page name without emoji
    
    # Theme selector
    st.markdown("### 🎨 Theme")
    theme = st.radio("", ["Light", "Dark", "Blue"], horizontal=True)
    
    # Apply theme
    if theme == "Dark":
        st.markdown("""
        <style>
            body {
                color: #f1f1f1;
                background-color: #121212;
            }
            .css-1d391kg, .css-1wrcr25, .stApp {
                background-color: #121212;
            }
            .metric-card {
                background-color: #1e1e1e;
                color: #f1f1f1;
            }
            .sidebar-content {
                background-color: #1e1e1e;
                color: #f1f1f1;
            }
            .subheader {
                border-bottom: 2px solid #bb86fc;
                color: #bb86fc;
            }
            .main-header {
                color: #bb86fc;
            }
        </style>
        """, unsafe_allow_html=True)
    elif theme == "Blue":
        st.markdown("""
        <style>
            body {
                color: #ffffff;
                background-color: #0D47A1;
            }
            .css-1d391kg, .css-1wrcr25, .stApp {
                background-color: #0D47A1;
                background-image: linear-gradient(160deg, #0D47A1 0%, #1976D2 100%);
            }
            .metric-card {
                background-color: rgba(255, 255, 255, 0.1);
                color: #ffffff;
                backdrop-filter: blur(10px);
            }
            .sidebar-content {
                background-color: rgba(255, 255, 255, 0.1);
                color: #ffffff;
                backdrop-filter: blur(10px);
            }
            .subheader {
                border-bottom: 2px solid #64B5F6;
                color: #E3F2FD;
            }
            .main-header {
                color: #E3F2FD;
            }
        </style>
        """, unsafe_allow_html=True)
    
    # Market status
    st.markdown("### 📊 Market Status")
    market_status = "Open"  # This would be determined by API in a full implementation
    st.markdown(f"<div style='text-align: center; margin: 10px 0;'><span style='background-color: {'#4CAF50' if market_status=='Open' else '#F44336'}; color: white; padding: 5px 10px; border-radius: 20px;'>{market_status}</span></div>", unsafe_allow_html=True)
    
    # Display current time
    st.markdown("### 🕒 Current Time")
    st.markdown(f"<div style='text-align: center;'>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Disclaimer
    with st.expander("Disclaimer"):
        st.markdown("""
        This dashboard is for educational purposes only. Cryptocurrency investments are subject to market risks. 
        Data is provided by CoinGecko API and may not be real-time.
        """)


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


# Overview Page
if page == "Overview":
    st.markdown("<h2 class='subheader'>Cryptocurrency Market Overview</h2>", unsafe_allow_html=True)
    
    # Show loading animation while fetching data
    with st.spinner("Fetching market data..."):
        # Load top cryptocurrencies data
        top_cryptos = get_data_with_progress(get_top_cryptos, limit=25)
    
    if not top_cryptos.empty:
        # Display market stats
        st.markdown("<div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;'>", unsafe_allow_html=True)
        
        # Total Market Cap
        market_cap_sum = int(top_cryptos['market_cap'].sum() / 1e9)
        st.markdown(display_metric("Total Market Cap", f"{market_cap_sum:,}B", prefix="$"), unsafe_allow_html=True)
        
        # 24h Volume
        volume_sum = int(top_cryptos['total_volume'].sum() / 1e9)
        st.markdown(display_metric("24h Volume", f"{volume_sum:,}B", prefix="$"), unsafe_allow_html=True)
        
        # Bitcoin Dominance
        btc_data = top_cryptos[top_cryptos['id'] == 'bitcoin']
        if not btc_data.empty:
            bitcoin_dominance = (btc_data['market_cap'].values[0] / top_cryptos['market_cap'].sum()) * 100
            st.markdown(display_metric("Bitcoin Dominance", f"{bitcoin_dominance:.2f}", suffix="%"), unsafe_allow_html=True)
        else:
            st.markdown(display_metric("Bitcoin Dominance", "N/A", suffix="%"), unsafe_allow_html=True)
        
        # Avg 24h Change
        avg_change = top_cryptos['price_change_percentage_24h'].mean()
        delta = f"{avg_change:.2f}%"
        st.markdown(display_metric("Avg 24h Change", f"{avg_change:.2f}", delta=delta, suffix="%"), unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Market Trends Visualization
        st.markdown("<h3 class='subheader'>Market Trends</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Market trend chart showing price movement for top 5 cryptos
            st.markdown("<h4>Top 5 Cryptocurrencies Price Trend (7d)</h4>", unsafe_allow_html=True)
            
            # Get price data for top 5 cryptos
            trend_fig = go.Figure()
            
            has_data = False
            for i, crypto in enumerate(top_cryptos.head(5)['id']):
                df = get_crypto_data(crypto, days=7)
                if not df.empty:
                    has_data = True
                    trend_fig.add_trace(go.Scatter(
                        x=df['date'], 
                        y=df['price'],
                        mode='lines',
                        name=top_cryptos.iloc[i]['name']
                    ))
            
            if has_data:
                trend_fig.update_layout(
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=20, r=20, t=40, b=20),
                    hovermode="x unified"
                )
                st.plotly_chart(trend_fig, use_container_width=True)
            else:
                st.warning("Unable to fetch price trends. Please try again later.")
        
        with col2:
            # Market sentiment gauge based on 24h price changes
            st.markdown("<h4>Market Sentiment</h4>", unsafe_allow_html=True)
            
            positive_cryptos = len(top_cryptos[top_cryptos['price_change_percentage_24h'] > 0])
            total_cryptos = len(top_cryptos)
            sentiment_score = (positive_cryptos / total_cryptos) * 100
            
            # Create a gauge chart for sentiment
            sentiment_labels = {
                (0, 20): "Very Bearish",
                (20, 40): "Bearish",
                (40, 60): "Neutral",
                (60, 80): "Bullish",
                (80, 100): "Very Bullish"
            }
            
            # Determine current sentiment
            current_sentiment = None
            for (lower, upper), label in sentiment_labels.items():
                if lower <= sentiment_score <= upper:
                    current_sentiment = label
                    break
            
            # Create gauge chart
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sentiment_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "rgba(50, 150, 255, 0.8)"},
                    'steps': [
                        {'range': [0, 20], 'color': "#EF5350"},
                        {'range': [20, 40], 'color': "#FFA726"},
                        {'range': [40, 60], 'color': "#FFEE58"},
                        {'range': [60, 80], 'color': "#66BB6A"},
                        {'range': [80, 100], 'color': "#26A69A"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': sentiment_score
                    }
                },
                title={'text': f"Current: {current_sentiment}"}
            ))
            
            gauge_fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=70, b=20),
            )
            
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Display top cryptocurrencies table
        st.markdown("<h3 class='subheader'>Top Cryptocurrencies by Market Cap</h3>", unsafe_allow_html=True)
        
        # Format the table
        formatted_df = top_cryptos[['name', 'symbol', 'current_price', 'market_cap', 
                                 'price_change_percentage_24h', 'total_volume']].copy()
        
        formatted_df.columns = ['Name', 'Symbol', 'Price (USD)', 'Market Cap (USD)', 
                            '24h Change (%)', '24h Volume (USD)']
        
        # Format the values
        formatted_df['Symbol'] = formatted_df['Symbol'].str.upper()
        formatted_df['Price (USD)'] = formatted_df['Price (USD)'].apply(lambda x: f"${x:,.2f}")
        formatted_df['Market Cap (USD)'] = formatted_df['Market Cap (USD)'].apply(lambda x: f"${x:,.0f}")
        
        # Store the raw values for styling but display the formatted values
        change_vals = top_cryptos['price_change_percentage_24h']
        formatted_df['24h Change (%)'] = formatted_df['24h Change (%)'].apply(lambda x: f"{x:.2f}%")
        
        formatted_df['24h Volume (USD)'] = formatted_df['24h Volume (USD)'].apply(lambda x: f"${x:,.0f}")
        
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
        st.dataframe(
            formatted_df,
            column_config={
                "24h Change (%)": st.column_config.Column(
                    "24h Change (%)",
                    help="Price change in the last 24 hours",
                    width="medium",
                )
            },
            use_container_width=True,
            hide_index=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Market visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Market Cap Distribution
            st.markdown("<h3 class='subheader'>Market Cap Distribution</h3>", unsafe_allow_html=True)
            fig = px.pie(
                top_cryptos.head(8), 
                values='market_cap', 
                names='name', 
                title='Top 8 Cryptocurrencies by Market Cap',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                margin=dict(l=20, r=20, t=60, b=60)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 24h Price Change
            st.markdown("<h3 class='subheader'>24h Price Change</h3>", unsafe_allow_html=True)
            
            # Get top 5 gainers and losers
            gainers = top_cryptos.sort_values(by='price_change_percentage_24h', ascending=False).head(5)
            losers = top_cryptos.sort_values(by='price_change_percentage_24h', ascending=True).head(5)
            
            combined = pd.concat([gainers, losers])
            
            fig = px.bar(
                combined, 
                x='name', 
                y='price_change_percentage_24h',
                title='Top Gainers and Losers (24h)',
                labels={'name': 'Cryptocurrency', 'price_change_percentage_24h': '24h Change (%)'},
                color='price_change_percentage_24h',
                color_continuous_scale=['#F44336', '#FFFFFF', '#4CAF50'],
                range_color=[-max(abs(combined['price_change_percentage_24h'])), max(abs(combined['price_change_percentage_24h']))]
            )
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=60, b=100),
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("No data available. Please check your internet connection or try again later.")
        
        # Show demo data option
        if st.button("Load Demo Data"):
            # Create demo data for display purposes
            demo_df = create_demo_data()
            
            # Display demo data with a clear notice
            st.info("⚠️ This is demo data for display purposes only. It does not reflect real market conditions.")
            
            # Format the demo data for display
            formatted_demo = demo_df[['name', 'symbol', 'current_price', 'market_cap', 
                                   'price_change_percentage_24h', 'total_volume']].copy()
            
            formatted_demo.columns = ['Name', 'Symbol', 'Price (USD)', 'Market Cap (USD)', 
                                  '24h Change (%)', '24h Volume (USD)']
            
            # Format the values
            formatted_demo['Symbol'] = formatted_demo['Symbol'].str.upper()
            formatted_demo['Price (USD)'] = formatted_demo['Price (USD)'].apply(lambda x: f"${x:,.2f}")
            formatted_demo['Market Cap (USD)'] = formatted_demo['Market Cap (USD)'].apply(lambda x: f"${x:,.0f}")
            formatted_demo['24h Change (%)'] = formatted_demo['24h Change (%)'].apply(lambda x: f"{x:.2f}%")
            formatted_demo['24h Volume (USD)'] = formatted_demo['24h Volume (USD)'].apply(lambda x: f"${x:,.0f}")
            
            st.dataframe(formatted_demo, use_container_width=True)

# Price Analysis Page
elif page == "Price Analysis":
    st.markdown("<h2 class='subheader'>Cryptocurrency Price Analysis</h2>", unsafe_allow_html=True)
    
    # Select cryptocurrency and timeframe
    col1, col2 = st.columns(2)
    
    with col1:
        selected_crypto = st.selectbox("Select Cryptocurrency", list(popular_cryptos.keys()))
    
    with col2:
        timeframe = st.selectbox("Select Timeframe", ["7 days", "30 days", "90 days", "1 year", "Max"])
    
    timeframe_days = {
        "7 days": 7,
        "30 days": 30,
        "90 days": 90,
        "1 year": 365,
        "Max": "max"
    }
    
    # Get data
    with st.spinner(f"Fetching {selected_crypto} data..."):
        crypto_id = popular_cryptos[selected_crypto]
        df = get_crypto_data(crypto_id, days=timeframe_days[timeframe])
    
    if not df.empty:
        # Calculate technical indicators
        df_indicators = calculate_indicators(df)
        
        # Display current price metrics
        latest_price = df['price'].iloc[-1]
        prev_price = df['price'].iloc[-2] if len(df) > 1 else latest_price
        price_change = ((latest_price - prev_price) / prev_price) * 100
        price_change_str = f"{price_change:.2f}%"
        
        st.markdown("<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 20px;'>", unsafe_allow_html=True)
        st.markdown(display_metric("Current Price", f"${latest_price:,.2f}", delta=price_change_str), unsafe_allow_html=True)
        st.markdown(display_metric("24h High", f"${df['price'].max():,.2f}"), unsafe_allow_html=True)
        st.markdown(display_metric("24h Low", f"${df['price'].min():,.2f}"), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Price chart with technical indicators
        st.markdown("<h3 class='subheader'>Price Chart with Indicators</h3>", unsafe_allow_html=True)
        
        # Create tabs for different chart views
        tab1, tab2, tab3 = st.tabs(["Price Trend", "Technical Indicators", "Candlestick"])
        
        with tab1:
            # Basic price trend
            fig = px.line(df, x='date', y='price', title=f"{selected_crypto} Price Trend")
            fig.update_layout(
                yaxis_title="Price (USD)",
                hovermode="x unified",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Technical indicators visualization
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(go.Scatter(
                x=df_indicators['date'],
                y=df_indicators['price'],
                name='Price',
                line=dict(color='#1E88E5', width=2)
            ))
            
            # Add moving averages
            fig.add_trace(go.Scatter(
                x=df_indicators['date'],
                y=df_indicators['SMA_7'],
                name='7-day SMA',
                line=dict(color='#FFA000', width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=df_indicators['date'],
                y=df_indicators['SMA_25'],
                name='25-day SMA',
                line=dict(color='#43A047', width=1)
            ))
            
            # Add Bollinger Bands
            fig.add_trace(go.Scatter(
                x=df_indicators['date'],
                y=df_indicators['Upper_Band'],
                name='Upper Band',
                line=dict(color='#757575', width=1, dash='dot')
            ))
            
            fig.add_trace(go.Scatter(
                x=df_indicators['date'],
                y=df_indicators['Lower_Band'],
                name='Lower Band',
                line=dict(color='#757575', width=1, dash='dot'),
                fill='tonexty',
                fillcolor='rgba(117,117,117,0.1)'
            ))
            
            fig.update_layout(
                title=f"{selected_crypto} Price with Technical Indicators",
                yaxis_title="Price (USD)",
                hovermode="x unified",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # RSI and MACD subplots
            st.markdown("<h4>Momentum Indicators</h4>", unsafe_allow_html=True)
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
            
            # RSI
            fig.add_trace(go.Scatter(
                x=df_indicators['date'],
                y=df_indicators['RSI'],
                name='RSI',
                line=dict(color='#7E57C2')
            ), row=1, col=1)
            
            # Add RSI thresholds
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
            
            # MACD
            fig.add_trace(go.Scatter(
                x=df_indicators['date'],
                y=df_indicators['MACD'],
                name='MACD',
                line=dict(color='#26A69A')
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=df_indicators['date'],
                y=df_indicators['Signal'],
                name='Signal',
                line=dict(color='#FF7043')
            ), row=2, col=1)
            
            # MACD Histogram
            colors = ['green' if val >= 0 else 'red' for val in df_indicators['MACD_Hist']]
            fig.add_trace(go.Bar(
                x=df_indicators['date'],
                y=df_indicators['MACD_Hist'],
                name='MACD Hist',
                marker_color=colors
            ), row=2, col=1)
            
            fig.update_layout(
                height=600,
                showlegend=True,
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Candlestick chart (requires OHLC data - using simple approximation here)
            if len(df) > 7:  # Need enough data points for meaningful candlesticks
                # Group by week for candlesticks
                df_candle = df.copy()
                df_candle['week'] = df_candle['date'].dt.isocalendar().week
                df_candle = df_candle.groupby('week').agg({
                    'price': ['first', 'max', 'min', 'last']
                }).reset_index()
                df_candle.columns = ['week', 'open', 'high', 'low', 'close']
                
                fig = go.Figure(go.Candlestick(
                    x=df_candle['week'],
                    open=df_candle['open'],
                    high=df_candle['high'],
                    low=df_candle['low'],
                    close=df_candle['close'],
                    name='Candlestick'
                ))
                
                fig.update_layout(
                    title=f"{selected_crypto} Weekly Candlestick Chart",
                    yaxis_title="Price (USD)",
                    xaxis_title="Week",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough data points for candlestick chart. Please select a longer timeframe.")
        
        # Price statistics
        st.markdown("<h3 class='subheader'>Price Statistics</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h4>Daily Returns Distribution</h4>", unsafe_allow_html=True)
            
            returns = df['price'].pct_change().dropna()
            
            fig = px.histogram(
                x=returns,
                nbins=50,
                title="Distribution of Daily Returns",
                labels={'x': 'Daily Return'}
            )
            
            fig.update_layout(
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("<h4>Volatility Analysis</h4>", unsafe_allow_html=True)
            
            rolling_volatility = returns.rolling(window=7).std() * np.sqrt(7)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['date'].iloc[7:],
                y=rolling_volatility.iloc[7:],
                name='7-day Rolling Volatility'
            ))
            
            fig.update_layout(
                yaxis_title="Volatility",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Volatility metrics
            st.markdown("<div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 20px;'>", unsafe_allow_html=True)
            st.markdown(display_metric("Avg Daily Return", f"{returns.mean() * 100:.2f}%"), unsafe_allow_html=True)
            st.markdown(display_metric("Annualized Volatility", f"{returns.std() * np.sqrt(365) * 100:.2f}%"), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        st.warning(f"Could not fetch data for {selected_crypto}. Please try again later.")
