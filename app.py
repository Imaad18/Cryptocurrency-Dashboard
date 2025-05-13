import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    
    /* News cards */
    .news-card {
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .news-title {
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .news-source {
        color: #666;
        font-size: 0.9rem;
    }
    
    /* Calendar events */
    .event-card {
        border-left: 4px solid #1E88E5;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
        background-color: #f9f9f9;
        border-radius: 0 8px 8px 0;
    }
    
    .event-title {
        font-weight: bold;
    }
    
    .event-date {
        color: #666;
        font-size: 0.9rem;
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
    page = st.radio("", ["📈 Overview", "💹 Price Analysis", "🔄 Comparison", "📰 Crypto News", "📅 Crypto Calendar", "💼 Portfolio Tracker"])
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
            .news-card, .event-card {
                background-color: #1e1e1e;
                color: #f1f1f1;
            }
            .news-source, .event-date {
                color: #aaa !important;
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
            .news-card, .event-card {
                background-color: rgba(255, 255, 255, 0.1);
                color: #ffffff;
                backdrop-filter: blur(10px);
            }
            .news-source, .event-date {
                color: #ddd !important;
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

# Function to get top cryptocurrencies
@st.cache_data(ttl=300, show_spinner="Fetching market data...")
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

# Function to get crypto data using CoinGecko API
@st.cache_data(ttl=300, show_spinner="Fetching cryptocurrency data...")
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

# Function to get crypto news (without API)
def get_crypto_news():
    # This is a static list of news - in a real app you would fetch from an API
    news = [
        {
            "title": "Bitcoin Halving Event Completed Successfully",
            "source": "CoinDesk",
            "date": "2023-05-15",
            "summary": "The fourth Bitcoin halving has reduced the block reward from 6.25 BTC to 3.125 BTC, potentially impacting miner profitability and supply dynamics."
        },
        {
            "title": "Ethereum Foundation Announces Major Protocol Upgrade",
            "source": "The Block",
            "date": "2023-05-10",
            "summary": "The upcoming Ethereum upgrade, codenamed 'Dencun', will introduce proto-danksharding to improve scalability and reduce transaction costs."
        },
        {
            "title": "SEC Approves First Spot Bitcoin ETFs",
            "source": "Bloomberg Crypto",
            "date": "2023-05-05",
            "summary": "After years of rejections, the SEC has approved multiple spot Bitcoin ETFs, providing traditional investors with easier access to cryptocurrency."
        },
        {
            "title": "Solana Network Experiences Temporary Outage",
            "source": "Decrypt",
            "date": "2023-05-01",
            "summary": "The Solana blockchain was down for approximately 4 hours due to a bug in the validator software, raising concerns about network stability."
        },
        {
            "title": "Binance CEO Steps Down in $4.3 Billion Settlement with US Authorities",
            "source": "Reuters",
            "date": "2023-04-28",
            "summary": "Changpeng Zhao has pleaded guilty to violating US anti-money laundering laws and stepped down as CEO of Binance in a landmark settlement."
        },
        {
            "title": "BlackRock Files for Spot Ethereum ETF",
            "source": "CNBC",
            "date": "2023-04-25",
            "summary": "The world's largest asset manager has filed paperwork with the SEC to launch a spot Ethereum ETF, following its successful Bitcoin ETF launch."
        }
    ]
    return news

# Function to get crypto calendar events
def get_crypto_calendar():
    # This is a static list of events - in a real app you would fetch from an API
    events = [
        {
            "title": "Ethereum Pectra Upgrade",
            "date": "2024-11-01",
            "description": "Major upgrade including EIP-3074 (account abstraction) and EIP-7251 (staking changes)",
            "coin": "Ethereum",
            "type": "Protocol Upgrade"
        },
        {
            "title": "Solana Token Unlock",
            "date": "2024-06-15",
            "description": "Release of 50M SOL tokens (approx $5B) from early investors and team allocations",
            "coin": "Solana",
            "type": "Token Unlock"
        },
        {
            "title": "Avalanche Summit 2024",
            "date": "2024-09-10",
            "description": "Annual developer conference with expected announcements about Avalanche roadmap",
            "coin": "Avalanche",
            "type": "Conference"
        },
        {
            "title": "Bitcoin Taproot Activation Anniversary",
            "date": "2024-11-14",
            "description": "Celebration of Bitcoin's last major upgrade improving privacy and efficiency",
            "coin": "Bitcoin",
            "type": "Anniversary"
        },
        {
            "title": "Polygon zkEVM Mainnet Beta",
            "date": "2024-07-22",
            "description": "Full launch of Polygon's zero-knowledge EVM compatible rollup",
            "coin": "Polygon",
            "type": "Mainnet Launch"
        },
        {
            "title": "Cosmos Hub Governance Proposal Voting",
            "date": "2024-08-05",
            "description": "Community voting on proposal to reduce inflation rate from 14% to 10%",
            "coin": "Cosmos",
            "type": "Governance"
        }
    ]
    return events

# Overview Page
if page == "Overview":
    st.markdown("<h2 class='subheader'>Cryptocurrency Market Overview</h2>", unsafe_allow_html=True)
    
    # Load top cryptocurrencies data
    with st.spinner("Fetching market data..."):
        top_cryptos = get_top_cryptos(limit=25)
    
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
                    ]
                }
            ))
            
            gauge_fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=70, b=20),
            )
            
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Display top cryptocurrencies table
        st.markdown("<h3 class='subheader'>Top Cryptocurrencies by Market Cap</h3>", unsafe_allow_html=True)
        
        formatted_df = top_cryptos[['name', 'symbol', 'current_price', 'market_cap', 
                                 'price_change_percentage_24h', 'total_volume']].copy()
        
        formatted_df.columns = ['Name', 'Symbol', 'Price (USD)', 'Market Cap (USD)', 
                            '24h Change (%)', '24h Volume (USD)']
        
        # Format the values
        formatted_df['Symbol'] = formatted_df['Symbol'].str.upper()
        formatted_df['Price (USD)'] = formatted_df['Price (USD)'].apply(lambda x: f"${x:,.2f}")
        formatted_df['Market Cap (USD)'] = formatted_df['Market Cap (USD)'].apply(lambda x: f"${x:,.0f}")
        formatted_df['24h Change (%)'] = formatted_df['24h Change (%)'].apply(lambda x: f"{x:.2f}%")
        formatted_df['24h Volume (USD)'] = formatted_df['24h Volume (USD)'].apply(lambda x: f"${x:,.0f}")
        
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
        st.dataframe(
            formatted_df,
            use_container_width=True,
            hide_index=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.warning("No data available. Please check your internet connection or try again later.")

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
        
        tab1, tab2, tab3 = st.tabs(["Price Trend", "Technical Indicators", "Candlestick"])
        
        with tab1:
            fig = px.line(df, x='date', y='price', title=f"{selected_crypto} Price Trend")
            fig.update_layout(
                yaxis_title="Price (USD)",
                hovermode="x unified",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
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
            
            fig.update_layout(
                height=600,
                showlegend=True,
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            if len(df) > 7:
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
    
    else:
        st.warning(f"Could not fetch data for {selected_crypto}. Please try again later.")

# Comparison Page
elif page == "Comparison":
    st.markdown("<h2 class='subheader'>Cryptocurrency Comparison</h2>", unsafe_allow_html=True)
    
    # Select multiple cryptocurrencies to compare
    selected_cryptos = st.multiselect(
        "Select Cryptocurrencies to Compare", 
        list(popular_cryptos.keys()),
        default=["Bitcoin", "Ethereum"]
    )
    
    # Select timeframe
    timeframe = st.selectbox("Select Timeframe", ["7 days", "30 days", "90 days", "1 year"], key="compare_timeframe")
    
    timeframe_days = {
        "7 days": 7,
        "30 days": 30,
        "90 days": 90,
        "1 year": 365
    }
    
    if selected_cryptos:
        # Get data for all selected cryptos
        all_data = {}
        with st.spinner("Fetching comparison data..."):
            for crypto in selected_cryptos:
                df = get_crypto_data(popular_cryptos[crypto], days=timeframe_days[timeframe])
                if not df.empty:
                    all_data[crypto] = df
        
        if all_data:
            # Normalize prices for better comparison
            norm_data = {}
            for crypto, df in all_data.items():
                norm_df = df.copy()
                norm_df['norm_price'] = (norm_df['price'] / norm_df['price'].iloc[0]) * 100
                norm_data[crypto] = norm_df
            
            # Create comparison chart
            fig = go.Figure()
            
            for crypto, df in norm_data.items():
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['norm_price'],
                    name=crypto,
                    mode='lines'
                ))
            
            fig.update_layout(
                title=f"Normalized Price Comparison ({timeframe})",
                yaxis_title="Normalized Price (Base 100)",
                hovermode="x unified",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation matrix
            st.markdown("<h3 class='subheader'>Price Correlation</h3>", unsafe_allow_html=True)
            
            # Create a DataFrame with all prices
            corr_df = pd.DataFrame()
            for crypto, df in all_data.items():
                corr_df[crypto] = df.set_index('date')['price']
            
            # Calculate correlations
            correlation = corr_df.corr()
            
            # Display correlation heatmap
            fig = px.imshow(
                correlation,
                text_auto=True,
                color_continuous_scale='RdBu',
                zmin=-1,
                zmax=1,
                title="Cryptocurrency Price Correlation"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance metrics table
            st.markdown("<h3 class='subheader'>Performance Metrics</h3>", unsafe_allow_html=True)
            
            metrics = []
            for crypto, df in all_data.items():
                start_price = df['price'].iloc[0]
                end_price = df['price'].iloc[-1]
                return_pct = ((end_price - start_price) / start_price) * 100
                volatility = df['price'].pct_change().std() * np.sqrt(365) * 100  # Annualized
                metrics.append({
                    "Cryptocurrency": crypto,
                    "Start Price": f"${start_price:,.2f}",
                    "End Price": f"${end_price:,.2f}",
                    "Return %": f"{return_pct:.2f}%",
                    "Volatility %": f"{volatility:.2f}%"
                })
            
            metrics_df = pd.DataFrame(metrics)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Could not fetch data for selected cryptocurrencies. Please try again later.")
    else:
        st.info("Please select at least one cryptocurrency to compare.")

# Crypto News Page
elif page == "News":
    st.markdown("<h2 class='subheader'>Latest Cryptocurrency News</h2>", unsafe_allow_html=True)
    
    # Get news data
    news_items = get_crypto_news()
    
    # Display news cards
    for news in news_items:
        st.markdown(f"""
        <div class="news-card">
            <div class="news-title">{news['title']}</div>
            <div>{news['summary']}</div>
            <div class="news-source">{news['source']} • {news['date']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add a refresh button (though it won't actually fetch new data in this implementation)
    if st.button("Refresh News"):
        st.rerun()

# Crypto Calendar Page
elif page == "Calendar":
    st.markdown("<h2 class='subheader'>Cryptocurrency Events Calendar</h2>", unsafe_allow_html=True)
    
    # Get calendar events
    events = get_crypto_calendar()
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        filter_coin = st.selectbox("Filter by Coin", ["All"] + list(popular_cryptos.keys()))
    
    with col2:
        filter_type = st.selectbox("Filter by Event Type", ["All", "Protocol Upgrade", "Token Unlock", "Conference", "Mainnet Launch", "Governance"])
    
    # Apply filters
    filtered_events = events
    if filter_coin != "All":
        filtered_events = [e for e in filtered_events if e['coin'] == filter_coin]
    if filter_type != "All":
        filtered_events = [e for e in filtered_events if e['type'] == filter_type]
    
    # Display events
    if filtered_events:
        for event in filtered_events:
            st.markdown(f"""
            <div class="event-card">
                <div class="event-title">{event['title']}</div>
                <div>{event['description']}</div>
                <div class="event-date">
                    <span style="margin-right: 20px;">📅 {event['date']}</span>
                    <span style="margin-right: 20px;">🪙 {event['coin']}</span>
                    <span>🏷️ {event['type']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No events match your filters.")
    
    # Add a legend
    with st.expander("Event Types Legend"):
        st.markdown("""
        - **Protocol Upgrade**: Major network upgrades or hard forks
        - **Token Unlock**: Scheduled release of locked tokens
        - **Conference**: Major industry events and hackathons
        - **Mainnet Launch**: New blockchain networks going live
        - **Governance**: Important community votes and proposals
        """)

# Portfolio Tracker Page
elif page == "Portfolio":
    st.markdown("<h2 class='subheader'>Cryptocurrency Portfolio Tracker</h2>", unsafe_allow_html=True)
    
    # Initialize session state for portfolio
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = create_sample_portfolio()
    
    # Portfolio management
    with st.expander("Add/Edit Holdings"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            crypto_name = st.selectbox("Cryptocurrency", list(popular_cryptos.keys()))
        
        with col2:
            quantity = st.number_input("Quantity", min_value=0.0, value=1.0, step=0.1)
        
        with col3:
            purchase_price = st.number_input("Purchase Price (USD)", min_value=0.0, value=1000.0, step=1.0)
        
        with col4:
            purchase_date = st.date_input("Purchase Date", value=datetime.now() - timedelta(days=7))
        
        if st.button("Add to Portfolio"):
            # Get current price
            crypto_id = popular_cryptos[crypto_name]
            df = get_crypto_data(crypto_id, days=1)
            
            if not df.empty:
                current_price = df['price'].iloc[-1]
                current_value = quantity * current_price
                cost_basis = quantity * purchase_price
                profit_loss = current_value - cost_basis
                profit_loss_pct = (profit_loss / cost_basis) * 100
                
                new_holding = {
                    "crypto_name": crypto_name,
                    "crypto_id": crypto_id,
                    "quantity": quantity,
                    "purchase_price": purchase_price,
                    "purchase_date": purchase_date,
                    "current_price": current_price,
                    "current_value": current_value,
                    "profit_loss": profit_loss,
                    "profit_loss_percentage": profit_loss_pct
                }
                
                st.session_state.portfolio.append(new_holding)
                st.success("Holding added to portfolio!")
            else:
                st.error("Could not fetch current price. Please try again later.")
    
    # Display portfolio
    if st.session_state.portfolio:
        # Calculate portfolio summary
        total_value = sum(item['current_value'] for item in st.session_state.portfolio)
        total_cost = sum(item['quantity'] * item['purchase_price'] for item in st.session_state.portfolio)
        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost != 0 else 0
        
        # Display portfolio metrics
        st.markdown("<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 20px;'>", unsafe_allow_html=True)
        st.markdown(display_metric("Total Value", f"${total_value:,.2f}"), unsafe_allow_html=True)
        st.markdown(display_metric("Total Cost", f"${total_cost:,.2f}"), unsafe_allow_html=True)
        st.markdown(display_metric("Total P/L", f"${total_pnl:,.2f}", delta=f"{total_pnl_pct:.2f}%"), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Portfolio holdings table
        st.markdown("<h3 class='subheader'>Your Holdings</h3>", unsafe_allow_html=True)
        
        # Prepare data for display
        display_data = []
        for item in st.session_state.portfolio:
            display_data.append({
                "Cryptocurrency": item['crypto_name'],
                "Quantity": item['quantity'],
                "Avg Cost": f"${item['purchase_price']:,.2f}",
                "Current Price": f"${item['current_price']:,.2f}",
                "Current Value": f"${item['current_value']:,.2f}",
                "P/L": f"${item['profit_loss']:,.2f}",
                "P/L %": f"{item['profit_loss_percentage']:.2f}%"
            })
        
        holdings_df = pd.DataFrame(display_data)
        st.dataframe(holdings_df, use_container_width=True, hide_index=True)
        
        # Portfolio allocation chart
        st.markdown("<h3 class='subheader'>Portfolio Allocation</h3>", unsafe_allow_html=True)
        
        fig = px.pie(
            pd.DataFrame(st.session_state.portfolio),
            values='current_value',
            names='crypto_name',
            title='Portfolio Allocation by Value',
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance over time chart
        st.markdown("<h3 class='subheader'>Portfolio Performance</h3>", unsafe_allow_html=True)
        
        # Simplified performance visualization
        dates = pd.date_range(end=datetime.now(), periods=30)
        perf_data = []
        
        for i, date in enumerate(dates):
            day_value = 0
            for item in st.session_state.portfolio:
                growth_factor = 1 + (item['profit_loss_percentage'] / 100 * (i / 30))
                day_value += item['quantity'] * item['purchase_price'] * growth_factor
            perf_data.append(day_value)
        
        fig = px.line(
            x=dates,
            y=perf_data,
            title="Portfolio Value Over Time (Simulated)",
            labels={'x': 'Date', 'y': 'Portfolio Value (USD)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Your portfolio is empty. Add some holdings to get started.")

# Footer
st.markdown("""
<div class="footer">
    <p>Cryptocurrency Dashboard • Powered by CoinGecko API</p>
    <p>Data updates every 5 minutes</p>
</div>
""", unsafe_allow_html=True)
