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
        color_class = "crypto-positive" if float(delta.replace("%", "")) >= 0 else "crypto-negative"
        html += f'<div class="{color_class}">{delta}</div>'
    
    html += "</div>"
    return html

# Function to calculate technical indicators
def calculate_indicators(df):
    # Calculate Simple Moving Averages
    df['SMA_7'] = df['price'].rolling(window=7).mean()
    df['SMA_25'] = df['price'].rolling(window=25).mean()
    
    # Calculate Relative Strength Index (RSI)
    delta = df['price'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    df['EMA_12'] = df['price'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    
    # Calculate Bollinger Bands
    df['MA_20'] = df['price'].rolling(window=20).mean()
    df['STD_20'] = df['price'].rolling(window=20).std()
    df['Upper_Band'] = df['MA_20'] + (df['STD_20'] * 2)
    df['Lower_Band'] = df['MA_20'] - (df['STD_20'] * 2)
    
    return df

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
        bitcoin_dominance = (top_cryptos[top_cryptos['id'] == 'bitcoin']['market_cap'].values[0] / 
                           top_cryptos['market_cap'].sum()) * 100
        st.markdown(display_metric("Bitcoin Dominance", f"{bitcoin_dominance:.2f}", suffix="%"), unsafe_allow_html=True)
        
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
            
            for i, crypto in enumerate(top_cryptos.head(5)['id']):
                df = get_crypto_data(crypto, days=7)
                if not df.empty:
                    trend_fig.add_trace(go.Scatter(
                        x=df['date'], 
                        y=df['price'],
                        mode='lines',
                        name=top_cryptos.iloc[i]['name']
                    ))
            
            trend_fig.update_layout(
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode="x unified"
            )
            st.plotly_chart(trend_fig, use_container_width=True)
        
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
        
        # Add styling to the table
        def highlight_change(val):
            color = '#4CAF50' if val > 0 else '#F44336'
            return f'color: {color}; font-weight: bold'
        
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
            demo_data = {
                "name": ["Bitcoin", "Ethereum", "Binance Coin", "Cardano", "Solana"],
                "symbol": ["btc", "eth", "bnb", "ada", "sol"],
                "current_price": [45000, 3200, 450, 1.2, 100],
                "market_cap": [800000000000, 350000000000, 80000000000, 40000000000, 30000000000],
                "price_change_percentage_24h": [2.5, -1.3, 0.8, -2.1, 3.7],
                "total_volume": [30000000000, 20000000000, 5000000000, 2000000000, 3000000000]
            }
            demo_df = pd.DataFrame(demo_data)
            
            # Display demo data with a clear notice
            st.info("⚠️ This is demo data for display purposes only. It does not reflect real market conditions.")
            st.dataframe(demo_df)${int(top_cryptos['total_volume'].sum() / 1e9)}B")
        
        with col3:
            bitcoin_dominance = (top_cryptos[top_cryptos['id'] == 'bitcoin']['market_cap'].values[0] / 
                               top_cryptos['market_cap'].sum()) * 100
            st.metric("Bitcoin Dominance", f"{bitcoin_dominance:.2f}%")
        
        with col4:
            avg_change = top_cryptos['price_change_percentage_24h'].mean()
            st.metric("Avg 24h Change", f"{avg_change:.2f}%", 
                     delta=f"{avg_change:.2f}%")
        
        # Display top cryptocurrencies table
        st.subheader("Top Cryptocurrencies by Market Cap")
        
        # Format the table
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
        
        st.dataframe(formatted_df, use_container_width=True)
        
        # Market Cap Distribution
        st.subheader("Market Cap Distribution")
        fig = px.pie(top_cryptos.head(10), values='market_cap', names='name', 
                    title='Top 10 Cryptocurrencies by Market Cap')
        st.plotly_chart(fig, use_container_width=True)
        
        # 24h Price Change
        st.subheader("24h Price Change")
        sorted_by_change = top_cryptos.sort_values(by='price_change_percentage_24h', ascending=False).head(10)
        fig = px.bar(sorted_by_change, x='name', y='price_change_percentage_24h', 
                    title='Top 10 Performers (24h)',
                    labels={'name': 'Cryptocurrency', 'price_change_percentage_24h': '24h Change (%)'},
                    color='price_change_percentage_24h',
                    color_continuous_scale=['red', 'green'])
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("No data available. Please check your internet connection or try again later.")

# Price Analysis Page
elif page == "Price Analysis":
    st.header("Cryptocurrency Price Analysis")
    
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
    crypto_id = popular_cryptos[selected_crypto]
    df = get_crypto_data(crypto_id, timeframe_days[timeframe])
    
    if not df.empty:
        # Price chart
        st.subheader(f"{selected_crypto} Price Chart")
        fig = px.line(df, x='date', y='price', title=f"{selected_crypto} Price (USD)")
        fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate technical indicators
        df_indicators = calculate_indicators(df)
        
        # Technical indicators
        st.subheader("Technical Indicators")
        tab1, tab2 = st.tabs(["Moving Averages", "RSI"])
        
        with tab1:
            # Moving Averages
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_indicators['date'], y=df_indicators['price'], 
                                    mode='lines', name='Price'))
            fig.add_trace(go.Scatter(x=df_indicators['date'], y=df_indicators['SMA_7'], 
                                    mode='lines', name='7-Day SMA', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=df_indicators['date'], y=df_indicators['SMA_25'], 
                                    mode='lines', name='25-Day SMA', line=dict(dash='dot')))
            fig.update_layout(title=f"{selected_crypto} Price with Moving Averages",
                            xaxis_title="Date", yaxis_title="Price (USD)")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # RSI
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_indicators['date'], y=df_indicators['RSI'], 
                                    mode='lines', name='RSI'))
            
            # Add overbought and oversold lines
            fig.add_shape(type='line', x0=df_indicators['date'].iloc[0], y0=70, 
                        x1=df_indicators['date'].iloc[-1], y1=70,
                        line=dict(color='red', dash='dash'))
            fig.add_shape(type='line', x0=df_indicators['date'].iloc[0], y0=30, 
                        x1=df_indicators['date'].iloc[-1], y1=30,
                        line=dict(color='green', dash='dash'))
            
            fig.update_layout(title=f"{selected_crypto} RSI (14-Day)",
                            xaxis_title="Date", yaxis_title="RSI Value",
                            yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig, use_container_width=True)
        
        # Price statistics
        st.subheader("Price Statistics")
        current_price = df["price"].iloc[-1]
        price_change = ((current_price - df["price"].iloc[0]) / df["price"].iloc[0]) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            st.metric(f"{timeframe} Change", f"{price_change:.2f}%", 
                    delta=f"{price_change:.2f}%")
        
        with col3:
            st.metric("All-Time High", f"${df['price'].max():.2f}")
        
        with col4:
            volatility = df["price"].pct_change().std() * 100
            st.metric("Volatility", f"{volatility:.2f}%")
    
    else:
        st.warning(f"No data available for {selected_crypto}. Please check your internet connection or try again later.")

# Comparison Page
elif page == "Comparison":
    st.header("Cryptocurrency Comparison")
    
    # Select cryptocurrencies to compare
    col1, col2, col3 = st.columns(3)
    
    with col1:
        crypto1 = st.selectbox("Select First Cryptocurrency", list(popular_cryptos.keys()), index=0)
    
    with col2:
        crypto2 = st.selectbox("Select Second Cryptocurrency", list(popular_cryptos.keys()), index=1)
    
    with col3:
        timeframe = st.selectbox("Select Timeframe", ["30 days", "90 days", "1 year"], index=0)
    
    timeframe_days = {
        "30 days": 30,
        "90 days": 90,
        "1 year": 365
    }
    
    # Get data
    df1 = get_crypto_data(popular_cryptos[crypto1], timeframe_days[timeframe])
    df2 = get_crypto_data(popular_cryptos[crypto2], timeframe_days[timeframe])
    
    if not df1.empty and not df2.empty:
        # Normalize data for comparison (starting from 100)
        df1['normalized'] = 100 * (df1['price'] / df1['price'].iloc[0])
        df2['normalized'] = 100 * (df2['price'] / df2['price'].iloc[0])
        
        # Comparison chart
        st.subheader(f"Price Comparison: {crypto1} vs {crypto2}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df1['date'], y=df1['normalized'], mode='lines', name=crypto1))
        fig.add_trace(go.Scatter(x=df2['date'], y=df2['normalized'], mode='lines', name=crypto2))
        fig.update_layout(title=f"Normalized Price Comparison (Base 100)",
                        xaxis_title="Date", yaxis_title="Normalized Price")
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.subheader("Correlation Analysis")
        
        # Merge dataframes for correlation
        df_merge = pd.merge(df1, df2, on='date', suffixes=('_1', '_2'))
        corr = df_merge['price_1'].corr(df_merge['price_2'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Price Correlation", f"{corr:.2f}")
            
            # Scatter plot
            fig = px.scatter(df_merge, x='price_1', y='price_2', trendline="ols",
                            title=f"{crypto1} vs {crypto2} Price Correlation",
                            labels={'price_1': f"{crypto1} Price (USD)", 'price_2': f"{crypto2} Price (USD)"})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Performance comparison
            perf1 = ((df1['price'].iloc[-1] - df1['price'].iloc[0]) / df1['price'].iloc[0]) * 100
            perf2 = ((df2['price'].iloc[-1] - df2['price'].iloc[0]) / df2['price'].iloc[0]) * 100
            
            st.metric(f"{crypto1} {timeframe} Performance", f"{perf1:.2f}%", delta=f"{perf1:.2f}%")
            st.metric(f"{crypto2} {timeframe} Performance", f"{perf2:.2f}%", delta=f"{perf2:.2f}%")
            
            # Volatility comparison
            vol1 = df1["price"].pct_change().std() * 100
            vol2 = df2["price"].pct_change().std() * 100
            
            st.metric(f"{crypto1} Volatility", f"{vol1:.2f}%")
            st.metric(f"{crypto2} Volatility", f"{vol2:.2f}%")
    
    else:
        st.warning("No data available for comparison. Please check your internet connection or try again later.")

# Portfolio Tracker Page
elif page == "Portfolio Tracker":
    st.header("Portfolio Tracker")
    
    # Initialize session state
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = []
    
    # Form to add cryptocurrency to portfolio
    st.subheader("Add Cryptocurrency to Portfolio")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        crypto_name = st.selectbox("Select Cryptocurrency", list(popular_cryptos.keys()))
    
    with col2:
        quantity = st.number_input("Quantity", min_value=0.0, step=0.01)
    
    with col3:
        purchase_price = st.number_input("Purchase Price (USD)", min_value=0.0, step=0.01)
    
    with col4:
        purchase_date = st.date_input("Purchase Date", value=datetime.now())
    
    if st.button("Add to Portfolio"):
        # Add to portfolio
        crypto_id = popular_cryptos[crypto_name]
        
        # Get current price
        df = get_crypto_data(crypto_id, days=1)
        if not df.empty:
            current_price = df["price"].iloc[-1]
            
            # Calculate current value
            current_value = quantity * current_price
            purchase_value = quantity * purchase_price
            profit_loss = current_value - purchase_value
            profit_loss_percentage = (profit_loss / purchase_value) * 100 if purchase_value > 0 else 0
            
            # Add to portfolio
            st.session_state.portfolio.append({
                "crypto_name": crypto_name,
                "crypto_id": crypto_id,
                "quantity": quantity,
                "purchase_price": purchase_price,
                "purchase_date": purchase_date,
                "current_price": current_price,
                "current_value": current_value,
                "profit_loss": profit_loss,
                "profit_loss_percentage": profit_loss_percentage
            })
            
            st.success(f"Added {quantity} {crypto_name} to your portfolio!")
        else:
            st.error(f"Failed to fetch current price for {crypto_name}.")
    
    # Display portfolio
    if st.session_state.portfolio:
        st.subheader("Your Portfolio")
        
        # Create portfolio dataframe
        portfolio_df = pd.DataFrame(st.session_state.portfolio)
        
        # Display portfolio stats
        total_value = portfolio_df["current_value"].sum()
        total_cost = (portfolio_df["quantity"] * portfolio_df["purchase_price"]).sum()
        total_profit_loss = portfolio_df["profit_loss"].sum()
        total_profit_loss_percentage = (total_profit_loss / total_cost) * 100 if total_cost > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Portfolio Value", f"${total_value:.2f}")
        
        with col2:
            st.metric("Total Investment", f"${total_cost:.2f}")
        
        with col3:
            st.metric("Total Profit/Loss", f"${total_profit_loss:.2f}", 
                    delta=f"{total_profit_loss_percentage:.2f}%")
        
        # Display portfolio table
        display_df = portfolio_df[["crypto_name", "quantity", "purchase_price", "current_price", 
                                "current_value", "profit_loss", "profit_loss_percentage"]].copy()
        
        display_df.columns = ["Cryptocurrency", "Quantity", "Purchase Price (USD)", "Current Price (USD)", 
                            "Current Value (USD)", "Profit/Loss (USD)", "Profit/Loss (%)"]
        
        # Format the values
        display_df["Purchase Price (USD)"] = display_df["Purchase Price (USD)"].apply(lambda x: f"${x:.2f}")
        display_df["Current Price (USD)"] = display_df["Current Price (USD)"].apply(lambda x: f"${x:.2f}")
        display_df["Current Value (USD)"] = display_df["Current Value (USD)"].apply(lambda x: f"${x:.2f}")
        display_df["Profit/Loss (USD)"] = display_df["Profit/Loss (USD)"].apply(lambda x: f"${x:.2f}")
        display_df["Profit/Loss (%)"] = display_df["Profit/Loss (%)"].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Portfolio Distribution
        st.subheader("Portfolio Distribution")
        fig = px.pie(portfolio_df, values='current_value', names='crypto_name', 
                    title='Portfolio Asset Allocation')
        st.plotly_chart(fig, use_container_width=True)
        
        # Clear portfolio button
        if st.button("Clear Portfolio"):
            st.session_state.portfolio = []
            st.success("Portfolio cleared!")
    else:
        st.info("Your portfolio is empty. Add some cryptocurrencies to track them.")
        
        # Sample portfolio demo
        if st.button("Load Sample Portfolio"):
            # Bitcoin
            crypto_id = popular_cryptos["Bitcoin"]
            df = get_crypto_data(crypto_id, days=1)
            if not df.empty:
                current_price = df["price"].iloc[-1]
                quantity = 0.5
                purchase_price = current_price * 0.9  # 10% lower than current price
                current_value = quantity * current_price
                purchase_value = quantity * purchase_price
                profit_loss = current_value - purchase_value
                profit_loss_percentage = (profit_loss / purchase_value) * 100
                
                st.session_state.portfolio.append({
                    "crypto_name": "Bitcoin",
                    "crypto_id": crypto_id,
                    "quantity": quantity,
                    "purchase_price": purchase_price,
                    "purchase_date": datetime.now() - timedelta(days=30),
                    "current_price": current_price,
                    "current_value": current_value,
                    "profit_loss": profit_loss,
                    "profit_loss_percentage": profit_loss_percentage
                })
            
            # Ethereum
            crypto_id = popular_cryptos["Ethereum"]
            df = get_crypto_data(crypto_id, days=1)
            if not df.empty:
                current_price = df["price"].iloc[-1]
                quantity = 3.0
                purchase_price = current_price * 0.95  # 5% lower than current price
                current_value = quantity * current_price
                purchase_value = quantity * purchase_price
                profit_loss = current_value - purchase_value
                profit_loss_percentage = (profit_loss / purchase_value) * 100
                
                st.session_state.portfolio.append({
                    "crypto_name": "Ethereum",
                    "crypto_id": crypto_id,
                    "quantity": quantity,
                    "purchase_price": purchase_price,
                    "purchase_date": datetime.now() - timedelta(days=14),
                    "current_price": current_price,
                    "current_value": current_value,
                    "profit_loss": profit_loss,
                    "profit_loss_percentage": profit_loss_percentage
                })
            
            # Cardano
            crypto_id = popular_cryptos["Cardano"]
            df = get_crypto_data(crypto_id, days=1)
            if not df.empty:
                current_price = df["price"].iloc[-1]
                quantity = 1000.0
                purchase_price = current_price * 1.05  # 5% higher than current price
                current_value = quantity * current_price
                purchase_value = quantity * purchase_price
                profit_loss = current_value - purchase_value
                profit_loss_percentage = (profit_loss / purchase_value) * 100
                
                st.session_state.portfolio.append({
                    "crypto_name": "Cardano",
                    "crypto_id": crypto_id,
                    "quantity": quantity,
                    "purchase_price": purchase_price,
                    "purchase_date": datetime.now() - timedelta(days=7),
                    "current_price": current_price,
                    "current_value": current_value,
                    "profit_loss": profit_loss,
                    "profit_loss_percentage": profit_loss_percentage
                })
            
            st.success("Sample portfolio loaded!")

# Add footer
st.markdown("---")
st.markdown("📊 Crypto Dashboard - Built with Streamlit")
