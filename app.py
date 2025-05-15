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
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64

# Set page config with minimalist look
st.set_page_config(
    page_title="Crypto Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for minimalist UI
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #ffffff;
    }
    /* Metrics */
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
    }
    .metric-change {
        font-size: 14px;
    }
    .positive {
        color: #28a745;
    }
    .negative {
        color: #dc3545;
    }
    /* Hide elements */
    [data-testid="stToolbar"], [data-testid="stDecoration"], 
    [data-testid="stStatusWidget"], [data-testid="stHeader"] {
        display: none;
    }
    /* Compact layout */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    /* Hide fullscreen button */
    .stPlotlyFullScreenButton {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Cache data functions (unchanged from original)
@st.cache_data(ttl=3600, show_spinner="Fetching cryptocurrency data...")
def get_crypto_data(ticker, period="1y"):
    try:
        crypto = yf.Ticker(ticker)
        hist = crypto.history(period=period, interval="1d")
        if hist.empty:
            st.error(f"No data available for {ticker}")
            return pd.DataFrame()
        return hist
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner="Fetching multiple cryptocurrency data...")
def get_multiple_cryptos(tickers, period="1y"):
    data = {}
    progress_bar = st.progress(0)
    for i, ticker in enumerate(tickers):
        try:
            progress_bar.progress((i + 1) / len(tickers), text=f"Fetching {ticker}...")
            crypto = yf.Ticker(ticker)
            hist = crypto.history(period=period, interval="1d")
            if not hist.empty:
                data[ticker] = hist
        except Exception as e:
            st.warning(f"Could not fetch data for {ticker}: {str(e)}")
    progress_bar.empty()
    return data

# Forecasting function (unchanged from original)
def forecast_crypto_price(data, days=30):
    if data.empty or len(data) < 30:
        return None, None, None, None
    
    try:
        df = data[['Close']].copy()
        df['Date'] = df.index
        df['Days'] = (df['Date'] - df['Date'].min()).dt.days
        df = df[['Days', 'Close']].dropna()
        
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_30'] = df['Close'].rolling(window=30).mean()
        df['Daily_Return'] = df['Close'].pct_change()
        df = df.dropna()
        
        X = df[['Days', 'MA_7', 'MA_30', 'Daily_Return']]
        y = df['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        last_day = df['Days'].max()
        future_days = pd.DataFrame({
            'Days': range(last_day + 1, last_day + days + 1),
            'MA_7': df['MA_7'].iloc[-1],
            'MA_30': df['MA_30'].iloc[-1],
            'Daily_Return': df['Daily_Return'].iloc[-1]
        })
        
        future_prices = []
        for i in range(days):
            if i == 0:
                pred = model.predict(future_days.iloc[[i]])[0]
            else:
                ma_7_values = list(df['Close'].iloc[-6:]) + (future_prices[-6:] if len(future_prices) > 6 else [])
                future_days.at[future_days.index[i], 'MA_7'] = np.mean(ma_7_values)
                
                ma_30_values = list(df['Close'].iloc[-29:]) + (future_prices[-29:] if len(future_prices) > 29 else [])
                future_days.at[future_days.index[i], 'MA_30'] = np.mean(ma_30_values)
                
                pred = model.predict(future_days.iloc[[i]])[0]
            future_prices.append(pred)
        
        last_date = data.index.max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days
        )
        
        return future_dates, future_prices, mae, rmse
    
    except Exception as e:
        st.error(f"Error in forecasting: {str(e)}")
        return None, None, None, None

# RSI calculation (unchanged from original)
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Simplified sidebar matching the image
def sidebar():
    with st.sidebar:
        st.title("Portfolio")
        selected_cryptos = ["BTC-USD", "ETH-USD"]  # Default selection as in image
        
        st.markdown("---")
        st.subheader("Total")
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">$28,964</div>
            <div class="metric-change positive">+3.2%</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("24h Change")
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">23,432</div>
            <div class="metric-change positive">+2.5%</div>
        </div>
        """, unsafe_allow_html=True)
        
        return selected_cryptos, "1y", 30  # Default period and forecast days

# Simplified price tab matching the image
def price_tab(selected_cryptos, time_period):
    col1, col2, col3 = st.columns([1,3,1])
    
    with col1:
        st.markdown("""
        <div style="margin-bottom: 20px;">
            <div style="font-size: 16px; font-weight: bold; margin-bottom: 5px;">BTC</div>
            <div class="metric-card">
                <div class="metric-value">15%</div>
            </div>
        </div>
        <div style="margin-bottom: 20px;">
            <div style="font-size: 16px; font-weight: bold; margin-bottom: 5px;">ETH</div>
            <div class="metric-card">
                <div class="metric-value">20%</div>
            </div>
        </div>
        <div style="margin-bottom: 20px;">
            <div style="font-size: 16px; font-weight: bold; margin-bottom: 5px;">BNB</div>
            <div class="metric-card">
                <div class="metric-value">25%</div>
            </div>
        </div>
        <div style="margin-bottom: 20px;">
            <div style="font-size: 16px; font-weight: bold; margin-bottom: 5px;">SOL</div>
            <div class="metric-card">
                <div class="metric-value">15%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Get data
        crypto_data = get_multiple_cryptos(selected_cryptos, time_period)
        
        # Plot price chart
        fig = go.Figure()
        
        for ticker, data in crypto_data.items():
            if not data.empty:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    name=ticker.split("-")[0],
                    mode='lines',
                    line=dict(width=2),
                    hovertemplate="%{y:$,.2f}<extra>%{x|%b %d, %Y}</extra>"
                ))
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=True,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown("""
        <div style="margin-bottom: 20px;">
            <div style="font-size: 16px; font-weight: bold; margin-bottom: 5px;">BTC</div>
            <div class="metric-card">
                <div class="metric-change positive">+2.5%</div>
            </div>
        </div>
        <div style="margin-bottom: 20px;">
            <div style="font-size: 16px; font-weight: bold; margin-bottom: 5px;">ETH</div>
            <div class="metric-card">
                <div class="metric-change positive">+1.2%</div>
            </div>
        </div>
        <div style="margin-bottom: 20px;">
            <div style="font-size: 16px; font-weight: bold; margin-bottom: 5px;">BNB</div>
            <div class="metric-card">
                <div class="metric-change positive">+2.3%</div>
            </div>
        </div>
        <div style="margin-bottom: 20px;">
            <div style="font-size: 16px; font-weight: bold; margin-bottom: 5px;">SOL</div>
            <div class="metric-card">
                <div class="metric-change">432</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Main App with simplified interface
def main():
    # Sidebar
    selected_cryptos, time_period, forecast_days = sidebar()
    
    # Main content
    price_tab(selected_cryptos, time_period)
    
    # Hidden tabs (keeping functionality but not showing in UI)
    with st.expander("Advanced Tools", expanded=False):
        tab1, tab2, tab3 = st.tabs(["Technical Analysis", "Forecast", "News"])
        
        with tab1:
            # Original analysis_tab functionality
            if not selected_cryptos:
                st.warning("Please select at least one cryptocurrency.")
            else:
                crypto_data = get_multiple_cryptos(selected_cryptos, time_period)
                if crypto_data:
                    st.write("Technical analysis tools would appear here")
        
        with tab2:
            # Original forecast_tab functionality
            if not selected_cryptos:
                st.warning("Please select at least one cryptocurrency.")
            else:
                crypto_data = get_multiple_cryptos(selected_cryptos, time_period)
                if crypto_data:
                    st.write("Forecast tools would appear here")
        
        with tab3:
            # Original news_tab functionality
            st.write("News and resources would appear here")

if __name__ == "__main__":
    main()
