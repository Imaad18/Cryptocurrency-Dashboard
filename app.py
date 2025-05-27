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

# Set page config with professional look
st.set_page_config(
    page_title="CryptoVision Pro",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# CryptoVision Pro - Advanced Cryptocurrency Analytics"
    }
)

# Custom CSS for professional UI
def inject_futuristic_css():
    futuristic_css = """
    <style>
    /* === GLOBAL === */
    html, body, .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        font-family: 'Segoe UI', sans-serif;
        color: #ecf0f1;
    }

    /* === HEADINGS === */
    h1, h2, h3, h4, h5, h6 {
        color: #00ffe7;
        text-shadow: 0 0 5px #00ffe7, 0 0 10px #00ffe7;
    }

    /* === BUTTONS === */
    button[kind="primary"] {
        background: #00ffe7;
        color: #000;
        border: none;
        border-radius: 10px;
        box-shadow: 0 0 10px #00ffe7;
        transition: 0.3s ease-in-out;
    }
    button[kind="primary"]:hover {
        background: #00bfa6;
        box-shadow: 0 0 20px #00ffe7;
        transform: scale(1.05);
    }

    /* === METRICS === */
    .st-emotion-cache-1xarl3l {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    /* === CARDS & CONTAINERS === */
    .resource-card {
        background: rgba(0, 0, 0, 0.4);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 0 10px rgba(0, 255, 231, 0.2);
        transition: 0.4s ease-in-out;
    }
    .resource-card:hover {
        box-shadow: 0 0 25px rgba(0, 255, 231, 0.4);
        transform: translateY(-5px);
    }

    /* === SIDEBAR === */
    section[data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.85);
        color: #00ffe7;
        border-right: 2px solid #00ffe7;
    }

    /* === SLIDER === */
    .stSlider > div[data-baseweb="slider"] {
        background-color: rgba(255, 255, 255, 0.1);
    }

    /* === TABS === */
    .stTabs [role="tab"] {
        background: #000;
        color: #00ffe7;
        border-radius: 10px 10px 0 0;
        border-bottom: 2px solid #00ffe7;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background: #00ffe7;
        color: #000;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 255, 231, 0.3);
    }

    /* === DATAFRAMES === */
    .stDataFrame {
        border: 1px solid #00ffe7;
        border-radius: 12px;
        overflow: hidden;
    }

    /* === FOOTER === */
    footer {
        color: #aaa;
        text-align: center;
        font-size: 0.8em;
        padding-top: 10px;
    }
    </style>
    """
    st.markdown(futuristic_css, unsafe_allow_html=True)


# Cache data functions with enhanced error handling
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

# Enhanced forecasting function with multiple models
def forecast_crypto_price(data, days=30):
    if data.empty or len(data) < 30:  # Need at least 30 days of data
        return None, None, None, None
    
    try:
        # Prepare data
        df = data[['Close']].copy()
        df['Date'] = df.index
        df['Days'] = (df['Date'] - df['Date'].min()).dt.days
        df = df[['Days', 'Close']].dropna()
        
        # Create additional features
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_30'] = df['Close'].rolling(window=30).mean()
        df['Daily_Return'] = df['Close'].pct_change()
        df = df.dropna()
        
        # Split data
        X = df[['Days', 'MA_7', 'MA_30', 'Daily_Return']]
        y = df['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Future prediction
        last_day = df['Days'].max()
        future_days = pd.DataFrame({
            'Days': range(last_day + 1, last_day + days + 1),
            'MA_7': df['MA_7'].iloc[-1],
            'MA_30': df['MA_30'].iloc[-1],
            'Daily_Return': df['Daily_Return'].iloc[-1]
        })
        
        # Adjust future MA values based on predicted prices
        future_prices = []
        for i in range(days):
            if i == 0:
                pred = model.predict(future_days.iloc[[i]])[0]
            else:
                # Update MA values based on previous predictions
                # Fixed the list concatenation syntax here
                ma_7_values = list(df['Close'].iloc[-6:]) + (future_prices[-6:] if len(future_prices) > 6 else [])
                future_days.at[future_days.index[i], 'MA_7'] = np.mean(ma_7_values)
                
                ma_30_values = list(df['Close'].iloc[-29:]) + (future_prices[-29:] if len(future_prices) > 29 else [])
                future_days.at[future_days.index[i], 'MA_30'] = np.mean(ma_30_values)
                
                pred = model.predict(future_days.iloc[[i]])[0]
            future_prices.append(pred)
        
        # Create dates for future predictions
        last_date = data.index.max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days
        )
        
        return future_dates, future_prices, mae, rmse
    
    except Exception as e:
        st.error(f"Error in forecasting: {str(e)}")
        return None, None, None, None


# Professional sidebar with enhanced features
def sidebar():
    with st.sidebar:
        # Modern crypto-themed image
        st.markdown(
            """
            <div style="text-align: center;">
                <img src="https://images.unsplash.com/photo-1639762681057-408e52192e55?q=80&w=2232&auto=format&fit=crop" 
                     class="sidebar-image" 
                     alt="Cryptocurrency concept">
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.title("CryptoVision Pro")
        st.markdown("---")
        
        # Popular cryptocurrencies with logos
        crypto_options = {
            "Bitcoin (BTC)": "BTC-USD",
            "Ethereum (ETH)": "ETH-USD",
            "Binance Coin (BNB)": "BNB-USD",
            "Cardano (ADA)": "ADA-USD",
            "Solana (SOL)": "SOL-USD",
            "XRP (XRP)": "XRP-USD",
            "Polkadot (DOT)": "DOT-USD",
            "Dogecoin (DOGE)": "DOGE-USD",
            "Shiba Inu (SHIB)": "SHIB-USD",
            "Polygon (MATIC)": "MATIC-USD",
            "Litecoin (LTC)": "LTC-USD",
            "Chainlink (LINK)": "LINK-USD"
        }
        
        selected_names = st.multiselect(
            "Select Cryptocurrencies",
            list(crypto_options.keys()),
            default=["Bitcoin (BTC)", "Ethereum (ETH)"],
            help="Select multiple cryptocurrencies to compare"
        )
        
        selected_cryptos = [crypto_options[name] for name in selected_names]
        
        st.markdown("---")
        
        # Time period selection with presets
        time_period = st.selectbox(
            "Select Time Period",
            options=[
                "1 Week", "1 Month", "3 Months", 
                "6 Months", "1 Year", "2 Years", 
                "5 Years", "All Time"
            ],
            index=4,
            help="Select the historical time period to analyze"
        )
        
        # Map selection to Yahoo Finance periods
        period_map = {
            "1 Week": "5d",
            "1 Month": "1mo",
            "3 Months": "3mo",
            "6 Months": "6mo",
            "1 Year": "1y",
            "2 Years": "2y",
            "5 Years": "5y",
            "All Time": "max"
        }
        yf_period = period_map[time_period]
        
        st.markdown("---")
        
        # Forecast settings
        st.subheader("Forecast Settings")
        forecast_days = st.slider(
            "Forecast Horizon (Days)",
            min_value=7,
            max_value=90,
            value=30,
            step=7,
            help="Number of days to forecast into the future"
        )
        
        st.markdown("---")
        
        # Display market summary
        st.subheader("Market Summary")
        try:
            btc = yf.Ticker("BTC-USD")
            btc_info = btc.fast_info
            st.metric("Bitcoin Price", f"${btc_info.last_price:,.2f}", 
                    f"{btc_info.last_price - btc_info.previous_close:,.2f}")
            st.metric("24h Change", f"{((btc_info.last_price / btc_info.previous_close - 1) * 100):.2f}%")
        except:
            st.warning("Couldn't fetch market data")
        
        return selected_cryptos, yf_period, forecast_days

# Enhanced Price Tab with professional charts
def price_tab(selected_cryptos, time_period):
    st.header("📈 Cryptocurrency Price Analysis")
    st.markdown("Analyze historical price trends and performance metrics")
    
    if not selected_cryptos:
        st.warning("Please select at least one cryptocurrency from the sidebar.")
        return
    
    with st.spinner("Loading price data..."):
        crypto_data = get_multiple_cryptos(selected_cryptos, time_period)
    
    if not crypto_data:
        st.error("No data available for the selected cryptocurrencies.")
        return
    
    # Create tabs for different chart types
    chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Interactive Chart", "Candlestick Chart", "Performance Metrics"])
    
    with chart_tab1:
        # Plot price chart with enhanced features
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
            title="Cryptocurrency Price Trends",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode="x unified",
            height=600,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, b=50, t=80),
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            ),
            yaxis=dict(
                tickprefix="$",
                tickformat=",.0f"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_tab2:
        # Candlestick chart for the first selected crypto
        if len(selected_cryptos) > 0:
            primary_crypto = selected_cryptos[0]
            if primary_crypto in crypto_data and not crypto_data[primary_crypto].empty:
                data = crypto_data[primary_crypto]
                fig = go.Figure(data=[go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name=primary_crypto.split("-")[0]
                )])
                
                fig.update_layout(
                    title=f"{primary_crypto.split('-')[0]} Candlestick Chart",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    height=600,
                    template="plotly_white",
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No candlestick data available for {primary_crypto}")
    
    with chart_tab3:
        # Performance metrics table
        if len(crypto_data) > 0:
            metrics = []
            for ticker, data in crypto_data.items():
                if not data.empty and len(data) > 1:
                    start_price = data['Close'].iloc[0]
                    end_price = data['Close'].iloc[-1]
                    pct_change = (end_price / start_price - 1) * 100
                    volatility = data['Close'].pct_change().std() * np.sqrt(365)  # Annualized volatility
                    metrics.append({
                        'Cryptocurrency': ticker.split("-")[0],
                        'Start Price': f"${start_price:,.2f}",
                        'End Price': f"${end_price:,.2f}",
                        'Return (%)': f"{pct_change:.2f}%",
                        'Volatility (Annualized)': f"{volatility:.2%}",
                        'High': f"${data['High'].max():,.2f}",
                        'Low': f"${data['Low'].min():,.2f}"
                    })
            
            if metrics:
                metrics_df = pd.DataFrame(metrics)
                st.dataframe(
                    metrics_df,
                    column_config={
                        "Return (%)": st.column_config.ProgressColumn(
                            "Return (%)",
                            help="Percentage return over selected period",
                            format="%.2f%%",
                            min_value=min(metrics_df['Return (%)'].str.replace('%', '').astype(float)),
                            max_value=max(metrics_df['Return (%)'].str.replace('%', '').astype(float))
                        )
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.warning("No metrics available for the selected cryptocurrencies")

# Enhanced Analysis Tab with more technical indicators
def analysis_tab(selected_cryptos, time_period):
    st.header("📊 Technical Analysis")
    st.markdown("Advanced technical indicators and correlation analysis")
    
    if not selected_cryptos:
        st.warning("Please select at least one cryptocurrency from the sidebar.")
        return
    
    with st.spinner("Loading analysis data..."):
        crypto_data = get_multiple_cryptos(selected_cryptos, time_period)
    
    if not crypto_data:
        st.error("No data available for the selected cryptocurrencies.")
        return
    
    # Create tabs for different analysis types
    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["Returns Analysis", "Correlation Matrix", "Technical Indicators"])
    
    with analysis_tab1:
        # Calculate daily returns
        returns_data = {}
        for ticker, data in crypto_data.items():
            if not data.empty:
                returns_data[ticker] = data['Close'].pct_change().dropna()
        
        # Plot returns distribution
        if len(returns_data) > 0:
            st.subheader("Daily Returns Distribution")
            
            # Use columns for better layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                for ticker, returns in returns_data.items():
                    sns.kdeplot(returns, label=ticker.split("-")[0], ax=ax)
                ax.set_title("Density of Daily Returns")
                ax.set_xlabel("Daily Return")
                ax.set_ylabel("Density")
                ax.legend()
                st.pyplot(fig)
            
            with col2:
                # Display key statistics
                stats = []
                for ticker, returns in returns_data.items():
                    stats.append({
                        'Cryptocurrency': ticker.split("-")[0],
                        'Mean Return': f"{returns.mean():.4%}",
                        'Std Dev': f"{returns.std():.4%}",
                        'Skewness': f"{returns.skew():.2f}",
                        'Kurtosis': f"{returns.kurtosis():.2f}"
                    })
                
                if stats:
                    st.dataframe(
                        pd.DataFrame(stats),
                        hide_index=True,
                        use_container_width=True
                    )
    
    with analysis_tab2:
        # Correlation matrix
        if len(selected_cryptos) > 1:
            st.subheader("Cryptocurrency Correlations")
            
            # Create a DataFrame with closing prices
            close_prices = pd.DataFrame()
            for ticker, data in crypto_data.items():
                if not data.empty:
                    close_prices[ticker.split("-")[0]] = data['Close']
            
            if not close_prices.empty:
                # Calculate correlations
                corr_matrix = close_prices.corr()
                
                # Plot heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    corr_matrix, 
                    annot=True, 
                    cmap="coolwarm", 
                    vmin=-1, 
                    vmax=1, 
                    center=0,
                    ax=ax,
                    fmt=".2f",
                    linewidths=.5
                )
                ax.set_title("Cryptocurrency Price Correlations")
                st.pyplot(fig)
                
                # Interpretation
                st.markdown("""
                **Correlation Interpretation:**
                - 1.0: Perfect positive correlation
                - 0.8-1.0: Very strong positive correlation
                - 0.6-0.8: Strong positive correlation
                - 0.4-0.6: Moderate positive correlation
                - 0.2-0.4: Weak positive correlation
                - -0.2-0.2: Little to no correlation
                - Negative values indicate inverse relationships
                """)
    
    with analysis_tab3:
        # Technical indicators for the first selected crypto
        if len(selected_cryptos) > 0:
            primary_crypto = selected_cryptos[0]
            if primary_crypto in crypto_data and not crypto_data[primary_crypto].empty:
                data = crypto_data[primary_crypto]
                
                # Calculate indicators
                data['SMA_20'] = data['Close'].rolling(window=20).mean()
                data['SMA_50'] = data['Close'].rolling(window=50).mean()
                data['RSI'] = compute_rsi(data['Close'], 14)
                
                # Plot indicators
                fig = go.Figure()
                
                # Price and SMAs
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    name="Price",
                    line=dict(color='royalblue', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['SMA_20'],
                    name="20-Day SMA",
                    line=dict(color='orange', width=1)
                ))
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['SMA_50'],
                    name="50-Day SMA",
                    line=dict(color='green', width=1)
                ))
                
                fig.update_layout(
                    title=f"{primary_crypto.split('-')[0]} with Technical Indicators",
                    height=400,
                    template="plotly_white",
                    showlegend=True
                )
                
                # RSI subplot
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    name="RSI (14)",
                    line=dict(color='purple', width=2)
                ))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                fig_rsi.update_layout(
                    height=200,
                    template="plotly_white",
                    showlegend=True,
                    margin=dict(t=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.plotly_chart(fig_rsi, use_container_width=True)
                
                # Interpretation
                st.markdown("""
                **Technical Indicators Interpretation:**
                - **SMA (Simple Moving Average):** 
                    - 20-day SMA: Short-term trend indicator
                    - 50-day SMA: Medium-term trend indicator
                    - When price crosses above SMA, potential bullish signal
                    - When price crosses below SMA, potential bearish signal
                - **RSI (Relative Strength Index):**
                    - Above 70: Overbought condition (potential sell signal)
                    - Below 30: Oversold condition (potential buy signal)
                """)
            else:
                st.warning(f"No data available for {primary_crypto}")

# Helper function to compute RSI
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Enhanced Forecast Tab with confidence intervals
def forecast_tab(selected_cryptos, time_period, forecast_days):
    st.header("🔮 Price Forecast")
    st.markdown("Machine learning-based price predictions with confidence intervals")
    
    if not selected_cryptos:
        st.warning("Please select at least one cryptocurrency from the sidebar.")
        return
    
    with st.spinner("Preparing forecasts..."):
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
            future_dates, future_prices, mae, rmse = forecast_crypto_price(data, forecast_days)
            
            if future_dates is not None:
                # Plot historical and forecasted data with confidence interval
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    name="Historical",
                    mode='lines',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                # Forecasted data with confidence interval
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_prices,
                    name="Forecast",
                    mode='lines',
                    line=dict(color='#ff7f0e', width=2, dash='dash')
                ))
                
                # Confidence interval (using RMSE as proxy for uncertainty)
                fig.add_trace(go.Scatter(
                    x=list(future_dates) + list(future_dates)[::-1],
                    y=list(np.array(future_prices) + rmse) + list(np.array(future_prices) - rmse)[::-1],
                    fill='toself',
                    fillcolor='rgba(255, 127, 14, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name="Confidence Interval"
                ))
                
                fig.update_layout(
                    title=f"{ticker.split('-')[0]} Price Forecast with {forecast_days}-Day Horizon",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    hovermode="x unified",
                    height=500,
                    template="plotly_white",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show forecast metrics in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Last Price", 
                        f"${data['Close'].iloc[-1]:,.2f}",
                        delta=f"{(data['Close'].iloc[-1] - data['Close'].iloc[-2]):,.2f}",
                        delta_color="normal"
                    )
                with col2:
                    st.metric("Forecast MAE", f"${mae:,.2f}", "Mean Absolute Error")
                with col3:
                    st.metric("Forecast RMSE", f"${rmse:,.2f}", "Root Mean Squared Error")
                
                # Show forecast table with download option
                if st.checkbox(f"Show detailed forecast for {ticker.split('-')[0]}"):
                    forecast_df = pd.DataFrame({
                        "Date": future_dates,
                        "Forecasted Price": future_prices,
                        "Upper Bound": np.array(future_prices) + rmse,
                        "Lower Bound": np.array(future_prices) - rmse
                    }).set_index("Date")
                    
                    st.dataframe(
                        forecast_df.style.format({
                            "Forecasted Price": "${:,.2f}",
                            "Upper Bound": "${:,.2f}",
                            "Lower Bound": "${:,.2f}"
                        }),
                        use_container_width=True
                    )
                    
                    # Download button
                    csv = forecast_df.to_csv().encode('utf-8')
                    st.download_button(
                        label="Download Forecast Data",
                        data=csv,
                        file_name=f"{ticker.split('-')[0]}_forecast.csv",
                        mime="text/csv"
                    )
                
                # Forecast interpretation
                st.markdown("""
                **Forecast Interpretation:**
                - The forecast is generated using a Random Forest Regressor with technical indicators as features
                - The shaded area represents a confidence interval based on model error metrics
                - Longer forecast horizons generally have higher uncertainty
                - Consider multiple indicators before making investment decisions
                """)

# Enhanced News Tab with crypto resources
def news_tab():
    st.header("📰 Crypto Resources & News")
    st.markdown("""
    Explore the latest cryptocurrency news, analysis, and educational resources from trusted sources.
    """)
    
    # Create tabs for different resource categories
    tab1, tab2, tab3 = st.tabs(["News Websites", "Educational Resources", "Market Data"])
    
    with tab1:
        st.subheader("Top Cryptocurrency News Websites")
        
        # News resources
        news_resources = [
            {
                "name": "CoinDesk",
                "url": "https://www.coindesk.com/",
                "description": "Leading cryptocurrency news outlet covering blockchain, DeFi, and more.",
                "image": "https://www.coindesk.com/resizer/8Vj6e7p5X6Z6Q5X6Q5X6Q5X6Q5=/1200x628/cloudfront-us-east-1.images.arcpublishing.com/coindesk/ZJ5K5X5Z5FJ5X5Z5FJ5X5Z5FJ5X5.png"
            },
            {
                "name": "Cointelegraph",
                "url": "https://cointelegraph.com/",
                "description": "Independent news platform covering blockchain technology, crypto assets, and more.",
                "image": "https://cointelegraph.com/assets/images/logo.svg"
            },
            {
                "name": "Crypto News",
                "url": "https://cryptonews.com/",
                "description": "Comprehensive cryptocurrency news aggregator with price analysis.",
                "image": "https://cryptonews.com/wp-content/uploads/2021/04/cryptonews-logo.svg"
            },
            {
                "name": "The Block",
                "url": "https://www.theblock.co/",
                "description": "Research-driven news about cryptocurrency and blockchain.",
                "image": "https://www.theblock.co/wp-content/uploads/2021/06/the-block-logo.svg"
            },
            {
                "name": "Decrypt",
                "url": "https://decrypt.co/",
                "description": "Human-readable cryptocurrency and blockchain news.",
                "image": "https://decrypt.co/wp-content/uploads/2019/10/decrypt-logo-white.png"
            }
        ]
        
        # Display news resources as cards
        for resource in news_resources:
            with st.container():
                st.markdown(
                    f"""
                    <div class="resource-card">
                        <h3><a href="{resource['url']}" target="_blank">{resource['name']}</a></h3>
                        <p>{resource['description']}</p>
                        <a href="{resource['url']}" target="_blank">Visit Website →</a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    with tab2:
        st.subheader("Educational Resources")
        
        # Educational resources
        edu_resources = [
            {
                "name": "Binance Academy",
                "url": "https://academy.binance.com/",
                "description": "Free cryptocurrency and blockchain education for beginners to advanced users.",
                "image": "https://academy.binance.com/images/logo.svg"
            },
            {
                "name": "CoinGecko Learn",
                "url": "https://www.coingecko.com/learn",
                "description": "Educational content about cryptocurrency and blockchain basics.",
                "image": "https://static.coingecko.com/s/coingecko-logo-8903d34ce19ca4be1c81f0db30e924154750d208683fad7ae6f2ce06c76d0a56.png"
            },
            {
                "name": "Andreessen Horowitz Crypto",
                "url": "https://a16z.com/crypto/",
                "description": "Educational resources from one of the top crypto VC firms.",
                "image": "https://a16z.com/wp-content/uploads/2020/09/a16z-logo.png"
            },
            {
                "name": "Crypto Canon",
                "url": "https://a16z.com/2018/02/10/crypto-readings-resources/",
                "description": "Curated list of essential crypto readings from a16z.",
                "image": "https://a16z.com/wp-content/uploads/2020/09/a16z-logo.png"
            },
            {
                "name": "Ethereum.org Learn",
                "url": "https://ethereum.org/en/learn/",
                "description": "Official Ethereum learning portal with beginner to advanced content.",
                "image": "https://ethereum.org/static/28214aa68a0e6260f3d3a7fe31f5f853/31907/eth-diamond-purple.png"
            }
        ]
        
        # Display educational resources as cards
        for resource in edu_resources:
            with st.container():
                st.markdown(
                    f"""
                    <div class="resource-card">
                        <h3><a href="{resource['url']}" target="_blank">{resource['name']}</a></h3>
                        <p>{resource['description']}</p>
                        <a href="{resource['url']}" target="_blank">Visit Website →</a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    with tab3:
        st.subheader("Market Data & Analytics")
        
        # Market data resources
        market_resources = [
            {
                "name": "CoinMarketCap",
                "url": "https://coinmarketcap.com/",
                "description": "Cryptocurrency market capitalizations, prices, and charts.",
                "image": "https://coinmarketcap.com/static/img/logo_200x200.png"
            },
            {
                "name": "CoinGecko",
                "url": "https://www.coingecko.com/",
                "description": "Independent cryptocurrency data aggregator with comprehensive metrics.",
                "image": "https://static.coingecko.com/s/coingecko-logo-8903d34ce19ca4be1c81f0db30e924154750d208683fad7ae6f2ce06c76d0a56.png"
            },
            {
                "name": "Glassnode",
                "url": "https://glassnode.com/",
                "description": "On-chain cryptocurrency data and intelligence platform.",
                "image": "https://glassnode.com/static/img/logo.svg"
            },
            {
                "name": "Messari",
                "url": "https://messari.io/",
                "description": "Crypto market intelligence platform with research and data.",
                "image": "https://messari.io/assets/images/logo.svg"
            },
            {
                "name": "CryptoCompare",
                "url": "https://www.cryptocompare.com/",
                "description": "Cryptocurrency market data, indices, and research.",
                "image": "https://www.cryptocompare.com/media/20562/logo.png"
            }
        ]
        
        # Display market data resources as cards
        for resource in market_resources:
            with st.container():
                st.markdown(
                    f"""
                    <div class="resource-card">
                        <h3><a href="{resource['url']}" target="_blank">{resource['name']}</a></h3>
                        <p>{resource['description']}</p>
                        <a href="{resource['url']}" target="_blank">Visit Website →</a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# Main App with enhanced error handling
def main():
    try:
        # Sidebar
        selected_cryptos, time_period, forecast_days = sidebar()
        
        # Main content with loading animation
        with st.spinner("Loading dashboard..."):
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Price Analysis", 
                "📊 Technical Tools", 
                "🔮 Forecast", 
                "📰 News & Insights"
            ])
            
            with tab1:
                price_tab(selected_cryptos, time_period)
            
            with tab2:
                analysis_tab(selected_cryptos, time_period)
            
            with tab3:
                forecast_tab(selected_cryptos, time_period, forecast_days)
            
            with tab4:
                news_tab()
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; color: #666; font-size: 0.9em;">
                <p>CryptoVision Pro • Data provided by Yahoo Finance • Updated at {}</p>
                <p>Disclaimer: This is for informational purposes only and not financial advice.</p>
            </div>
            """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            unsafe_allow_html=True
        )
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please refresh the page and try again. If the problem persists, contact support.")

if __name__ == "__main__":
    main()
