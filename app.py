import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from newsapi import NewsApiClient
import os
from dotenv import load_dotenv
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64

# Load environment variables
load_dotenv()

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

# Initialize NewsAPI
try:
    newsapi = NewsApiClient(api_key=os.getenv('NEWSAPI_KEY'))
except Exception as e:
    st.error(f"Error initializing NewsAPI: {str(e)}")

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
def get_crypto_news(query="cryptocurrency", language="en", page_size=10):
    try:
        news = newsapi.get_everything(
            q=query,
            language=language,
            page_size=page_size,
            sort_by="publishedAt"
        )
        return news.get('articles', [])
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
    
    # Create dates for future predictions
    last_date = data.index.max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    
    return future_dates, future_prices, mae

# Custom components
def card_component(title, value, change=None, icon="💰"):
    change_color = "green" if change and change >= 0 else "red" if change else "gray"
    change_text = f"<span style='color:{change_color};font-size:14px;'>{change:+.2f}%</span>" if change is not None else ""
    
    card = f"""
    <div class="card">
        <div class="card-icon">{icon}</div>
        <div class="card-content">
            <div class="card-title">{title}</div>
            <div class="card-value">{value}</div>
            <div class="card-change">{change_text}</div>
        </div>
    </div>
    """
    st.markdown(card, unsafe_allow_html=True)

def loading_spinner(text="Loading..."):
    with st.spinner(text):
        time.sleep(0.5)

# Sidebar
def sidebar():
    st.sidebar.title("Crypto Dashboard")
    st.sidebar.markdown("### Settings")
    
    # Default cryptos
    default_cryptos = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"]
    
    selected_cryptos = st.sidebar.multiselect(
        "Select Cryptocurrencies",
        options=default_cryptos,
        default=default_cryptos[:3]
    )
    
    time_period = st.sidebar.selectbox(
        "Time Period",
        options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=5
    )
    
    forecast_days = st.sidebar.slider(
        "Forecast Days",
        min_value=7,
        max_value=90,
        value=30
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    This dashboard provides analysis of cryptocurrency prices using data from Yahoo Finance.
    """)
    
    return selected_cryptos, time_period, forecast_days

# Price Tab
def price_tab(selected_cryptos, time_period):
    st.header("Cryptocurrency Prices")
    
    if not selected_cryptos:
        st.warning("Please select at least one cryptocurrency from the sidebar.")
        return
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Price Trends")
        
        # Get data for all selected cryptos
        crypto_data = get_multiple_cryptos(selected_cryptos, time_period)
        
        if not crypto_data:
            st.error("No data available for the selected cryptocurrencies.")
            return
        
        # Create a combined DataFrame for plotting
        combined_df = pd.DataFrame()
        for ticker, data in crypto_data.items():
            if not data.empty:
                temp_df = data[['Close']].copy()
                temp_df['Crypto'] = ticker
                combined_df = pd.concat([combined_df, temp_df])
        
        if combined_df.empty:
            st.error("No valid data to display.")
            return
        
        # Plot price trends
        fig = px.line(
            combined_df.reset_index(),
            x='Date',
            y='Close',
            color='Crypto',
            title=f'Price Trends ({time_period})',
            labels={'Close': 'Price (USD)', 'Date': 'Date'},
            height=500
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Current Stats")
        
        # Display cards for each crypto
        for ticker in selected_cryptos:
            if ticker in crypto_data and not crypto_data[ticker].empty:
                data = crypto_data[ticker]
                current_price = data['Close'].iloc[-1]
                prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                change_pct = ((current_price - prev_price) / prev_price) * 100
                
                card_component(
                    title=ticker,
                    value=f"${current_price:,.2f}",
                    change=change_pct,
                    icon="📈" if change_pct >= 0 else "📉"
                )
        
        st.markdown("---")
        st.markdown("### Market Summary")
        
        # Calculate 24h volume if available
        if all('Volume' in data.columns for data in crypto_data.values() if not data.empty):
            volumes = {
                ticker: data['Volume'].iloc[-1] 
                for ticker, data in crypto_data.items() 
                if not data.empty
            }
            total_volume = sum(volumes.values())
            
            for ticker, volume in volumes.items():
                st.metric(
                    label=f"{ticker} 24h Volume",
                    value=f"${volume:,.0f}",
                    delta=f"{(volume/total_volume)*100:.1f}% of total"
                )

# Analysis Tab
def analysis_tab(selected_cryptos, time_period):
    st.header("Cryptocurrency Analysis")
    
    if len(selected_cryptos) < 2:
        st.warning("Please select at least two cryptocurrencies for comparison.")
        return
    
    crypto_data = get_multiple_cryptos(selected_cryptos, time_period)
    
    if not crypto_data or all(data.empty for data in crypto_data.values()):
        st.error("No data available for analysis.")
        return
    
    # Calculate daily returns
    returns_data = {}
    for ticker, data in crypto_data.items():
        if not data.empty and 'Close' in data.columns:
            returns_data[ticker] = data['Close'].pct_change().dropna()
    
    if not returns_data:
        st.error("Could not calculate returns for any cryptocurrency.")
        return
    
    # Create DataFrame for returns
    returns_df = pd.DataFrame(returns_data)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Correlation", "Volatility", "Returns Distribution", "Cumulative Returns"])
    
    with tab1:
        st.markdown("### Correlation Matrix")
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)
        
        st.markdown("""
        **Interpretation:**
        - Values close to 1 indicate strong positive correlation
        - Values close to -1 indicate strong negative correlation
        - Values around 0 indicate little to no correlation
        """)
    
    with tab2:
        st.markdown("### Volatility Comparison")
        
        # Calculate rolling volatility (30-day std)
        volatility_df = returns_df.rolling(window=30).std() * np.sqrt(365)  # Annualized
        
        # Plot volatility
        fig = px.line(
            volatility_df.reset_index().melt(id_vars='Date', var_name='Crypto', value_name='Volatility'),
            x='Date',
            y='Volatility',
            color='Crypto',
            title='30-Day Rolling Volatility (Annualized)',
            labels={'Volatility': 'Volatility', 'Date': 'Date'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show current volatility rankings
        current_volatility = volatility_df.iloc[-1].sort_values(ascending=False)
        st.markdown("#### Current Volatility Ranking")
        st.dataframe(current_volatility.rename('Volatility').to_frame().style.format("{:.2%}"))
    
    with tab3:
        st.markdown("### Returns Distribution")
        
        # Plot distribution of returns
        fig = px.histogram(
            returns_df.melt(var_name='Crypto', value_name='Return'),
            x='Return',
            color='Crypto',
            marginal='box',
            nbins=100,
            title='Distribution of Daily Returns',
            barmode='overlay',
            opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show statistics
        st.markdown("#### Returns Statistics")
        stats_df = returns_df.describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        stats_df['Sharpe'] = (stats_df['mean'] * 365) / (stats_df['std'] * np.sqrt(365))
        st.dataframe(stats_df.style.format("{:.4%}"))
    
    with tab4:
        st.markdown("### Cumulative Returns")
        
        # Calculate cumulative returns
        cumulative_returns = (1 + returns_df).cumprod() - 1
        
        # Plot cumulative returns
        fig = px.line(
            cumulative_returns.reset_index().melt(id_vars='Date', var_name='Crypto', value_name='Cumulative Return'),
            x='Date',
            y='Cumulative Return',
            color='Crypto',
            title='Cumulative Returns Over Time',
            labels={'Cumulative Return': 'Cumulative Return', 'Date': 'Date'}
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show final cumulative returns
        st.markdown("#### Final Cumulative Returns")
        st.dataframe(cumulative_returns.iloc[-1].sort_values(ascending=False).to_frame('Cumulative Return').style.format("{:.2%}"))

# Forecast Tab
def forecast_tab(selected_cryptos, time_period, forecast_days):
    st.header("Cryptocurrency Price Forecast")
    
    if not selected_cryptos:
        st.warning("Please select at least one cryptocurrency from the sidebar.")
        return
    
    selected_crypto = st.selectbox("Select cryptocurrency to forecast", selected_cryptos)
    
    crypto_data = get_crypto_data(selected_crypto, time_period)
    
    if crypto_data.empty:
        st.error(f"No data available for {selected_crypto}.")
        return
    
    # Show historical data
    st.markdown("### Historical Price Data")
    st.line_chart(crypto_data['Close'])
    
    # Run forecast
    with st.spinner("Generating forecast..."):
        future_dates, future_prices, mae = forecast_crypto_price(crypto_data, forecast_days)
    
    if future_dates is None:
        st.error("Could not generate forecast due to insufficient data.")
        return
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': future_prices
    }).set_index('Date')
    
    # Combine historical and forecast data
    combined_df = pd.concat([
        crypto_data[['Close']].rename(columns={'Close': 'Actual'}),
        forecast_df
    ])
    
    # Plot forecast
    st.markdown(f"### {forecast_days}-Day Price Forecast")
    
    fig = go.Figure()
    
    # Add actual data
    fig.add_trace(go.Scatter(
        x=combined_df.index,
        y=combined_df['Actual'],
        name='Actual Price',
        line=dict(color='blue')
    ))
    
    # Add forecast data
    fig.add_trace(go.Scatter(
        x=combined_df.index,
        y=combined_df['Forecast'],
        name='Forecast',
        line=dict(color='orange', dash='dash')
    ))
    
    # Highlight forecast period
    forecast_start = forecast_df.index[0]
    fig.add_vline(
        x=forecast_start,
        line_width=2,
        line_dash="dash",
        line_color="red",
        annotation_text="Forecast Start"
    )
    
    fig.update_layout(
        title=f"{selected_crypto} Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show forecast summary
    st.markdown("### Forecast Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_price = crypto_data['Close'].iloc[-1]
        st.metric("Current Price", f"${current_price:,.2f}")
    
    with col2:
        forecast_end_price = forecast_df['Forecast'].iloc[-1]
        price_change = forecast_end_price - current_price
        pct_change = (price_change / current_price) * 100
        st.metric(
            f"{forecast_days}-Day Forecast",
            f"${forecast_end_price:,.2f}",
            delta=f"{pct_change:.2f}%"
        )
    
    with col3:
        st.metric("Model Error (MAE)", f"${mae:,.2f}")
    
    # Show forecast details
    st.markdown("#### Forecast Details")
    st.dataframe(forecast_df.style.format({"Forecast": "${:,.2f}"}))
    
    # Model interpretation
    st.markdown("""
    **Interpretation Notes:**
    - Forecast is generated using a Random Forest Regressor model
    - MAE (Mean Absolute Error) represents the average error of the model on test data
    - Past performance is not indicative of future results
    - Cryptocurrency markets are highly volatile and unpredictable
    """)

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
                if article['urlToImage']:
                    st.image(article['urlToImage'], width=200)
                else:
                    st.image("https://via.placeholder.com/200x150?text=No+Image", width=200)
            
            with col2:
                st.markdown(f"### [{article['title']}]({article['url']})")
                st.caption(f"Published at: {article['publishedAt'][:10]} | Source: {article['source']['name']}")
                st.write(article['description'])
                
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
