import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from crypto_utils import get_crypto_data, popular_cryptos, calculate_indicators

def show_price_analysis_page():
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

# Import the display_metric function
from crypto_utils import display_metric
