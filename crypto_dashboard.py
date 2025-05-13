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
