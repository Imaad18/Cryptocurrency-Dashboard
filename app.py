# Setup & Configuration:

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import networkx as nx
import requests
import warnings
from faker import Faker

# Suppress warnings
warnings.filterwarnings('ignore')

# Set visual styles
plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')
pd.set_option('display.max_columns', None)

# Constants
BTC_COLOR = '#F7931A'  # Bitcoin orange
ETH_COLOR = '#627EEA'  # Ethereum blue
DARK_BG = '#0E1117'
PLOT_BG = '#1E1E1E'

# Initialize Faker for demo data
fake = Faker()

# Configure Streamlit
st.set_page_config(
    page_title="CryptoInsight Analytics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown("""
            <style>
            .main { background-color: #0E1117; }
            .st-bb { background-color: transparent; }
            .st-at { background-color: #0E1117; }
            footer { visibility: hidden; }
            </style>
        """, unsafe_allow_html=True)

local_css("style.css")


# Data Processing:

class CryptoDataPreprocessor:
    def __init__(self, data):
        self.data = data.copy()
        self.btc_data = None
        self.eth_data = None
        self.preprocessed = False

    def preprocess(self):
        """Clean and prepare transaction data."""
        with st.spinner('Preprocessing data...'):
            # Convert timestamp to datetime
            self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])

            # Extract time features
            self.data['Date'] = self.data['Timestamp'].dt.date
            self.data['Hour'] = self.data['Timestamp'].dt.hour
            self.data['Day_of_Week'] = self.data['Timestamp'].dt.day_name()
            self.data['Day_of_Week_Num'] = self.data['Timestamp'].dt.dayofweek
            self.data['Month'] = self.data['Timestamp'].dt.month_name()
            self.data['Year'] = self.data['Timestamp'].dt.year

            # Separate BTC and ETH
            self.btc_data = self.data[self.data['Currency'] == 'BTC']
            self.eth_data = self.data[self.data['Currency'] == 'ETH']

            # Calculate fees
            self.data['Fee_Percentage'] = (self.data['Transaction_Fee'] / self.data['Amount']) * 100

            # Ethereum gas calculations
            if 'Gas_Price_Gwei' in self.data.columns:
                self.data['Gas_Cost_ETH'] = self.data['Gas_Price_Gwei'] * 1e-9 * 21000

            # Transaction size categories
            self.data['Size_Category'] = pd.cut(
                self.data['Amount'],
                bins=[0, 0.1, 1, 10, 100, float('inf')],
                labels=['<0.1', '0.1-1', '1-10', '10-100', '>100']
            )

            self.preprocessed = True
        st.success('Data preprocessing completed!')
        self._summarize_data()

    def _summarize_data(self):
        """Display dataset summary."""
        with st.expander("📊 Dataset Summary", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Transactions", f"{len(self.data):,}")
                st.metric("BTC Transactions", f"{len(self.btc_data):,} ({len(self.btc_data)/len(self.data)*100:.1f}%)")
                
            with col2:
                st.metric("Date Range", f"{self.data['Timestamp'].min().date()} to {self.data['Timestamp'].max().date()}")
                st.metric("ETH Transactions", f"{len(self.eth_data):,} ({len(self.eth_data)/len(self.data)*100:.1f}%)")
                
            with col3:
                st.metric("Unique Senders", f"{self.data['Sender_Address'].nunique():,}")
                st.metric("Unique Receivers", f"{self.data['Receiver_Address'].nunique():,}")

            if 'Mining_Pool' in self.data.columns:
                st.metric("Mining Pools", f"{self.data['Mining_Pool'].nunique()}")

            # Transaction distributions
            st.subheader("Transaction Distribution")
            dist_col1, dist_col2 = st.columns(2)
            
            with dist_col1:
                st.write("**Transaction Status**")
                status_dist = self.data['Transaction_Status'].value_counts(normalize=True)
                fig = px.pie(status_dist, values=status_dist.values, names=status_dist.index,
                            color_discrete_sequence=[BTC_COLOR, ETH_COLOR])
                st.plotly_chart(fig, use_container_width=True)
                
            with dist_col2:
                st.write("**Transaction Type**")
                type_dist = self.data['Transaction_Type'].value_counts(normalize=True)
                fig = px.pie(type_dist, values=type_dist.values, names=type_dist.index,
                            color_discrete_sequence=[BTC_COLOR, ETH_COLOR])
                st.plotly_chart(fig, use_container_width=True)

            # Hourly patterns
            st.subheader("Hourly Transaction Patterns")
            hourly_data = self.data.groupby(['Hour', 'Currency']).size().reset_index(name='Count')
            fig = px.line(hourly_data, x='Hour', y='Count', color='Currency',
                         title='Transactions by Hour of Day',
                         color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR})
            st.plotly_chart(fig, use_container_width=True)

    def get_processed_data(self):
        """Return processed datasets."""
        if not self.preprocessed:
            self.preprocess()
        return self.data, self.btc_data, self.eth_data


# Real Time Price Data

def get_crypto_prices():
    """Fetch current crypto prices from CoinGecko API."""
    url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids=bitcoin,ethereum"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        btc_data = next(item for item in data if item["id"] == "bitcoin")
        eth_data = next(item for item in data if item["id"] == "ethereum")
        
        # Get historical data
        btc_history = requests.get(
            "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=90&interval=daily",
            timeout=10
        ).json()
        
        eth_history = requests.get(
            "https://api.coingecko.com/api/v3/coins/ethereum/market_chart?vs_currency=usd&days=90&interval=daily",
            timeout=10
        ).json()
        
        return {
            'BTC': {
                'price': btc_data['current_price'],
                'change_24h': btc_data['price_change_percentage_24h'],
                'market_cap': btc_data['market_cap'],
                'volume': btc_data['total_volume'],
                'history': btc_history
            },
            'ETH': {
                'price': eth_data['current_price'],
                'change_24h': eth_data['price_change_percentage_24h'],
                'market_cap': eth_data['market_cap'],
                'volume': eth_data['total_volume'],
                'history': eth_history
            }
        }
    except Exception as e:
        st.error(f"Error fetching crypto prices: {e}")
        return None

# Mining Pool Analysis

def analyze_mining_pools(data):
    """Analyze mining pool activity."""
    st.header("⛏️ Mining Pool Analysis")
    
    if 'Mining_Pool' not in data.columns:
        st.warning("No mining pool data available.")
        return None

    # Pool statistics
    pool_counts = data.groupby(['Mining_Pool', 'Currency']).agg(
        Transaction_Count=('Transaction_ID', 'count'),
        Total_Amount=('Amount', 'sum'),
        Avg_Fee=('Transaction_Fee', 'mean')
    ).reset_index()

    # Top pools
    top_n = st.slider("Select number of top pools", 5, 20, 10)
    top_pools = pool_counts.sort_values('Transaction_Count', ascending=False).head(top_n)

    # Visualizations
    st.subheader(f"Top {top_n} Mining Pools")
    fig = px.bar(top_pools, x='Mining_Pool', y='Transaction_Count', color='Currency',
                color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pool Efficiency")
    pool_daily = data.groupby(['Mining_Pool', 'Currency', 'Date']).size().reset_index(name='Daily_Tx')
    pool_efficiency = pool_daily.groupby(['Mining_Pool', 'Currency']).agg(
        Avg_Daily_Tx=('Daily_Tx', 'mean'),
        Std_Daily_Tx=('Daily_Tx', 'std')
    ).reset_index()

    fig = px.scatter(pool_efficiency, x='Avg_Daily_Tx', y='Std_Daily_Tx',
                    color='Currency', size='Avg_Daily_Tx',
                    hover_name='Mining_Pool')
    st.plotly_chart(fig, use_container_width=True)

    return pool_counts

# Main App Function

def main():
    st.title("CryptoInsight Analytics Dashboard")
    
    # File upload or demo data
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        try:
            raw_data = pd.read_csv(uploaded_file)
            preprocessor = CryptoDataPreprocessor(raw_data)
            data, btc_data, eth_data = preprocessor.get_processed_data()
            
            # Navigation tabs
            tabs = ["📈 Price Data", "📊 Data Overview", "⛏️ Mining Pools", 
                   "💸 Transaction Fees", "📨 Address Activity", "⏱️ Temporal Patterns"]
            selected_tab = st.sidebar.radio("Navigation", tabs)
            
            if selected_tab == "📈 Price Data":
                show_price_tab()
            elif selected_tab == "📊 Data Overview":
                preprocessor._summarize_data()
            elif selected_tab == "⛏️ Mining Pools":
                analyze_mining_pools(data)
            elif selected_tab == "💸 Transaction Fees":
                analyze_transaction_fees(data)
            elif selected_tab == "📨 Address Activity":
                analyze_address_activity(data)
            elif selected_tab == "⏱️ Temporal Patterns":
                detect_temporal_patterns(data)
                
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Upload a CSV file or use demo data.")
        if st.button("Load Demo Data"):
            demo_data = generate_demo_data()
            preprocessor = CryptoDataPreprocessor(demo_data)
            data, _, _ = preprocessor.get_processed_data()
            st.success("Demo data loaded!")

if __name__ == "__main__":
    main()
