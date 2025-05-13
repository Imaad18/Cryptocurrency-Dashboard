# Enhanced CryptoInsight Analytics Dashboard
# - Complete implementation with all missing functions
# - Improved error handling and API rate limiting
# - Enhanced visualizations and better defaults
# - Fully functional demo data generation
# - Performance optimizations and structural improvements

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
import time
import random

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
ACCENT_COLOR = '#FF4B4B'
SUCCESS_COLOR = '#0ECB81'
WARNING_COLOR = '#F0B90B'

# Configure Streamlit
st.set_page_config(
    page_title="CryptoInsight Analytics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def local_css(file_name=None):
    """Apply custom CSS styling to the app."""
    try:
        if file_name:
            with open(file_name) as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        st.markdown("""
            <style>
            .main { background-color: #0E1117; }
            .st-bb { background-color: transparent; }
            .st-at { background-color: #0E1117; }
            footer { visibility: hidden; }
            .css-1v3fvcr { padding-top: 0; }
            .stTabs [data-baseweb="tab-list"] { gap: 2px; }
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                white-space: pre-wrap;
                background-color: #262730;
                border-radius: 4px 4px 0px 0px;
                gap: 1px;
                padding-top: 10px;
                padding-bottom: 10px;
            }
            .stTabs [aria-selected="true"] {
                background-color: #1E1E1E;
                border-bottom: 2px solid #FF4B4B;
            }
            .metric-card {
                background-color: #1E1E1E;
                border: 1px solid #333;
                padding: 15px;
                border-radius: 5px;
                margin: 10px 0;
            }
            .crypto-price-up {
                color: #0ECB81 !important;
                font-weight: bold;
            }
            .crypto-price-down {
                color: #FF4B4B !important;
                font-weight: bold;
            }
            .stAlert {
                background-color: #1E1E1E;
                border: 1px solid #333;
            }
            </style>
        """, unsafe_allow_html=True)

# Demo Data Generation
def generate_demo_data(n_transactions=1000):
    """Generate synthetic cryptocurrency transaction data."""
    np.random.seed(42)
    random.seed(42)
    
    # Generate timestamps (last 90 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    timestamps = [start_date + timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds()))) 
                  for _ in range(n_transactions)]
    
    # Generate addresses
    def generate_address():
        return '0x' + ''.join(random.choices('0123456789abcdef', k=40))
    
    addresses = [generate_address() for _ in range(n_transactions * 2)]
    senders = random.choices(addresses, k=n_transactions)
    receivers = random.choices(addresses, k=n_transactions)
    
    # Ensure sender != receiver
    for i in range(n_transactions):
        while senders[i] == receivers[i]:
            receivers[i] = generate_address()
    
    # Generate transaction data
    currencies = random.choices(['BTC', 'ETH'], weights=[0.6, 0.4], k=n_transactions)
    amounts = np.random.lognormal(mean=0, sigma=2, size=n_transactions)
    amounts = np.clip(amounts, 0.0001, 1000)  # Limit range for realism
    
    # Transaction fees (dependent on currency)
    fees = []
    gas_prices = []
    for i, currency in enumerate(currencies):
        if currency == 'BTC':
            fee = amounts[i] * random.uniform(0.0001, 0.005)  # 0.01% to 0.5% of amount
            gas_price = 0
        else:  # ETH
            fee = amounts[i] * random.uniform(0.001, 0.01)  # 0.1% to 1% of amount
            gas_price = random.uniform(20, 200)  # Gas price in Gwei
        fees.append(fee)
        gas_prices.append(gas_price)
    
    # Transaction status and type
    statuses = random.choices(['Confirmed', 'Pending', 'Failed'], weights=[0.9, 0.08, 0.02], k=n_transactions)
    types = random.choices(['Transfer', 'Contract', 'Swap'], weights=[0.7, 0.2, 0.1], k=n_transactions)
    
    # Mining pools
    mining_pools = random.choices(['Antpool', 'F2Pool', 'Poolin', 'Binance Pool', 'ViaBTC',
                                  'SlushPool', 'BTC.com', 'Foundry USA', 'SBI Crypto', 'Luxor'],
                                 k=n_transactions)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Timestamp': timestamps,
        'Currency': currencies,
        'Amount': amounts,
        'Transaction_Fee': fees,
        'Sender_Address': senders,
        'Receiver_Address': receivers,
        'Transaction_Status': statuses,
        'Transaction_Type': types,
        'Transaction_ID': ['tx_' + ''.join(random.choices('0123456789abcdef', k=64)) for _ in range(n_transactions)],
        'Gas_Price_Gwei': gas_prices,
        'Mining_Pool': mining_pools
    })
    
    # Add Transaction_Hash
    data['Transaction_Hash'] = data['Transaction_ID']
    
    return data

# Data Processing:
class CryptoDataPreprocessor:
    """Class to preprocess and analyze cryptocurrency transaction data."""
    
    def __init__(self, data):
        """Initialize with raw transaction data."""
        self.data = data.copy()
        self.btc_data = None
        self.eth_data = None
        self.preprocessed = False
        
        # Check for required columns
        required_cols = ['Timestamp', 'Currency', 'Amount', 'Transaction_Fee', 
                         'Sender_Address', 'Receiver_Address', 'Transaction_Status',
                         'Transaction_Type', 'Transaction_ID']
        
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            st.stop()

    def preprocess(self):
        """Clean and prepare transaction data."""
        with st.spinner('Preprocessing data...'):
            try:
                # Convert timestamp to datetime
                self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])
                
                # Handle any missing values
                for col in ['Amount', 'Transaction_Fee']:
                    if self.data[col].isnull().any():
                        self.data[col] = self.data[col].fillna(self.data[col].median())
                        
                for col in ['Transaction_Status', 'Transaction_Type']:
                    if self.data[col].isnull().any():
                        self.data[col] = self.data[col].fillna('Unknown')

                # Extract time features
                self.data['Date'] = self.data['Timestamp'].dt.date
                self.data['Hour'] = self.data['Timestamp'].dt.hour
                self.data['Day_of_Week'] = self.data['Timestamp'].dt.day_name()
                self.data['Day_of_Week_Num'] = self.data['Timestamp'].dt.dayofweek
                self.data['Month'] = self.data['Timestamp'].dt.month_name()
                self.data['Year'] = self.data['Timestamp'].dt.year
                self.data['Week'] = self.data['Timestamp'].dt.isocalendar().week

                # Separate BTC and ETH
                self.btc_data = self.data[self.data['Currency'] == 'BTC']
                self.eth_data = self.data[self.data['Currency'] == 'ETH']

                # Calculate fees
                self.data['Fee_Percentage'] = (self.data['Transaction_Fee'] / self.data['Amount']) * 100
                
                # Handle infinite values that might result from division by zero
                self.data['Fee_Percentage'] = self.data['Fee_Percentage'].replace([np.inf, -np.inf], np.nan)
                self.data['Fee_Percentage'] = self.data['Fee_Percentage'].fillna(0)

                # Ethereum gas calculations (if available)
                if 'Gas_Price_Gwei' in self.data.columns:
                    # Standard Ethereum transaction uses 21000 gas
                    self.data['Gas_Cost_ETH'] = self.data['Gas_Price_Gwei'] * 1e-9 * 21000
                    self.data.loc[self.data['Currency'] != 'ETH', 'Gas_Cost_ETH'] = 0

                # Transaction size categories
                self.data['Size_Category'] = pd.cut(
                    self.data['Amount'],
                    bins=[0, 0.1, 1, 10, 100, float('inf')],
                    labels=['<0.1', '0.1-1', '1-10', '10-100', '>100']
                )
                
                # Add transaction_hash if it doesn't exist (for network analysis)
                if 'Transaction_Hash' not in self.data.columns:
                    self.data['Transaction_Hash'] = self.data['Transaction_ID']
                
                self.preprocessed = True
                
            except Exception as e:
                st.error(f"Error during preprocessing: {str(e)}")
                st.stop()
                
        st.success('Data preprocessing completed!')
        return self.data

    def _summarize_data(self):
        """Display dataset summary."""
        if not self.preprocessed:
            self.preprocess()
            
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
                            color_discrete_sequence=px.colors.qualitative.Set3,
                            hole=0.4)
                fig.update_layout(
                    plot_bgcolor=PLOT_BG,
                    paper_bgcolor=DARK_BG,
                    font=dict(color="white"),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2)
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with dist_col2:
                st.write("**Transaction Type**")
                type_dist = self.data['Transaction_Type'].value_counts(normalize=True)
                fig = px.pie(type_dist, values=type_dist.values, names=type_dist.index,
                            color_discrete_sequence=px.colors.qualitative.Pastel,
                            hole=0.4)
                fig.update_layout(
                    plot_bgcolor=PLOT_BG,
                    paper_bgcolor=DARK_BG,
                    font=dict(color="white"),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2)
                )
                st.plotly_chart(fig, use_container_width=True)

            # Hourly patterns
            st.subheader("Hourly Transaction Patterns")
            hourly_data = self.data.groupby(['Hour', 'Currency']).size().reset_index(name='Count')
            fig = px.line(hourly_data, x='Hour', y='Count', color='Currency',
                         title='Transactions by Hour of Day',
                         color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR})
            fig.update_layout(
                xaxis=dict(tickmode='linear', tick0=0, dtick=2),
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=DARK_BG,
                font=dict(color="white")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Transaction amounts by day of week
            st.subheader("Transaction Volume by Day of Week")
            daily_volume = self.data.groupby(['Day_of_Week_Num', 'Day_of_Week', 'Currency'])['Amount'].sum().reset_index()
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_volume['Day_of_Week'] = pd.Categorical(daily_volume['Day_of_Week'], categories=days_order, ordered=True)
            daily_volume = daily_volume.sort_values('Day_of_Week')
            
            fig = px.bar(daily_volume, x='Day_of_Week', y='Amount', color='Currency',
                       barmode='group', color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR})
            fig.update_layout(
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=DARK_BG,
                font=dict(color="white")
            )
            st.plotly_chart(fig, use_container_width=True)

    def get_processed_data(self):
        """Return processed datasets."""
        if not self.preprocessed:
            self.preprocess()
        return self.data, self.btc_data, self.eth_data

# Real Time Price Data
def get_crypto_prices():
    """Fetch current crypto prices from CoinGecko API with rate limiting."""
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "ids": "bitcoin,ethereum",
        "order": "market_cap_desc",
        "per_page": 100,
        "page": 1,
        "sparkline": False,
        "price_change_percentage": "24h,7d,30d"
    }
    
    try:
        # First attempt
        response = requests.get(url, params=params, timeout=10)
        
        # Check for rate limiting
        if response.status_code == 429:
            st.warning("CoinGecko API rate limit reached. Waiting before retry...")
            time.sleep(2)  # Wait before retry
            response = requests.get(url, params=params, timeout=10)
        
        # Check for other errors
        if response.status_code != 200:
            st.error(f"API Error: Status code {response.status_code}")
            return None
            
        data = response.json()
        btc_data = next((item for item in data if item["id"] == "bitcoin"), None)
        eth_data = next((item for item in data if item["id"] == "ethereum"), None)
        
        if not btc_data or not eth_data:
            st.error("Could not find Bitcoin or Ethereum data in the API response")
            return None
        
        # Get historical data with careful handling
        try:
            btc_history_url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
            btc_history_params = {
                "vs_currency": "usd",
                "days": 90,
                "interval": "daily"
            }
            btc_history_response = requests.get(btc_history_url, params=btc_history_params, timeout=10)
            
            if btc_history_response.status_code == 429:
                time.sleep(2)  # Wait before retry
                btc_history_response = requests.get(btc_history_url, params=btc_history_params, timeout=10)
                
            btc_history = btc_history_response.json() if btc_history_response.status_code == 200 else {"prices": []}
            
            # Small delay between requests to avoid rate limiting
            time.sleep(0.5)
            
            eth_history_url = "https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
            eth_history_params = {
                "vs_currency": "usd",
                "days": 90,
                "interval": "daily"
            }
            eth_history_response = requests.get(eth_history_url, params=eth_history_params, timeout=10)
            
            if eth_history_response.status_code == 429:
                time.sleep(2)  # Wait before retry
                eth_history_response = requests.get(eth_history_url, params=eth_history_params, timeout=10)
                
            eth_history = eth_history_response.json() if eth_history_response.status_code == 200 else {"prices": []}
            
        except Exception as e:
            st.warning(f"Could not fetch historical data: {e}")
            btc_history = {"prices": []}
            eth_history = {"prices": []}
        
        return {
            'BTC': {
                'price': btc_data['current_price'],
                'change_24h': btc_data.get('price_change_percentage_24h', 0),
                'change_7d': btc_data.get('price_change_percentage_7d_in_currency', 0),
                'market_cap': btc_data['market_cap'],
                'volume': btc_data['total_volume'],
                'history': btc_history,
                'image': btc_data.get('image', '')
            },
            'ETH': {
                'price': eth_data['current_price'],
                'change_24h': eth_data.get('price_change_percentage_24h', 0),
                'change_7d': eth_data.get('price_change_percentage_7d_in_currency', 0),
                'market_cap': eth_data['market_cap'],
                'volume': eth_data['total_volume'],
                'history': eth_history,
                'image': eth_data.get('image', '')
            }
        }
    except Exception as e:
        st.error(f"Error fetching crypto prices: {str(e)}")
        
        # Return mock data as fallback
        return {
            'BTC': {
                'price': 38456.78,
                'change_24h': 2.34,
                'change_7d': -1.23,
                'market_cap': 743569874123,
                'volume': 23456789123,
                'history': {"prices": [[time.time() * 1000, 38456.78] for _ in range(90)]},
                'image': ''
            },
            'ETH': {
                'price': 2345.67,
                'change_24h': -1.45,
                'change_7d': 3.56,
                'market_cap': 234567891234,
                'volume': 12345678912,
                'history': {"prices": [[time.time() * 1000, 2345.67] for _ in range(90)]},
                'image': ''
            }
        }

def show_price_tab():
    """Display cryptocurrency price information and charts."""
    st.header("📈 Cryptocurrency Market Data")
    
    # Fetch current prices
    with st.spinner("Fetching latest cryptocurrency data..."):
        crypto_data = get_crypto_prices()
    
    if not crypto_data:
        st.error("Unable to fetch cryptocurrency data. Please try again later.")
        return
    
    # Price cards in row
    col1, col2 = st.columns(2)
    
    with col1:
        btc_change_24h = crypto_data['BTC']['change_24h']
        btc_change_class = "crypto-price-up" if btc_change_24h >= 0 else "crypto-price-down"
        btc_change_icon = "📈" if btc_change_24h >= 0 else "📉"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Bitcoin (BTC)</h3>
            <h2>${crypto_data['BTC']['price']:,.2f} <span class="{btc_change_class}">{btc_change_icon} {btc_change_24h:.2f}%</span></h2>
            <p>Market Cap: ${crypto_data['BTC']['market_cap']:,.0f}</p>
            <p>24h Volume: ${crypto_data['BTC']['volume']:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        eth_change_24h = crypto_data['ETH']['change_24h']
        eth_change_class = "crypto-price-up" if eth_change_24h >= 0 else "crypto-price-down"
        eth_change_icon = "📈" if eth_change_24h >= 0 else "📉"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Ethereum (ETH)</h3>
            <h2>${crypto_data['ETH']['price']:,.2f} <span class="{eth_change_class}">{eth_change_icon} {eth_change_24h:.2f}%</span></h2>
            <p>Market Cap: ${crypto_data['ETH']['market_cap']:,.0f}</p>
            <p>24h Volume: ${crypto_data['ETH']['volume']:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Historical price charts
    st.subheader("90-Day Price History")
    
    # Process historical data
    if crypto_data['BTC']['history'].get('prices') and crypto_data['ETH']['history'].get('prices'):
        btc_history = pd.DataFrame(
            crypto_data['BTC']['history']['prices'], 
            columns=['timestamp', 'price']
        )
        btc_history['timestamp'] = pd.to_datetime(btc_history['timestamp'], unit='ms')
        btc_history['currency'] = 'BTC'
        
        eth_history = pd.DataFrame(
            crypto_data['ETH']['history']['prices'], 
            columns=['timestamp', 'price']
        )
        eth_history['timestamp'] = pd.to_datetime(eth_history['timestamp'], unit='ms')
        eth_history['currency'] = 'ETH'
        
        # Create combined chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # BTC line
        fig.add_trace(
            go.Scatter(
                x=btc_history['timestamp'], 
                y=btc_history['price'],
                name="Bitcoin",
                line=dict(color=BTC_COLOR, width=2),
            ),
            secondary_y=False
        )
        
        # ETH line (secondary y-axis)
        fig.add_trace(
            go.Scatter(
                x=eth_history['timestamp'], 
                y=eth_history['price'],
                name="Ethereum",
                line=dict(color=ETH_COLOR, width=2),
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title_text="BTC and ETH Price History (90 Days)",
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=DARK_BG,
            font=dict(color="white"),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Update axes
        fig.update_xaxes(
            title_text="Date",
            gridcolor='rgba(255, 255, 255, 0.1)',
            tickformat="%m/%d"
        )
        
        fig.update_yaxes(
            title_text="BTC Price (USD)",
            secondary_y=False,
            gridcolor='rgba(255, 255, 255, 0.1)',
            tickprefix="$"
        )
        
        fig.update_yaxes(
            title_text="ETH Price (USD)",
            secondary_y=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            tickprefix="$"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional price statistics
        st.subheader("Price Statistics")
        
        stats_col1, stats_col2 = st.columns(2)
        
        with stats_col1:
            st.markdown("### Bitcoin")
            btc_stats = {
                "Current Price": f"${crypto_data['BTC']['price']:,.2f}",
                "24h Change": f"{crypto_data['BTC']['change_24h']:+.2f}%",
                "7d Change": f"{crypto_data['BTC']['change_7d']:+.2f}%",
                "90d High": f"${btc_history['price'].max():,.2f}",
                "90d Low": f"${btc_history['price'].min():,.2f}",
                "90d Avg": f"${btc_history['price'].mean():,.2f}",
                "90d Volatility": f"{btc_history['price'].std() / btc_history['price'].mean() * 100:.2f}%"
            }
            
            for key, value in btc_stats.items():
                st.write(f"**{key}:** {value}")
        
        with stats_col2:
            st.markdown("### Ethereum")
            eth_stats = {
                "Current Price": f"${crypto_data['ETH']['price']:,.2f}",
                "24h Change": f"{crypto_data['ETH']['change_24h']:+.2f}%",
                "7d Change": f"{crypto_data['ETH']['change_7d']:+.2f}%",
                "90d High": f"${eth_history['price'].max():,.2f}",
                "90d Low": f"${eth_history['price'].min():,.2f}",
                "90d Avg": f"${eth_history['price'].mean():,.2f}",
                "90d Volatility": f"{eth_history['price'].std() / eth_history['price'].mean() * 100:.2f}%"
            }
            
            for key, value in eth_stats.items():
                st.write(f"**{key}:** {value}")
                
        # Price correlation
        if len(btc_history) == len(eth_history):
            correlation = btc_history['price'].corr(eth_history['price'])
            st.info(f"**BTC-ETH Price Correlation:** {correlation:.4f}")
            
    else:
        st.warning("Historical price data is not available")

# Mining Pool Analysis
def analyze_mining_pools(data):
    """Analyze mining pool activity."""
    st.header("⛏️ Mining Pool Analysis")
    
    if 'Mining_Pool' not in data.columns:
        st.warning("No mining pool data available. Using demo mining pool data.")
        # Generate synthetic mining pool data for demo
        random_pools = ['Antpool', 'F2Pool', 'Poolin', 'Binance Pool', 'ViaBTC', 
                        'SlushPool', 'BTC.com', 'Foundry USA', 'SBI Crypto', 'Luxor']
        data['Mining_Pool'] = np.random.choice(random_pools, size=len(data))
    
    # Pool statistics
    pool_counts = data.groupby(['Mining_Pool', 'Currency']).agg(
        Transaction_Count=('Transaction_ID', 'count'),
        Total_Amount=('Amount', 'sum'),
        Avg_Fee=('Transaction_Fee', 'mean')
    ).reset_index()

    # Top pools
    metric_options = {
        'Transaction_Count': 'Number of Transactions',
        'Total_Amount': 'Total Transaction Volume',
        'Avg_Fee': 'Average Fee'
    }
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        top_n = st.slider("Select number of top pools", 5, 20, 10)
        sort_metric = st.selectbox("Sort by", options=list(metric_options.keys()), 
                                  format_func=lambda x: metric_options[x])
    
    # Top pools chart
    top_pools = pool_counts.sort_values(sort_metric, ascending=False).head(top_n)
    
    with col2:
        st.subheader(f"Top {top_n} Mining Pools by {metric_options[sort_metric]}")
        
        fig = px.bar(top_pools, x='Mining_Pool', y=sort_metric, color='Currency',
                    color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR},
                    title=f"Top {top_n} Mining Pools by {metric_options[sort_metric]}")
        
        fig.update_layout(
            xaxis_title="Mining Pool",
            yaxis_title=metric_options[sort_metric],
            legend_title="Currency",
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=DARK_BG,
            font=dict(color="white")
        )
        
        # Rotate x-axis labels if there are many pools
        if len(top_pools) > 5:
            fig.update_xaxes(tickangle=45)
            
        st.plotly_chart(fig, use_container_width=True)

    # Pool efficiency analysis
    st.subheader("Mining Pool Efficiency Analysis")
    
    # Calculate daily metrics
    pool_daily = data.groupby(['Mining_Pool', 'Currency', pd.Grouper(key='Timestamp', freq='D')]).agg(
        Daily_Tx=('Transaction_ID', 'count'),
        Daily_Amount=('Amount', 'sum')
    ).reset_index()
    
    # Calculate efficiency metrics
    pool_efficiency = pool_daily.groupby(['Mining_Pool', 'Currency']).agg(
        Avg_Daily_Tx=('Daily_Tx', 'mean'),
        Std_Daily_Tx=('Daily_Tx', 'std'),
        Cv_Daily_Tx=('Daily_Tx', lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
        Avg_Daily_Amount=('Daily_Amount', 'mean'),
        Active_Days=('Daily_Tx', 'count')
    ).reset_index()
    
    # Fill NaN values from std calc when only one day of data
    pool_efficiency.fill(0, inplace=True)
    
    # Efficiency scatter plot
    eff_col1, eff_col2 = st.columns([3, 1])
    
    with eff_col1:
        size_norm = (pool_efficiency['Avg_Daily_Tx'] - pool_efficiency['Avg_Daily_Tx'].min()) / \
                    (pool_efficiency['Avg_Daily_Tx'].max() - pool_efficiency['Avg_Daily_Tx'].min() + 0.1) * 25 + 5
                    
        fig = px.scatter(pool_efficiency, 
                         x='Avg_Daily_Tx', 
                         y='Cv_Daily_Tx',
                         size=size_norm,
                         color='Currency',
                         hover_name='Mining_Pool',
                         hover_data=['Active_Days', 'Avg_Daily_Amount'],
                         color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR},
                         title="Mining Pool Consistency (Lower CV = More Consistent)"
                        )
        
        fig.update_layout(
            xaxis_title="Average Daily Transactions",
            yaxis_title="Coefficient of Variation (Std/Mean)",
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=DARK_BG,
            font=dict(color="white")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    with eff_col2:
        st.markdown("""
        ### Interpreting the Chart
        
        **X-axis**: Higher values mean more transactions processed daily
        
        **Y-axis**: Lower values mean more consistent performance
        
        **Size**: Larger bubbles = more daily transactions
        
        **Best performers**: Upper right quadrant pools process high volumes consistently
        """)
    
    # Mining pool market share over time
    st.subheader("Mining Pool Market Share Over Time")
    
    # Prepare time series data
    pool_share = pool_daily.copy()
    
    # For each day and currency, calculate the share of each pool
    pool_share_btc = pool_share[pool_share['Currency'] == 'BTC'].copy()
    pool_share_eth = pool_share[pool_share['Currency'] == 'ETH'].copy()
    
    # Calculate daily totals
    btc_daily_totals = pool_share_btc.groupby('Timestamp')['Daily_Tx'].sum().reset_index()
    btc_daily_totals.rename(columns={'Daily_Tx': 'Total_Daily_Tx'}, inplace=True)
    
    eth_daily_totals = pool_share_eth.groupby('Timestamp')['Daily_Tx'].sum().reset_index()
    eth_daily_totals.rename(columns={'Daily_Tx': 'Total_Daily_Tx'}, inplace=True)
    
    # Merge back to get percentages
    pool_share_btc = pool_share_btc.merge(btc_daily_totals, on='Timestamp')
    pool_share_btc['Market_Share'] = (pool_share_btc['Daily_Tx'] / pool_share_btc['Total_Daily_Tx']) * 100
    
    pool_share_eth = pool_share_eth.merge(eth_daily_totals, on='Timestamp')
    pool_share_eth['Market_Share'] = (pool_share_eth['Daily_Tx'] / pool_share_eth['Total_Daily_Tx']) * 100
    
    # Combine data back
    pool_share = pd.concat([pool_share_btc, pool_share_eth])
    
    # Create market share charts
    currency_tab1, currency_tab2 = st.tabs(["Bitcoin", "Ethereum"])
    
    with currency_tab1:
        btc_share_data = pool_share[pool_share['Currency'] == 'BTC']
        if not btc_share_data.empty:
            fig = px.area(btc_share_data, x='Timestamp', y='Market_Share', color='Mining_Pool',
                         title="Bitcoin Mining Pool Market Share Over Time")
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Market Share (%)",
                yaxis=dict(range=[0, 100]),
                legend_title="Mining Pool",
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=DARK_BG,
                font=dict(color="white")
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No Bitcoin mining pool data available for the selected time period")
    
    with currency_tab2:
        eth_share_data = pool_share[pool_share['Currency'] == 'ETH']
        if not eth_share_data.empty:
            fig = px.area(eth_share_data, x='Timestamp', y='Market_Share', color='Mining_Pool',
                         title="Ethereum Mining Pool Market Share Over Time")
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Market Share (%)",
                yaxis=dict(range=[0, 100]),
                legend_title="Mining Pool",
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=DARK_BG,
                font=dict(color="white")
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No Ethereum mining pool data available for the selected time period")
    
    return pool_counts

# Transaction Fee Analysis
def analyze_transaction_fees(data):
    """Analyze transaction fees and patterns."""
    st.header("💸 Transaction Fee Analysis")
    
    # Ensure Fee_Percentage is calculated
    if 'Fee_Percentage' not in data.columns:
        data['Fee_Percentage'] = (data['Transaction_Fee'] / data['Amount']) * 100
    
    # Basic fee statistics
    fee_stats = data.groupby('Currency').agg(
        Avg_Fee=('Transaction_Fee', 'mean'),
        Min_Fee=('Transaction_Fee', 'min'),
        Max_Fee=('Transaction_Fee', 'max'),
        Median_Fee=('Transaction_Fee', 'median'),
        Avg_Fee_Pct=('Fee_Percentage', 'mean')
    ).reset_index()
    
    # Display statistics
    st.subheader("Fee Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        btc_stats = fee_stats[fee_stats['Currency'] == 'BTC'].iloc[0] if 'BTC' in fee_stats['Currency'].values else None
        
        if btc_stats is not None:
            st.markdown(f"""
            ### Bitcoin
            - **Average Fee:** {btc_stats['Avg_Fee']:.8f} BTC
            - **Median Fee:** {btc_stats['Median_Fee']:.8f} BTC
            - **Min Fee:** {btc_stats['Min_Fee']:.8f} BTC
            - **Max Fee:** {btc_stats['Max_Fee']:.8f} BTC
            - **Avg Fee %:** {btc_stats['Avg_Fee_Pct']:.4f}%
            """)
        else:
            st.info("No Bitcoin transaction data available")
            
    with col2:
        eth_stats = fee_stats[fee_stats['Currency'] == 'ETH'].iloc[0] if 'ETH' in fee_stats['Currency'].values else None
        
        if eth_stats is not None:
            st.markdown(f"""
            ### Ethereum
            - **Average Fee:** {eth_stats['Avg_Fee']:.8f} ETH
            - **Median Fee:** {eth_stats['Median_Fee']:.8f} ETH
            - **Min Fee:** {eth_stats['Min_Fee']:.8f} ETH
            - **Max Fee:** {eth_stats['Max_Fee']:.8f} ETH
            - **Avg Fee %:** {eth_stats['Avg_Fee_Pct']:.4f}%
            """)
        else:
            st.info("No Ethereum transaction data available")
    
    # Fee distribution
    st.subheader("Fee Distribution")
    
    # Set up tabs for different visualizations
    dist_tab1, dist_tab2, dist_tab3 = st.tabs(["Histogram", "Box Plot", "Scatter Plot"])
    
    with dist_tab1:
        # Use a log scale for better visualization
        fig = px.histogram(
            data, x='Transaction_Fee', color='Currency', 
            log_x=True, # Use log scale for fees
            marginal='box', # Add box plot on the margin
            color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR},
            title="Transaction Fee Distribution (Log Scale)",
            nbins=50
        )
        
        fig.update_layout(
            xaxis_title="Transaction Fee (Log Scale)",
            yaxis_title="Count",
            legend_title="Currency",
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=DARK_BG,
            font=dict(color="white")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with dist_tab2:
        # Box plot for fee percentage
        fig = px.box(
            data, y='Fee_Percentage', x='Currency',
            color='Currency',
            points='all', # Show all points
            color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR},
            title="Transaction Fee Percentage Distribution"
        )
        
        fig.update_layout(
            xaxis_title="Currency",
            yaxis_title="Fee Percentage (%)",
            yaxis=dict(type='log'),  # Log scale for better visualization
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=DARK_BG,
            font=dict(color="white")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    with dist_tab3:
        # Scatter plot of fee vs amount
        fig = px.scatter(
            data, x='Amount', y='Transaction_Fee', 
            color='Currency',
            opacity=0.7,
            log_x=True, log_y=True, # Log scales
            color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR},
            title="Transaction Fee vs Amount",
            hover_data=['Transaction_ID', 'Fee_Percentage']
        )
        
        fig.update_layout(
            xaxis_title="Transaction Amount (Log Scale)",
            yaxis_title="Transaction Fee (Log Scale)",
            legend_title="Currency",
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=DARK_BG,
            font=dict(color="white")
        )
        
        # Add trend lines
        fig.update_traces(marker=dict(size=5))
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Fee trends over time
    st.subheader("Fee Trends Over Time")
    
    # Aggregation options
    time_options = {
        'D': 'Daily',
        'W': 'Weekly',
        'M': 'Monthly'
    }
    
    # Allow user to select aggregation level
    time_agg = st.selectbox("Time Aggregation", options=list(time_options.keys()),
                          format_func=lambda x: time_options[x], index=1)
    
    # Calculate time trends
    fee_trends = data.groupby(['Currency', pd.Grouper(key='Timestamp', freq=time_agg)]).agg(
        Avg_Fee=('Transaction_Fee', 'mean'),
        Median_Fee=('Transaction_Fee', 'median'),
        Transaction_Count=('Transaction_ID', 'count')
    ).reset_index()
    
    # Create time trend chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # BTC Fee line
    btc_trend = fee_trends[fee_trends['Currency'] == 'BTC']
    if not btc_trend.empty:
        fig.add_trace(
            go.Scatter(
                x=btc_trend['Timestamp'],
                y=btc_trend['Avg_Fee'],
                name="BTC Avg Fee",
                line=dict(color=BTC_COLOR, width=2)
            ),
            secondary_y=False
        )
    
    # ETH Fee line
    eth_trend = fee_trends[fee_trends['Currency'] == 'ETH']
    if not eth_trend.empty:
        fig.add_trace(
            go.Scatter(
                x=eth_trend['Timestamp'],
                y=eth_trend['Avg_Fee'],
                name="ETH Avg Fee",
                line=dict(color=ETH_COLOR, width=2)
            ),
            secondary_y=True
        )
    
    # Transaction volume as bars
    btc_volume = fee_trends[fee_trends['Currency'] == 'BTC']
    if not btc_volume.empty:
        fig.add_trace(
            go.Bar(
                x=btc_volume['Timestamp'],
                y=btc_volume['Transaction_Count'],
                name="BTC Tx Count",
                marker_color=BTC_COLOR,
                opacity=0.3
            ),
            secondary_y=False
        )
    
    eth_volume = fee_trends[fee_trends['Currency'] == 'ETH']
    if not eth_volume.empty:
        fig.add_trace(
            go.Bar(
                x=eth_volume['Timestamp'],
                y=eth_volume['Transaction_Count'],
                name="ETH Tx Count",
                marker_color=ETH_COLOR,
                opacity=0.3
            ),
            secondary_y=True
        )
    
    # Update layout
    fig.update_layout(
        title_text=f"Average Fee and Transaction Count ({time_options[time_agg]})",
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    # Update axes
    fig.update_xaxes(
        title_text="Date",
        gridcolor='rgba(255, 255, 255, 0.1)'
    )
    
    fig.update_yaxes(
        title_text="BTC Fee / Transaction Count",
        secondary_y=False,
        gridcolor='rgba(255, 255, 255, 0.1)'
    )
    
    fig.update_yaxes(
        title_text="ETH Fee / Transaction Count",
        secondary_y=True,
        gridcolor='rgba(255, 255, 255, 0.1)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Gas price analysis for Ethereum (if available)
    if 'Gas_Price_Gwei' in data.columns:
        st.subheader("Ethereum Gas Price Analysis")
        
        eth_data = data[data['Currency'] == 'ETH']
        
        if not eth_data.empty:
            # Gas price over time
            gas_trends = eth_data.groupby(pd.Grouper(key='Timestamp', freq=time_agg)).agg(
                Avg_Gas_Price=('Gas_Price_Gwei', 'mean'),
                Median_Gas_Price=('Gas_Price_Gwei', 'median'),
                Min_Gas_Price=('Gas_Price_Gwei', 'min'),
                Max_Gas_Price=('Gas_Price_Gwei', 'max'),
                Transaction_Count=('Transaction_ID', 'count')
            ).reset_index()
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Gas price line
            fig.add_trace(
                go.Scatter(
                    x=gas_trends['Timestamp'],
                    y=gas_trends['Avg_Gas_Price'],
                    name="Avg Gas Price (Gwei)",
                    line=dict(color=ETH_COLOR, width=2)
                ),
                secondary_y=False
            )
            
            # Min-Max range
            fig.add_trace(
                go.Scatter(
                    x=gas_trends['Timestamp'],
                    y=gas_trends['Min_Gas_Price'],
                    name="Min Gas Price",
                    line=dict(width=0),
                    showlegend=False
                ),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(
                    x=gas_trends['Timestamp'],
                    y=gas_trends['Max_Gas_Price'],
                    name="Gas Price Range",
                    fill='tonexty',
                    fillcolor=f'rgba{tuple(int(ETH_COLOR.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}',
                    line=dict(width=0),
                ),
                secondary_y=False
            )
            
            # Transaction count line
            fig.add_trace(
                go.Scatter(
                    x=gas_trends['Timestamp'],
                    y=gas_trends['Transaction_Count'],
                    name="Transaction Count",
                    line=dict(color='#F0B90B', width=2, dash='dash')
                ),
                secondary_y=True
            )
            
            # Update layout
            fig.update_layout(
                title_text=f"Ethereum Gas Price Trends ({time_options[time_agg]})",
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=DARK_BG,
                font=dict(color="white"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )
            
            # Update axes
            fig.update_xaxes(
                title_text="Date",
                gridcolor='rgba(255, 255, 255, 0.1)'
            )
            
            fig.update_yaxes(
                title_text="Gas Price (Gwei)",
                secondary_y=False,
                gridcolor='rgba(255, 255, 255, 0.1)'
            )
            
            fig.update_yaxes(
                title_text="Transaction Count",
                secondary_y=True,
                gridcolor='rgba(255, 255, 255, 0.1)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Gas price distribution
            st.subheader("Gas Price Distribution")
            
            fig = px.histogram(
                eth_data, x='Gas_Price_Gwei',
                marginal='box',
                title="Gas Price Distribution",
                color_discrete_sequence=[ETH_COLOR],
                nbins=50
            )
            
            fig.update_layout(
                xaxis_title="Gas Price (Gwei)",
                yaxis_title="Count",
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=DARK_BG,
                font=dict(color="white")
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No Ethereum gas price data available")
    
    # Fee optimization insights
    st.subheader("Fee Optimization Insights")
    
    # Analyze fee efficiency by hour of day
    hourly_fees = data.groupby(['Currency', 'Hour']).agg(
        Avg_Fee=('Transaction_Fee', 'mean'),
        Median_Fee=('Transaction_Fee', 'median'),
        Transaction_Count=('Transaction_ID', 'count')
    ).reset_index()
    
    # Find optimal hours
    btc_optimal = hourly_fees[hourly_fees['Currency'] == 'BTC'].sort_values('Avg_Fee')
    eth_optimal = hourly_fees[hourly_fees['Currency'] == 'ETH'].sort_values('Avg_Fee')
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.write("**Bitcoin Fee Insights**")
        
        if not btc_optimal.empty:
            lowest_fee_hour_btc = btc_optimal.iloc[0]['Hour']
            highest_fee_hour_btc = btc_optimal.iloc[-1]['Hour']
            
            st.markdown(f"""
            - **Lowest fee hour:** {int(lowest_fee_hour_btc)}:00 - {int(lowest_fee_hour_btc)+1}:00 UTC
            - **Highest fee hour:** {int(highest_fee_hour_btc)}:00 - {int(highest_fee_hour_btc)+1}:00 UTC
            - **Fee saving potential:** {(btc_optimal.iloc[-1]['Avg_Fee'] / btc_optimal.iloc[0]['Avg_Fee'] - 1) * 100:.1f}%
            """)
            
            # Hourly fee chart for BTC
            fig = px.line(
                btc_optimal, x='Hour', y='Avg_Fee',
                title="Bitcoin Avg Fees by Hour of Day",
                color_discrete_sequence=[BTC_COLOR]
            )
            
            fig.update_layout(
                xaxis=dict(tickmode='linear', tick0=0, dtick=2),
                xaxis_title="Hour of Day (UTC)",
                yaxis_title="Average Fee (BTC)",
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=DARK_BG,
                font=dict(color="white")
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient Bitcoin transaction data")
    
    with insight_col2:
        st.write("**Ethereum Fee Insights**")
        
        if not eth_optimal.empty:
            lowest_fee_hour_eth = eth_optimal.iloc[0]['Hour']
            highest_fee_hour_eth = eth_optimal.iloc[-1]['Hour']
            
            st.markdown(f"""
            - **Lowest fee hour:** {int(lowest_fee_hour_eth)}:00 - {int(lowest_fee_hour_eth)+1}:00 UTC
            - **Highest fee hour:** {int(highest_fee_hour_eth)}:00 - {int(highest_fee_hour_eth)+1}:00 UTC
            - **Fee saving potential:** {(eth_optimal.iloc[-1]['Avg_Fee'] / eth_optimal.iloc[0]['Avg_Fee'] - 1) * 100:.1f}%
            """)
            
            # Hourly fee chart for ETH
            fig = px.line(
                eth_optimal, x='Hour', y='Avg_Fee',
                title="Ethereum Avg Fees by Hour of Day",
                color_discrete_sequence=[ETH_COLOR]
            )
            
            fig.update_layout(
                xaxis=dict(tickmode='linear', tick0=0, dtick=2),
                xaxis_title="Hour of Day (UTC)",
                yaxis_title="Average Fee (ETH)",
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=DARK_BG,
                font=dict(color="white")
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient Ethereum transaction data")
    
    # Fee by transaction size
    st.subheader("Fees by Transaction Size")
    
    # Calculate average fees by size category
    size_fees = data.groupby(['Currency', 'Size_Category']).agg(
        Avg_Fee=('Transaction_Fee', 'mean'),
        Median_Fee=('Transaction_Fee', 'median'),
        Avg_Fee_Pct=('Fee_Percentage', 'mean'),
        Transaction_Count=('Transaction_ID', 'count')
    ).reset_index()
    
    # Sort size categories correctly
    size_order = ['<0.1', '0.1-1', '1-10', '10-100', '>100']
    size_fees['Size_Category'] = pd.Categorical(size_fees['Size_Category'], categories=size_order, ordered=True)
    size_fees = size_fees.sort_values(['Currency', 'Size_Category'])
    
    # Create chart
    fig = px.bar(
        size_fees, x='Size_Category', y='Avg_Fee', color='Currency',
        barmode='group',
        color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR},
        title="Average Fee by Transaction Size"
    )
    
    fig.update_layout(
        xaxis_title="Transaction Size",
        yaxis_title="Average Fee",
        legend_title="Currency",
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color="white")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Fee percentage by size
    fig = px.bar(
        size_fees, x='Size_Category', y='Avg_Fee_Pct', color='Currency',
        barmode='group',
        color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR},
        title="Average Fee Percentage by Transaction Size"
    )
    
    fig.update_layout(
        xaxis_title="Transaction Size",
        yaxis_title="Average Fee Percentage (%)",
        legend_title="Currency",
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color="white")
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Address Analysis
def analyze_address_activity(data):
    """Analyze wallet address activity and patterns."""
    st.header("📨 Address Activity Analysis")
    
    # Basic address statistics
    sender_counts = data['Sender_Address'].value_counts()
    receiver_counts = data['Receiver_Address'].value_counts()
    
    # Summary metrics
    unique_senders = len(sender_counts)
    unique_receivers = len(receiver_counts)
    total_addresses = len(set(list(data['Sender_Address']) + list(data['Receiver_Address'])))
    one_time_senders = sum(sender_counts == 1)
    one_time_receivers = sum(receiver_counts == 1)
    
    # Display statistics
    st.subheader("Address Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Unique Addresses", f"{total_addresses:,}")
        st.metric("One-time Senders", f"{one_time_senders:,} ({one_time_senders/unique_senders*100:.1f}%)")
        
    with col2:
        st.metric("Unique Senders", f"{unique_senders:,}")
        st.metric("One-time Receivers", f"{one_time_receivers:,} ({one_time_receivers/unique_receivers*100:.1f}%)")
        
    with col3:
        st.metric("Unique Receivers", f"{unique_receivers:,}")
        st.metric("Addresses Both Sending & Receiving", 
                 f"{len(set(data['Sender_Address']) & set(data['Receiver_Address'])):,}")
    
    # Activity distribution
    st.subheader("Activity Distribution")
    
    # Sender activity
    activity_tab1, activity_tab2 = st.tabs(["Sender Activity", "Receiver Activity"])
    
    with activity_tab1:
        # Get top senders
        top_n = st.slider("Number of top senders to display", 10, 100, 20, key="top_senders")
        top_senders = sender_counts.head(top_n).reset_index()
        top_senders.columns = ['Address', 'Transaction_Count']
        
        # Calculate percentage of total
        total_tx = len(data)
        top_senders['Percentage'] = top_senders['Transaction_Count'] / total_tx * 100
        
        # Create chart
        fig = px.bar(
            top_senders, x='Address', y='Transaction_Count',
            title=f"Top {top_n} Senders by Transaction Count",
            color='Transaction_Count',
            color_continuous_scale=px.colors.sequential.Viridis,
            height=500
        )
        
        fig.update_layout(
            xaxis_title="Address",
            yaxis_title="Transaction Count",
            xaxis_tickangle=45,
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=DARK_BG,
            font=dict(color="white")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add concentrated activity metric
        top_percentage = top_senders['Transaction_Count'].sum() / total_tx * 100
        st.info(f"The top {top_n} senders account for {top_percentage:.2f}% of all transactions")
        
        # Show sender activity distribution
        st.subheader("Sender Activity Distribution")
        
        # Create activity buckets
        tx_buckets = [1, 2, 5, 10, 20, 50, 100, sender_counts.max()]
        bucket_labels = [f"1", f"2-4", f"5-9", f"10-19", f"20-49", f"50-99", f"100+"]
        
        bucket_counts = []
        for i in range(len(tx_buckets)-1):
            if i == 0:
                count = sum(sender_counts == tx_buckets[i])
            else:
                count = sum((sender_counts > tx_buckets[i-1]) & (sender_counts <= tx_buckets[i]))
            bucket_counts.append(count)
        bucket_counts.append(sum(sender_counts > tx_buckets[-2]))
        
        # Create distribution chart
        fig = px.bar(
            x=bucket_labels, y=bucket_counts,
            title="Distribution of Sender Activity",
            color=bucket_counts,
            color_continuous_scale=px.colors.sequential.Viridis,
            labels={'x': 'Number of Transactions', 'y': 'Number of Addresses'}
        )
        
        fig.update_layout(
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=DARK_BG,
            font=dict(color="white")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with activity_tab2:
        # Get top receivers
        top_n = st.slider("Number of top receivers to display", 10, 100, 20, key="top_receivers")
        top_receivers = receiver_counts.head(top_n).reset_index()
        top_receivers.columns = ['Address', 'Transaction_Count']
        
        # Calculate percentage of total
        top_receivers['Percentage'] = top_receivers['Transaction_Count'] / total_tx * 100
        
        # Create chart
        fig = px.bar(
            top_receivers, x='Address', y='Transaction_Count',
            title=f"Top {top_n} Receivers by Transaction Count",
            color='Transaction_Count',
            color_continuous_scale=px.colors.sequential.Viridis,
            height=500
        )
        
        fig.update_layout(
            xaxis_title="Address",
            yaxis_title="Transaction Count",
            xaxis_tickangle=45,
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=DARK_BG,
            font=dict(color="white")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show receiver activity distribution
        st.subheader("Receiver Activity Distribution")
        
        # Create activity buckets (same as sender buckets)
        bucket_counts = []
        for i in range(len(tx_buckets)-1):
            if i == 0:
                count = sum(receiver_counts == tx_buckets[i])
            else:
                count = sum((receiver_counts > tx_buckets[i-1]) & (receiver_counts <= tx_buckets[i]))
            bucket_counts.append(count)
        bucket_counts.append(sum(receiver_counts > tx_buckets[-2]))
        
        # Create distribution chart
        fig = px.bar(
            x=bucket_labels, y=bucket_counts,
            title="Distribution of Receiver Activity",
            color=bucket_counts,
            color_continuous_scale=px.colors.sequential.Viridis,
            labels={'x': 'Number of Transactions', 'y': 'Number of Addresses'}
        )
        
        fig.update_layout(
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=DARK_BG,
            font=dict(color="white")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Address clustering analysis
    st.subheader("Address Clustering")
    
    # Prepare data for clustering (top addresses)
    min_tx_threshold = st.slider("Minimum transactions for clustering", 5, 100, 20)
    
    active_senders = sender_counts[sender_counts >= min_tx_threshold].index.tolist()
    active_receivers = receiver_counts[receiver_counts >= min_tx_threshold].index.tolist()
    
    if not active_senders or not active_receivers:
        st.warning(f"Insufficient addresses with {min_tx_threshold}+ transactions for clustering")
        return
    
    # Create activity dataframe
    sender_stats = data[data['Sender_Address'].isin(active_senders)].groupby('Sender_Address').agg(
        Tx_Count=('Transaction_ID', 'count'),
        Avg_Amount_Sent=('Amount', 'mean'),
        Total_Sent=('Amount', 'sum')
    ).reset_index()
    
    receiver_stats = data[data['Receiver_Address'].isin(active_receivers)].groupby('Receiver_Address').agg(
        Tx_Count=('Transaction_ID', 'count'),
        Avg_Amount_Received=('Amount', 'mean'),
        Total_Received=('Amount', 'sum')
    ).reset_index()
    
    # Merge sender and receiver stats
    address_stats = pd.merge(
        sender_stats, 
        receiver_stats, 
        left_on='Sender_Address', 
        right_on='Receiver_Address', 
        how='outer'
    )
    
    # Fill NaN values
    address_stats.fillna(0, inplace=True)
    
    # Select features for clustering
    features = address_stats[['Tx_Count_x', 'Avg_Amount_Sent', 'Total_Sent', 
                             'Tx_Count_y', 'Avg_Amount_Received', 'Total_Received']]
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Determine optimal clusters using elbow method
    distortions = []
    max_clusters = min(10, len(features_scaled)-1)
    
    if max_clusters < 2:
        st.warning("Not enough data points for clustering")
        return
    
    for i in range(1, max_clusters+1):
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(features_scaled)
        distortions.append(km.inertia_)
    
    # Plot elbow curve
    fig = px.line(
        x=range(1, max_clusters+1), 
        y=distortions,
        title='Elbow Method for Optimal Cluster Number',
        labels={'x': 'Number of Clusters', 'y': 'Distortion'}
    )
    
    fig.update_layout(
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color="white")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Perform clustering with selected number of clusters
    n_clusters = st.slider("Select number of clusters", 2, max_clusters, min(3, max_clusters))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)
    
    # Add clusters to address stats
    address_stats['Cluster'] = clusters
    
    # Visualize clusters
    st.subheader("Cluster Visualization")
    
    # Select two features to plot
    feature1 = st.selectbox("First feature for visualization", 
                           ['Tx_Count_x', 'Avg_Amount_Sent', 'Total_Sent'],
                           index=0)
    feature2 = st.selectbox("Second feature for visualization", 
                           ['Tx_Count_y', 'Avg_Amount_Received', 'Total_Received'],
                           index=0)
    
    fig = px.scatter(
        address_stats,
        x=feature1,
        y=feature2,
        color='Cluster',
        hover_name='Sender_Address',
        title=f"Address Clusters ({feature1} vs {feature2})",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    fig.update_layout(
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color="white")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster characteristics
    st.subheader("Cluster Characteristics")
    
    cluster_summary = address_stats.groupby('Cluster').agg({
        'Tx_Count_x': ['mean', 'median', 'count'],
        'Avg_Amount_Sent': 'mean',
        'Total_Sent': 'mean',
        'Tx_Count_y': 'mean',
        'Avg_Amount_Received': 'mean',
        'Total_Received': 'mean'
    }).reset_index()
    
    st.dataframe(cluster_summary.style.background_gradient(cmap='viridis'))
    
    # Network graph visualization (for a sample of addresses)
    st.subheader("Transaction Network")
    
    sample_size = st.slider("Network sample size", 10, 200, 50)
    
    if len(data) > sample_size:
        sample_data = data.sample(sample_size)
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for _, row in sample_data.iterrows():
            G.add_edge(row['Sender_Address'], row['Receiver_Address'], 
                      weight=row['Amount'], 
                      tx_id=row['Transaction_ID'])
        
        # Node positions
        pos = nx.spring_layout(G)
        
        # Create Plotly figure
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos  node_text.append(f"Address: {node[:10]}...")
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=10,
                color=[],
                line_width=2))
        
        # Color nodes by degree
        node_degrees = dict(G.degree())
        node_color = [node_degrees[node] for node in G.nodes()]
        node_trace.marker.color = node_color
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Transaction Network Sample',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=0,l=0,r=0,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor=DARK_BG,
                           paper_bgcolor=DARK_BG,
                           font=dict(color="white"))
                      )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Insufficient data for network visualization")

# Main Application
def main():
    """Main application function."""
    local_css()
    
    st.title("🔍 CryptoInsight Analytics Dashboard")
    st.markdown("Analyze cryptocurrency transactions, market data, and network patterns")
    
    # Sidebar for data input
    st.sidebar.header("Data Input")
    data_source = st.sidebar.selectbox("Select Data Source", ["Demo Data", "Upload CSV"])
    
    if data_source == "Demo Data":
        st.sidebar.info("Using synthetic transaction data for demonstration")
        data = generate_demo_data(n_transactions=1000)
    else:
        uploaded_file = st.sidebar.file_uploader("Upload transaction CSV", type=['csv'])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
        else:
            st.error("Please upload a CSV file or select Demo Data")
            return
    
    # Initialize preprocessor
    preprocessor = CryptoDataPreprocessor(data)
    processed_data, btc_data, eth_data = preprocessor.get_processed_data()
    
    # Display data summary
    preprocessor._summarize_data()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Market Data",
        "Mining Pools",
        "Transaction Fees",
        "Address Analysis"
    ])
    
    with tab1:
        show_price_tab()
        
    with tab2:
        analyze_mining_pools(processed_data)
        
    with tab3:
        analyze_transaction_fees(processed_data)
        
    with tab4:
        analyze_address_activity(processed_data)

if __name__ == "__main__":
    main()
