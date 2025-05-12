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
import json
import warnings

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')
pd.set_option('display.max_columns', None)

# Set page config
st.set_page_config(
    page_title="Cryptocurrency Analytics Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .st-bb {
        background-color: transparent;
    }
    .st-at {
        background-color: #0E1117;
    }
    footer {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

class CryptoDataPreprocessor:
    def __init__(self, data):
        self.data = data.copy()
        self.btc_data = None
        self.eth_data = None
        self.preprocessed = False
        self.btc_color = '#F7931A'  # Bitcoin orange
        self.eth_color = '#627EEA'  # Ethereum blue

    def preprocess(self):
        """Clean and prepare the transaction data"""
        st.write("Starting data preprocessing...")

        # Convert timestamp to datetime
        self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])

        # Create time-based features
        self.data['Date'] = self.data['Timestamp'].dt.date
        self.data['Hour'] = self.data['Timestamp'].dt.hour
        self.data['Day_of_Week'] = self.data['Timestamp'].dt.day_name()
        self.data['Day_of_Week_Num'] = self.data['Timestamp'].dt.dayofweek

        # Separate by currency
        self.btc_data = self.data[self.data['Currency'] == 'BTC']
        self.eth_data = self.data[self.data['Currency'] == 'ETH']

        # Calculate fee percentage
        self.data['Fee_Percentage'] = (self.data['Transaction_Fee'] / self.data['Amount']) * 100

        # Ethereum-specific calculations
        if 'Gas_Price_Gwei' in self.data.columns:
            self.data['Gas_Cost_ETH'] = self.data['Gas_Price_Gwei'] * 1e-9 * 21000

        self.preprocessed = True
        st.success("Data preprocessing completed.")
        self._summarize_data()

    def _summarize_data(self):
        """Print dataset summary"""
        st.subheader("Dataset Summary")
        st.write(f"Date range: {self.data['Timestamp'].min().date()} to {self.data['Timestamp'].max().date()}")
        st.write(f"Total transactions: {len(self.data):,}")
        st.write(f"Bitcoin (BTC) transactions: {len(self.btc_data):,} ({len(self.btc_data)/len(self.data)*100:.1f}%)")
        st.write(f"Ethereum (ETH) transactions: {len(self.eth_data):,} ({len(self.eth_data)/len(self.data)*100:.1f}%)")
        st.write(f"Unique addresses: {self.data['Sender_Address'].nunique():,} senders, {self.data['Receiver_Address'].nunique():,} receivers")
        
        if 'Mining_Pool' in self.data.columns:
            st.write(f"Mining pools: {self.data['Mining_Pool'].nunique()}")

        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Transaction status distribution:")
            st.write(self.data['Transaction_Status'].value_counts(normalize=True).map("{:.1%}".format))
        
        with col2:
            st.write("Transaction type distribution:")
            st.write(self.data['Transaction_Type'].value_counts(normalize=True).map("{:.1%}".format))

    def get_processed_data(self):
        """Return processed datasets"""
        if not self.preprocessed:
            self.preprocess()
        return self.data, self.btc_data, self.eth_data

def get_crypto_prices():
    """Fetch current cryptocurrency prices from CoinGecko API"""
    url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids=bitcoin,ethereum"
    try:
        response = requests.get(url)
        data = response.json()
        btc_price = next(item for item in data if item["id"] == "bitcoin")["current_price"]
        eth_price = next(item for item in data if item["id"] == "ethereum")["current_price"]
        
        # Get historical data for charts
        btc_history = requests.get("https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=30").json()
        eth_history = requests.get("https://api.coingecko.com/api/v3/coins/ethereum/market_chart?vs_currency=usd&days=30").json()
        
        return btc_price, eth_price, btc_history, eth_history
    except Exception as e:
        st.error(f"Error fetching crypto prices: {e}")
        return None, None, None, None

def show_price_tab():
    """Display current cryptocurrency prices and charts"""
    st.header("Current Cryptocurrency Prices")
    
    btc_price, eth_price, btc_history, eth_history = get_crypto_prices()
    
    if btc_price and eth_price:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Bitcoin (BTC)", f"${btc_price:,.2f}")
            
        with col2:
            st.metric("Ethereum (ETH)", f"${eth_price:,.2f}")
        
        # Price charts
        st.subheader("30-Day Price History")
        
        if btc_history and eth_history:
            # Process BTC data
            btc_dates = [datetime.fromtimestamp(x[0]/1000) for x in btc_history['prices']]
            btc_prices = [x[1] for x in btc_history['prices']]
            btc_df = pd.DataFrame({'Date': btc_dates, 'Price': btc_prices, 'Currency': 'BTC'})
            
            # Process ETH data
            eth_dates = [datetime.fromtimestamp(x[0]/1000) for x in eth_history['prices']]
            eth_prices = [x[1] for x in eth_history['prices']]
            eth_df = pd.DataFrame({'Date': eth_dates, 'Price': eth_prices, 'Currency': 'ETH'})
            
            # Combine data
            combined_df = pd.concat([btc_df, eth_df])
            
            # Plot
            fig = px.line(combined_df, x='Date', y='Price', color='Currency', 
                         title='Bitcoin vs Ethereum Price History',
                         color_discrete_map={'BTC': '#F7931A', 'ETH': '#627EEA'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume charts
            st.subheader("30-Day Trading Volume")
            
            # Process BTC volume
            btc_volumes = [x[1] for x in btc_history['total_volumes']]
            btc_vol_df = pd.DataFrame({'Date': btc_dates, 'Volume': btc_volumes, 'Currency': 'BTC'})
            
            # Process ETH volume
            eth_volumes = [x[1] for x in eth_history['total_volumes']]
            eth_vol_df = pd.DataFrame({'Date': eth_dates, 'Volume': eth_volumes, 'Currency': 'ETH'})
            
            # Combine data
            combined_vol_df = pd.concat([btc_vol_df, eth_vol_df])
            
            # Plot
            fig = px.bar(combined_vol_df, x='Date', y='Volume', color='Currency',
                        title='Bitcoin vs Ethereum Trading Volume',
                        color_discrete_map={'BTC': '#F7931A', 'ETH': '#627EEA'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not fetch cryptocurrency prices. Please try again later.")

def analyze_mining_pools(data):
    """Analyze mining pool activity"""
    st.header("Mining Pool Analysis")
    
    if 'Mining_Pool' not in data.columns:
        st.warning("No mining pool data available")
        return None, None

    # Pool statistics
    pool_counts = data.groupby(['Mining_Pool', 'Currency']).agg(
        Transaction_Count=('Transaction_ID', 'count'),
        Total_Amount=('Amount', 'sum'),
        Avg_Fee=('Transaction_Fee', 'mean')
    ).reset_index()

    # Top 10 pools
    top_pools = pool_counts.sort_values('Transaction_Count', ascending=False).head(10)

    # Visualizations
    st.subheader("Top Mining Pools by Transaction Count")
    fig = px.bar(top_pools, x='Mining_Pool', y='Transaction_Count', color='Currency',
                color_discrete_map={'BTC': '#F7931A', 'ETH': '#627EEA'})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Fees by Mining Pool")
    fig = px.box(pool_counts, x='Mining_Pool', y='Avg_Fee', color='Currency')
    st.plotly_chart(fig, use_container_width=True)

    # Pool efficiency analysis
    pool_daily = data.groupby(['Mining_Pool', 'Currency', 'Date']).size().reset_index(name='Daily_Tx')
    pool_efficiency = pool_daily.groupby(['Mining_Pool', 'Currency']).agg(
        Avg_Daily_Tx=('Daily_Tx', 'mean'),
        Std_Daily_Tx=('Daily_Tx', 'std')
    ).reset_index()

    st.subheader("Mining Pool Activity Consistency")
    fig = px.scatter(pool_efficiency, x='Avg_Daily_Tx', y='Std_Daily_Tx',
                    color='Currency', size='Avg_Daily_Tx',
                    hover_name='Mining_Pool')
    st.plotly_chart(fig, use_container_width=True)

    return pool_counts, pool_efficiency

def analyze_transaction_fees(data):
    """Analyze fee patterns"""
    st.header("Transaction Fee Analysis")

    # Fee statistics
    fee_stats = data.groupby('Currency')['Transaction_Fee'].agg([
        'count', 'mean', 'median', 'min', 'max', 'std'
    ]).reset_index()

    st.subheader("Fee Statistics")
    st.dataframe(fee_stats.style.format({
        'mean': '{:.4f}',
        'median': '{:.4f}',
        'std': '{:.4f}'
    }))

    # Daily fees
    daily_fees = data.groupby(['Date', 'Currency'])['Transaction_Fee'].mean().reset_index()

    st.subheader("Daily Average Fees")
    fig = px.line(daily_fees, x='Date', y='Transaction_Fee', color='Currency',
                 color_discrete_map={'BTC': '#F7931A', 'ETH': '#627EEA'})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Fee Distribution")
    fig = px.histogram(data, x='Transaction_Fee', color='Currency',
                      marginal='box',
                      color_discrete_map={'BTC': '#F7931A', 'ETH': '#627EEA'})
    fig.update_layout(barmode='overlay')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Fee vs Amount")
    fig = px.scatter(data[data['Fee_Percentage'] < 10],
                    x='Amount', y='Transaction_Fee', color='Currency',
                    trendline='lowess',
                    color_discrete_map={'BTC': '#F7931A', 'ETH': '#627EEA'})
    st.plotly_chart(fig, use_container_width=True)

    # Ethereum gas analysis
    if 'Gas_Price_Gwei' in data.columns:
        eth_data = data[data['Currency'] == 'ETH']
        
        st.subheader("Ethereum Gas Analysis")
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Gas Price Distribution', 'Gas Price Over Time'))

        fig.add_trace(go.Histogram(x=eth_data['Gas_Price_Gwei'], name='Gas Price'),
                     row=1, col=1)

        daily_gas = eth_data.groupby('Date')['Gas_Price_Gwei'].mean().reset_index()
        fig.add_trace(go.Scatter(x=daily_gas['Date'], y=daily_gas['Gas_Price_Gwei'],
                                mode='lines', name='Avg Gas Price'),
                     row=1, col=2)

        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    return fee_stats, daily_fees

def analyze_address_activity(data, top_n=10):
    """Analyze address patterns"""
    st.header("Address Activity Analysis")
    
    top_n = st.slider("Select number of top addresses to display", 5, 50, 10)

    # Top senders
    top_senders = data.groupby(['Sender_Address', 'Currency']).agg(
        Transaction_Count=('Transaction_ID', 'count'),
        Total_Sent=('Amount', 'sum')
    ).sort_values('Transaction_Count', ascending=False).head(top_n)

    # Top receivers
    top_receivers = data.groupby(['Receiver_Address', 'Currency']).agg(
        Transaction_Count=('Transaction_ID', 'count'),
        Total_Received=('Amount', 'sum')
    ).sort_values('Transaction_Count', ascending=False).head(top_n)

    # Visualizations
    st.subheader(f"Top {top_n} Senders")
    fig = px.bar(top_senders.reset_index(), x='Sender_Address', y='Transaction_Count',
                color='Currency',
                color_discrete_map={'BTC': '#F7931A', 'ETH': '#627EEA'})
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Top {top_n} Receivers")
    fig = px.bar(top_receivers.reset_index(), x='Receiver_Address', y='Transaction_Count',
                color='Currency',
                color_discrete_map={'BTC': '#F7931A', 'ETH': '#627EEA'})
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    # Network analysis (sampled)
    sample_size = st.slider("Select sample size for network analysis", 100, 5000, 1000, step=100)
    sample_data = data.sample(min(sample_size, len(data)), random_state=42)
    G = nx.from_pandas_edgelist(sample_data, 'Sender_Address', 'Receiver_Address',
                               edge_attr=['Amount', 'Currency'],
                               create_using=nx.DiGraph())

    st.subheader("Network Analysis")
    st.write(f"Nodes: {G.number_of_nodes():,}, Edges: {G.number_of_edges():,}")
    st.write(f"Density: {nx.density(G):.6f}")

    # Simple network visualization
    st.write("Network visualization (first 100 nodes for performance):")
    G_small = G.subgraph(list(G.nodes())[:100])
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw(G_small, with_labels=False, node_size=20, ax=ax)
    st.pyplot(fig)

    return top_senders, top_receivers, G

def detect_temporal_patterns(data):
    """Analyze time patterns and anomalies"""
    st.header("Temporal Patterns Analysis")

    # Day of week analysis
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_pattern = data.groupby(['Day_of_Week', 'Currency']).size().reset_index(name='Count')
    daily_pattern['Day_Index'] = daily_pattern['Day_of_Week'].apply(lambda x: day_order.index(x))
    daily_pattern = daily_pattern.sort_values('Day_Index')

    st.subheader("Transactions by Day of Week")
    fig = px.bar(daily_pattern, x='Day_of_Week', y='Count', color='Currency',
                category_orders={'Day_of_Week': day_order},
                color_discrete_map={'BTC': '#F7931A', 'ETH': '#627EEA'})
    st.plotly_chart(fig, use_container_width=True)

    # Hourly patterns
    hourly_pattern = data.groupby(['Hour', 'Currency']).size().reset_index(name='Count')
    
    st.subheader("Transactions by Hour of Day")
    fig = px.line(hourly_pattern, x='Hour', y='Count', color='Currency',
                 color_discrete_map={'BTC': '#F7931A', 'ETH': '#627EEA'})
    st.plotly_chart(fig, use_container_width=True)

    # Anomaly detection
    st.subheader("Anomaly Detection")
    z_threshold = st.slider("Select Z-score threshold for anomalies", 2.0, 5.0, 3.0, 0.5)
    
    data['Amount_ZScore'] = data.groupby('Currency')['Amount'].transform(
        lambda x: (x - x.mean()) / x.std())

    anomalies = data[abs(data['Amount_ZScore']) > z_threshold].copy()

    if not anomalies.empty:
        st.write(f"Detected {len(anomalies)} anomalies (Z-score > {z_threshold})")
        
        fig = px.scatter(anomalies, x='Timestamp', y='Amount', color='Currency',
                        size='Amount_ZScore', hover_data=['Amount_ZScore'],
                        color_discrete_map={'BTC': '#F7931A', 'ETH': '#627EEA'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("Top anomalies:")
        st.dataframe(anomalies.sort_values('Amount_ZScore', ascending=False).head())
    else:
        st.warning(f"No anomalies detected with Z-score > {z_threshold}")

    return daily_pattern, anomalies

def main():
    """Main function to run the Streamlit app"""
    st.title("Cryptocurrency Transaction Analytics Dashboard")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Cryptocurrency Transaction Data", type=["csv"])
    
    if uploaded_file is not None:
        try:
            raw_data = pd.read_csv(uploaded_file)
            st.sidebar.success("Data loaded successfully!")
            
            # Initialize and preprocess data
            preprocessor = CryptoDataPreprocessor(raw_data)
            data, btc_data, eth_data = preprocessor.get_processed_data()
            
            # Create tabs
            tabs = ["Price Data", "Data Overview", "Mining Pools", "Transaction Fees", 
                   "Address Activity", "Temporal Patterns"]
            selected_tab = st.sidebar.radio("Select Analysis", tabs)
            
            if selected_tab == "Price Data":
                show_price_tab()
                
            elif selected_tab == "Data Overview":
                st.header("Data Overview")
                preprocessor._summarize_data()
                
                st.subheader("Sample BTC Transactions")
                st.dataframe(btc_data.head())
                
                st.subheader("Sample ETH Transactions")
                st.dataframe(eth_data.head())
                
            elif selected_tab == "Mining Pools":
                analyze_mining_pools(data)
                
            elif selected_tab == "Transaction Fees":
                analyze_transaction_fees(data)
                
            elif selected_tab == "Address Activity":
                analyze_address_activity(data)
                
            elif selected_tab == "Temporal Patterns":
                detect_temporal_patterns(data)
                
        except Exception as e:
            st.error(f"Error processing data: {e}")
    else:
        st.info("Please upload a CSV file to begin analysis")
        
        # Show price tab even without data upload
        if st.checkbox("Show cryptocurrency prices without uploading data"):
            show_price_tab()

if __name__ == "__main__":
    main()
