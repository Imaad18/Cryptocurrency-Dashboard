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
from PIL import Image
import io
import base64
from faker import Faker

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')
pd.set_option('display.max_columns', None)

# Set page config
st.set_page_config(
    page_title="CryptoInsight Analytics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load custom CSS
local_css("style.css")

# Constants
BTC_COLOR = '#F7931A'
ETH_COLOR = '#627EEA'
DARK_BG = '#0E1117'
PLOT_BG = '#1E1E1E'

# Initialize Faker for demo data generation
fake = Faker()

class CryptoDataPreprocessor:
    def __init__(self, data):
        self.data = data.copy()
        self.btc_data = None
        self.eth_data = None
        self.preprocessed = False

    def preprocess(self):
        """Clean and prepare the transaction data"""
        with st.spinner('Preprocessing data...'):
            # Convert timestamp to datetime
            self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])

            # Create time-based features
            self.data['Date'] = self.data['Timestamp'].dt.date
            self.data['Hour'] = self.data['Timestamp'].dt.hour
            self.data['Day_of_Week'] = self.data['Timestamp'].dt.day_name()
            self.data['Day_of_Week_Num'] = self.data['Timestamp'].dt.dayofweek
            self.data['Month'] = self.data['Timestamp'].dt.month_name()
            self.data['Year'] = self.data['Timestamp'].dt.year

            # Separate by currency
            self.btc_data = self.data[self.data['Currency'] == 'BTC']
            self.eth_data = self.data[self.data['Currency'] == 'ETH']

            # Calculate fee percentage
            self.data['Fee_Percentage'] = (self.data['Transaction_Fee'] / self.data['Amount']) * 100

            # Ethereum-specific calculations
            if 'Gas_Price_Gwei' in self.data.columns:
                self.data['Gas_Cost_ETH'] = self.data['Gas_Price_Gwei'] * 1e-9 * 21000

            # Calculate transaction size categories
            self.data['Size_Category'] = pd.cut(
                self.data['Amount'],
                bins=[0, 0.1, 1, 10, 100, float('inf')],
                labels=['<0.1', '0.1-1', '1-10', '10-100', '>100']
            )

            self.preprocessed = True
        st.success('Data preprocessing completed!')
        self._summarize_data()

    def _summarize_data(self):
        """Print dataset summary"""
        with st.expander("📊 Dataset Summary", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Transactions", f"{len(self.data):,}")
                st.metric("BTC Transactions", f"{len(self.btc_data):,} ({len(self.btc_data)/len(self.data)*100:.1f}%)")
                
            with col2:
                st.metric("Date Range", 
                         f"{self.data['Timestamp'].min().date()} to {self.data['Timestamp'].max().date()}")
                st.metric("ETH Transactions", f"{len(self.eth_data):,} ({len(self.eth_data)/len(self.data)*100:.1f}%)")
                
            with col3:
                st.metric("Unique Senders", f"{self.data['Sender_Address'].nunique():,}")
                st.metric("Unique Receivers", f"{self.data['Receiver_Address'].nunique():,}")

            if 'Mining_Pool' in self.data.columns:
                st.metric("Mining Pools", f"{self.data['Mining_Pool'].nunique()}")

            # Transaction status and type distribution
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

            # Hourly transaction patterns
            st.subheader("Hourly Transaction Patterns")
            hourly_data = self.data.groupby(['Hour', 'Currency']).size().reset_index(name='Count')
            fig = px.line(hourly_data, x='Hour', y='Count', color='Currency',
                         title='Transactions by Hour of Day',
                         color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR})
            st.plotly_chart(fig, use_container_width=True)

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
        btc_data = next(item for item in data if item["id"] == "bitcoin")
        eth_data = next(item for item in data if item["id"] == "ethereum")
        
        # Get historical data for charts
        btc_history = requests.get("https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=90&interval=daily").json()
        eth_history = requests.get("https://api.coingecko.com/api/v3/coins/ethereum/market_chart?vs_currency=usd&days=90&interval=daily").json()
        
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

def generate_demo_data():
    """Generate synthetic demo data if no file is uploaded"""
    num_transactions = 5000
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    
    data = {
        'Transaction_ID': [fake.uuid4() for _ in range(num_transactions)],
        'Timestamp': [fake.date_time_between(start_date=start_date, end_date=end_date) for _ in range(num_transactions)],
        'Sender_Address': [fake.sha256()[:34] for _ in range(num_transactions)],
        'Receiver_Address': [fake.sha256()[:34] for _ in range(num_transactions)],
        'Amount': np.random.exponential(0.5, num_transactions).round(6),
        'Transaction_Fee': np.random.exponential(0.001, num_transactions).round(6),
        'Currency': np.random.choice(['BTC', 'ETH'], num_transactions, p=[0.6, 0.4]),
        'Transaction_Status': np.random.choice(['Confirmed', 'Pending', 'Failed'], num_transactions, p=[0.85, 0.1, 0.05]),
        'Transaction_Type': np.random.choice(['Regular', 'Contract', 'Token'], num_transactions, p=[0.7, 0.2, 0.1]),
        'Mining_Pool': np.random.choice(['AntPool', 'F2Pool', 'Poolin', 'ViaBTC', 'Unknown'], num_transactions),
        'Gas_Price_Gwei': np.where(np.random.choice(['BTC', 'ETH'], num_transactions, p=[0.6, 0.4]) == 'ETH',
                          np.random.normal(50, 15, num_transactions).clip(10, 200), np.nan)
    }
    
    return pd.DataFrame(data)

def show_price_tab():
    """Display current cryptocurrency prices and charts"""
    st.header("💰 Live Market Data")
    
    price_data = get_crypto_prices()
    
    if price_data:
        # Price cards
        col1, col2 = st.columns(2)
        
        with col1:
            btc_change_color = "green" if price_data['BTC']['change_24h'] >= 0 else "red"
            st.metric(
                "Bitcoin (BTC)", 
                f"${price_data['BTC']['price']:,.2f}", 
                f"{price_data['BTC']['change_24h']:.2f}%",
                delta_color="normal"
            )
            st.caption(f"Market Cap: ${price_data['BTC']['market_cap']/1e9:,.1f}B")
            st.caption(f"24h Volume: ${price_data['BTC']['volume']/1e9:,.1f}B")
            
        with col2:
            eth_change_color = "green" if price_data['ETH']['change_24h'] >= 0 else "red"
            st.metric(
                "Ethereum (ETH)", 
                f"${price_data['ETH']['price']:,.2f}", 
                f"{price_data['ETH']['change_24h']:.2f}%",
                delta_color="normal"
            )
            st.caption(f"Market Cap: ${price_data['ETH']['market_cap']/1e9:,.1f}B")
            st.caption(f"24h Volume: ${price_data['ETH']['volume']/1e9:,.1f}B")
        
        # Price charts
        st.subheader("90-Day Price History")
        tab1, tab2, tab3 = st.tabs(["Combined View", "Bitcoin", "Ethereum"])
        
        with tab1:
            # Process BTC data
            btc_dates = [datetime.fromtimestamp(x[0]/1000) for x in price_data['BTC']['history']['prices']]
            btc_prices = [x[1] for x in price_data['BTC']['history']['prices']]
            btc_df = pd.DataFrame({'Date': btc_dates, 'Price': btc_prices, 'Currency': 'BTC'})
            
            # Process ETH data
            eth_dates = [datetime.fromtimestamp(x[0]/1000) for x in price_data['ETH']['history']['prices']]
            eth_prices = [x[1] for x in price_data['ETH']['history']['prices']]
            eth_df = pd.DataFrame({'Date': eth_dates, 'Price': eth_prices, 'Currency': 'ETH'})
            
            # Combine data
            combined_df = pd.concat([btc_df, eth_df])
            
            # Plot
            fig = px.line(combined_df, x='Date', y='Price', color='Currency', 
                         title='Bitcoin vs Ethereum Price History',
                         color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR},
                         template='plotly_dark')
            fig.update_layout(
                hovermode='x unified',
                yaxis_title="Price (USD)",
                xaxis_title="Date",
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=DARK_BG
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=btc_dates, y=btc_prices,
                name='BTC Price',
                line=dict(color=BTC_COLOR, width=2),
                fill='tozeroy',
                fillcolor=f'rgba(247, 147, 26, 0.2)'
            ))
            fig.update_layout(
                title='Bitcoin Price History',
                yaxis_title="Price (USD)",
                xaxis_title="Date",
                template='plotly_dark',
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=DARK_BG
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=eth_dates, y=eth_prices,
                name='ETH Price',
                line=dict(color=ETH_COLOR, width=2),
                fill='tozeroy',
                fillcolor=f'rgba(98, 126, 234, 0.2)'
            ))
            fig.update_layout(
                title='Ethereum Price History',
                yaxis_title="Price (USD)",
                xaxis_title="Date",
                template='plotly_dark',
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=DARK_BG
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Market indicators
        st.subheader("Market Indicators")
        indicator_col1, indicator_col2, indicator_col3 = st.columns(3)
        
        with indicator_col1:
            st.plotly_chart(create_market_indicator(
                "Fear & Greed Index", 
                np.random.randint(0, 100), 
                "https://alternative.me/crypto/fear-and-greed-index.png"
            ), use_container_width=True)
        
        with indicator_col2:
            st.plotly_chart(create_market_indicator(
                "BTC Dominance", 
                np.random.uniform(40, 50), 
                ""
            ), use_container_width=True)
        
        with indicator_col3:
            st.plotly_chart(create_market_indicator(
                "ETH Gas Price", 
                np.random.uniform(20, 100), 
                ""
            ), use_container_width=True)
    else:
        st.warning("Could not fetch cryptocurrency prices. Please try again later.")

def create_market_indicator(title, value, image_url):
    """Create a market indicator gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': BTC_COLOR if "BTC" in title else ETH_COLOR},
            'steps': [
                {'range': [0, 30], 'color': "red"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "green"}
            ]
        }
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor=DARK_BG
    )
    return fig

def analyze_mining_pools(data):
    """Analyze mining pool activity"""
    st.header("⛏️ Mining Pool Analysis")
    
    if 'Mining_Pool' not in data.columns:
        st.warning("No mining pool data available in this dataset")
        return None, None

    # Pool statistics
    pool_counts = data.groupby(['Mining_Pool', 'Currency']).agg(
        Transaction_Count=('Transaction_ID', 'count'),
        Total_Amount=('Amount', 'sum'),
        Avg_Fee=('Transaction_Fee', 'mean'),
        Avg_Amount=('Amount', 'mean')
    ).reset_index()

    # Top pools selection
    top_n = st.slider("Select number of top pools to display", 5, 20, 10)

    # Visualizations
    st.subheader(f"Top {top_n} Mining Pools by Transaction Count")
    top_pools = pool_counts.sort_values('Transaction_Count', ascending=False).head(top_n)
    
    fig = px.bar(top_pools, x='Mining_Pool', y='Transaction_Count', color='Currency',
                color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR},
                template='plotly_dark')
    fig.update_layout(
        xaxis_title="Mining Pool",
        yaxis_title="Transaction Count",
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=DARK_BG
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mining Pool Activity Composition")
    pool_composition = pool_counts.sort_values('Transaction_Count', ascending=False).head(top_n)
    fig = px.sunburst(pool_composition, path=['Currency', 'Mining_Pool'], values='Transaction_Count',
                     color='Currency', color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR})
    fig.update_layout(
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=DARK_BG
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Fees by Mining Pool")
    fig = px.box(pool_counts, x='Mining_Pool', y='Avg_Fee', color='Currency',
                color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR},
                template='plotly_dark')
    fig.update_layout(
        xaxis_title="Mining Pool",
        yaxis_title="Average Fee",
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=DARK_BG
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Transaction Size vs Fee Relationship")
    fig = px.scatter(pool_counts, x='Avg_Amount', y='Avg_Fee', color='Currency',
                    size='Transaction_Count', hover_name='Mining_Pool',
                    color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR},
                    template='plotly_dark')
    fig.update_layout(
        xaxis_title="Average Transaction Amount",
        yaxis_title="Average Fee",
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=DARK_BG
    )
    st.plotly_chart(fig, use_container_width=True)

    return pool_counts

def analyze_transaction_fees(data):
    """Analyze fee patterns"""
    st.header("💸 Transaction Fee Analysis")

    # Fee statistics
    fee_stats = data.groupby('Currency')['Transaction_Fee'].agg([
        'count', 'mean', 'median', 'min', 'max', 'std', 'sum'
    ]).reset_index()

    st.subheader("Fee Statistics")
    st.dataframe(fee_stats.style.format({
        'mean': '{:.6f}',
        'median': '{:.6f}',
        'std': '{:.6f}',
        'sum': '{:.6f}'
    }).background_gradient(cmap='YlOrBr'), use_container_width=True)

    # Time-based fee analysis
    st.subheader("Fee Trends Over Time")
    time_resolution = st.radio("Select time resolution", ['Daily', 'Weekly', 'Monthly'], horizontal=True)
    
    if time_resolution == 'Daily':
        fee_trends = data.groupby(['Date', 'Currency'])['Transaction_Fee'].mean().reset_index()
    elif time_resolution == 'Weekly':
        data['Week'] = data['Timestamp'].dt.to_period('W').dt.start_time
        fee_trends = data.groupby(['Week', 'Currency'])['Transaction_Fee'].mean().reset_index()
    else:  # Monthly
        data['Month'] = data['Timestamp'].dt.to_period('M').dt.start_time
        fee_trends = data.groupby(['Month', 'Currency'])['Transaction_Fee'].mean().reset_index()

    fig = px.line(fee_trends, x=fee_trends.columns[0], y='Transaction_Fee', color='Currency',
                 color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR},
                 template='plotly_dark')
    fig.update_layout(
        xaxis_title=time_resolution,
        yaxis_title="Average Fee",
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=DARK_BG
    )
    st.plotly_chart(fig, use_container_width=True)

    # Fee distribution
    st.subheader("Fee Distribution Analysis")
    dist_col1, dist_col2 = st.columns(2)
    
    with dist_col1:
        fig = px.histogram(data, x='Transaction_Fee', color='Currency',
                          marginal='box', nbins=50,
                          color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR},
                          template='plotly_dark')
        fig.update_layout(
            xaxis_title="Transaction Fee",
            yaxis_title="Count",
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=DARK_BG,
            barmode='overlay'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with dist_col2:
        fig = px.box(data, x='Currency', y='Transaction_Fee', color='Currency',
                    color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR},
                    template='plotly_dark')
        fig.update_layout(
            xaxis_title="Currency",
            yaxis_title="Transaction Fee",
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=DARK_BG,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    # Fee vs Amount
    st.subheader("Fee vs Transaction Amount")
    sample_size = st.slider("Sample size for scatter plot", 100, 5000, 1000)
    sampled_data = data.sample(min(sample_size, len(data)), random_state=42)
    
    fig = px.scatter(sampled_data, x='Amount', y='Transaction_Fee', color='Currency',
                    trendline='lowess', opacity=0.5,
                    color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR},
                    template='plotly_dark')
    fig.update_layout(
        xaxis_title="Transaction Amount",
        yaxis_title="Transaction Fee",
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=DARK_BG
    )
    st.plotly_chart(fig, use_container_width=True)

    # Ethereum gas analysis
    if 'Gas_Price_Gwei' in data.columns:
        st.subheader("Ethereum Gas Analysis")
        eth_data = data[data['Currency'] == 'ETH'].copy()
        
        tab1, tab2 = st.tabs(["Gas Price Trends", "Gas Statistics"])
        
        with tab1:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               subplot_titles=('Gas Price Over Time', 'Daily Gas Price Distribution'))
            
            # Time series
            daily_gas = eth_data.groupby('Date')['Gas_Price_Gwei'].mean().reset_index()
            fig.add_trace(go.Scatter(
                x=daily_gas['Date'], y=daily_gas['Gas_Price_Gwei'],
                name='Avg Gas Price',
                line=dict(color=ETH_COLOR, width=2)
            ), row=1, col=1)
            
            # Hourly distribution
            hourly_gas = eth_data.groupby('Hour')['Gas_Price_Gwei'].mean().reset_index()
            fig.add_trace(go.Bar(
                x=hourly_gas['Hour'], y=hourly_gas['Gas_Price_Gwei'],
                name='Hourly Avg',
                marker_color=ETH_COLOR
            ), row=2, col=1)
            
            fig.update_layout(
                height=600,
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=DARK_BG,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            gas_stats = eth_data['Gas_Price_Gwei'].describe().to_frame().T
            st.dataframe(gas_stats.style.format('{:.2f}').background_gradient(cmap='Blues'), 
                        use_container_width=True)
            
            fig = px.violin(eth_data, y='Gas_Price_Gwei', box=True, points="all",
                           template='plotly_dark')
            fig.update_layout(
                yaxis_title="Gas Price (Gwei)",
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=DARK_BG
            )
            st.plotly_chart(fig, use_container_width=True)

    return fee_stats

def analyze_address_activity(data):
    """Analyze address patterns"""
    st.header("📨 Address Activity Analysis")
    
    top_n = st.slider("Select number of top addresses to display", 5, 50, 10)

    # Top senders
    st.subheader(f"Top {top_n} Active Addresses")
    tab1, tab2 = st.tabs(["Senders", "Receivers"])
    
    with tab1:
        top_senders = data.groupby(['Sender_Address', 'Currency']).agg(
            Transaction_Count=('Transaction_ID', 'count'),
            Total_Sent=('Amount', 'sum'),
            Avg_Amount=('Amount', 'mean')
        ).sort_values('Transaction_Count', ascending=False).head(top_n)
        
        fig = px.bar(top_senders.reset_index(), x='Sender_Address', y='Transaction_Count',
                    color='Currency', text='Transaction_Count',
                    color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR},
                    template='plotly_dark')
        fig.update_layout(
            xaxis_title="Sender Address",
            yaxis_title="Transaction Count",
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=DARK_BG,
            xaxis_tickangle=45
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(top_senders.style.format({
            'Total_Sent': '{:.6f}',
            'Avg_Amount': '{:.6f}'
        }).background_gradient(cmap='Oranges'), use_container_width=True)
    
    with tab2:
        top_receivers = data.groupby(['Receiver_Address', 'Currency']).agg(
            Transaction_Count=('Transaction_ID', 'count'),
            Total_Received=('Amount', 'sum'),
            Avg_Amount=('Amount', 'mean')
        ).sort_values('Transaction_Count', ascending=False).head(top_n)
        
        fig = px.bar(top_receivers.reset_index(), x='Receiver_Address', y='Transaction_Count',
                    color='Currency', text='Transaction_Count',
                    color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR},
                    template='plotly_dark')
        fig.update_layout(
            xaxis_title="Receiver Address",
            yaxis_title="Transaction Count",
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=DARK_BG,
            xaxis_tickangle=45
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(top_receivers.style.format({
            'Total_Received': '{:.6f}',
            'Avg_Amount': '{:.6f}'
        }).background_gradient(cmap='Blues'), use_container_width=True)

    # Network analysis
    st.subheader("Transaction Network Analysis")
    sample_size = st.slider("Select sample size for network analysis", 100, 5000, 1000, step=100)
    sample_data = data.sample(min(sample_size, len(data)), random_state=42)
    
    with st.spinner('Building transaction network...'):
        G = nx.from_pandas_edgelist(sample_data, 'Sender_Address', 'Receiver_Address',
                                   edge_attr=['Amount', 'Currency'],
                                   create_using=nx.DiGraph())

        # Network metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nodes", f"{G.number_of_nodes():,}")
        col2.metric("Edges", f"{G.number_of_edges():,}")
        col3.metric("Density", f"{nx.density(G):.6f}")
        col4.metric("Avg. Degree", f"{sum(dict(G.degree()).values())/G.number_of_nodes():.2f}")

        # Centrality measures
        st.write("**Top Central Addresses**")
        centrality_type = st.selectbox("Select centrality measure", 
                                      ['Degree', 'Betweenness', 'Closeness', 'Eigenvector'])
        
        if centrality_type == 'Degree':
            centrality = nx.degree_centrality(G)
        elif centrality_type == 'Betweenness':
            centrality = nx.betweenness_centrality(G)
        elif centrality_type == 'Closeness':
            centrality = nx.closeness_centrality(G)
        else:  # Eigenvector
            centrality = nx.eigenvector_centrality(G)
        
        top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        top_central_df = pd.DataFrame(top_central, columns=['Address', centrality_type])
        st.dataframe(top_central_df.style.format({centrality_type: '{:.4f}'}), 
                     use_container_width=True)

        # Simple network visualization
        st.write("**Network Visualization** (showing first 50 nodes for performance)")
        G_small = G.subgraph(list(G.nodes())[:50])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(G_small)
        
        # Color nodes by currency if possible
        node_colors = []
        for node in G_small.nodes():
            # Try to find the node in either sender or receiver data
            if node in sample_data['Sender_Address'].values:
                currency = sample_data[sample_data['Sender_Address'] == node]['Currency'].iloc[0]
            elif node in sample_data['Receiver_Address'].values:
                currency = sample_data[sample_data['Receiver_Address'] == node]['Currency'].iloc[0]
            else:
                currency = 'BTC'  # Default
            
            node_colors.append(BTC_COLOR if currency == 'BTC' else ETH_COLOR)
        
        nx.draw(G_small, pos, with_labels=False, node_size=50, 
               node_color=node_colors, alpha=0.8, ax=ax)
        ax.set_facecolor(DARK_BG)
        fig.patch.set_facecolor(DARK_BG)
        st.pyplot(fig)

    return top_senders, top_receivers, G

def detect_temporal_patterns(data):
    """Analyze time patterns and anomalies"""
    st.header("⏱️ Temporal Patterns Analysis")

    # Time resolution selection
    time_resolution = st.radio("Select time resolution", 
                              ['Hourly', 'Daily', 'Weekly', 'Monthly'], 
                              horizontal=True)

    # Prepare data based on resolution
    if time_resolution == 'Hourly':
        time_data = data.groupby(['Hour', 'Currency']).size().reset_index(name='Count')
        x_col = 'Hour'
    elif time_resolution == 'Daily':
        time_data = data.groupby(['Date', 'Currency']).size().reset_index(name='Count')
        x_col = 'Date'
    elif time_resolution == 'Weekly':
        data['Week'] = data['Timestamp'].dt.to_period('W').dt.start_time
        time_data = data.groupby(['Week', 'Currency']).size().reset_index(name='Count')
        x_col = 'Week'
    else:  # Monthly
        data['Month'] = data['Timestamp'].dt.to_period('M').dt.start_time
        time_data = data.groupby(['Month', 'Currency']).size().reset_index(name='Count')
        x_col = 'Month'

    # Time patterns visualization
    st.subheader(f"Transaction Volume by {time_resolution}")
    fig = px.line(time_data, x=x_col, y='Count', color='Currency',
                 color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR},
                 template='plotly_dark')
    fig.update_layout(
        xaxis_title=time_resolution,
        yaxis_title="Transaction Count",
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=DARK_BG
    )
    st.plotly_chart(fig, use_container_width=True)

    # Day of week analysis
    st.subheader("Day of Week Patterns")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_pattern = data.groupby(['Day_of_Week', 'Currency']).size().reset_index(name='Count')
    daily_pattern['Day_Index'] = daily_pattern['Day_of_Week'].apply(lambda x: day_order.index(x))
    daily_pattern = daily_pattern.sort_values('Day_Index')

    fig = px.bar(daily_pattern, x='Day_of_Week', y='Count', color='Currency',
                category_orders={'Day_of_Week': day_order},
                color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR},
                barmode='group',
                template='plotly_dark')
    fig.update_layout(
        xaxis_title="Day of Week",
        yaxis_title="Transaction Count",
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=DARK_BG
    )
    st.plotly_chart(fig, use_container_width=True)

    # Anomaly detection
    st.subheader("Anomaly Detection")
    z_threshold = st.slider("Select Z-score threshold for anomalies", 2.0, 5.0, 3.0, 0.5)
    
    data['Amount_ZScore'] = data.groupby('Currency')['Amount'].transform(
        lambda x: (x - x.mean()) / x.std())

    anomalies = data[abs(data['Amount_ZScore']) > z_threshold].copy()

    if not anomalies.empty:
        st.warning(f"⚠️ Detected {len(anomalies)} anomalies (Z-score > {z_threshold})")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Anomalies by Currency**")
            anomaly_dist = anomalies['Currency'].value_counts()
            fig = px.pie(anomaly_dist, values=anomaly_dist.values, names=anomaly_dist.index,
                        color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR},
                        template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Anomalies Over Time**")
            fig = px.scatter(anomalies, x='Timestamp', y='Amount', color='Currency',
                            size='Amount_ZScore', hover_data=['Amount_ZScore'],
                            color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR},
                            template='plotly_dark')
            fig.update_layout(
                xaxis_title="Timestamp",
                yaxis_title="Amount",
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=DARK_BG
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.write("**Anomaly Details**")
        st.dataframe(anomalies.sort_values('Amount_ZScore', ascending=False).head(10), 
                     use_container_width=True)
    else:
        st.info(f"No anomalies detected with Z-score > {z_threshold}")

    return daily_pattern, anomalies

def show_data_overview(data):
    """Display an overview of the transaction data"""
    st.header("📊 Transaction Data Overview")
    
    # Basic statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", f"{len(data):,}")
        st.metric("Total BTC Volume", f"{data[data['Currency'] == 'BTC']['Amount'].sum():.2f} BTC")
    
    with col2:
        st.metric("Date Range", f"{data['Timestamp'].min().date()} to {data['Timestamp'].max().date()}")
        st.metric("Total ETH Volume", f"{data[data['Currency'] == 'ETH']['Amount'].sum():.2f} ETH")
    
    with col3:
        st.metric("Unique Addresses", f"{len(set(data['Sender_Address']) | set(data['Receiver_Address'])):,}")
        st.metric("Average Transaction Amount", f"{data['Amount'].mean():.4f}")
    
    # Transaction currency breakdown
    st.subheader("Transaction Currency Distribution")
    currency_dist = data['Currency'].value_counts()
    fig = px.pie(currency_dist, values=currency_dist.values, names=currency_dist.index,
                color_discrete_sequence=[BTC_COLOR, ETH_COLOR],
                template='plotly_dark')
    fig.update_layout(
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=DARK_BG
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Transaction type breakdown
    st.subheader("Transaction Type & Status")
    col1, col2 = st.columns(2)
    
    with col1:
        type_dist = data['Transaction_Type'].value_counts()
        fig = px.bar(type_dist, x=type_dist.index, y=type_dist.values,
                    color=type_dist.index,
                    template='plotly_dark')
        fig.update_layout(
            xaxis_title="Transaction Type",
            yaxis_title="Count",
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=DARK_BG,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        status_dist = data['Transaction_Status'].value_counts()
        fig = px.bar(status_dist, x=status_dist.index, y=status_dist.values,
                    color=status_dist.index,
                    template='plotly_dark')
        fig.update_layout(
            xaxis_title="Transaction Status",
            yaxis_title="Count",
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=DARK_BG,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Transaction volume over time
    st.subheader("Transaction Volume Over Time")
    time_resolution = st.radio("Select time resolution for volume chart", 
                              ['Daily', 'Weekly', 'Monthly'], 
                              horizontal=True)
    
    if time_resolution == 'Daily':
        volume_data = data.groupby(['Date', 'Currency'])['Amount'].sum().reset_index()
        x_col = 'Date'
    elif time_resolution == 'Weekly':
        data['Week'] = data['Timestamp'].dt.to_period('W').dt.start_time
        volume_data = data.groupby(['Week', 'Currency'])['Amount'].sum().reset_index()
        x_col = 'Week'
    else:  # Monthly
        data['Month'] = data['Timestamp'].dt.to_period('M').dt.start_time
        volume_data = data.groupby(['Month', 'Currency'])['Amount'].sum().reset_index()
        x_col = 'Month'
    
    fig = px.area(volume_data, x=x_col, y='Amount', color='Currency',
                 color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR},
                 template='plotly_dark')
    fig.update_layout(
        xaxis_title=time_resolution,
        yaxis_title="Transaction Volume",
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=DARK_BG
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Transaction size distribution
    st.subheader("Transaction Size Distribution")
    show_outliers = st.checkbox("Show Outliers", value=False)
    
    if not show_outliers:
        # Remove outliers for better visualization
        q1 = data['Amount'].quantile(0.01)
        q3 = data['Amount'].quantile(0.99)
        filtered_data = data[(data['Amount'] >= q1) & (data['Amount'] <= q3)]
    else:
        filtered_data = data
        
    fig = px.histogram(filtered_data, x='Amount', color='Currency',
                      nbins=50, marginal='box',
                      color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR},
                      template='plotly_dark')
    fig.update_layout(
        xaxis_title="Transaction Amount",
        yaxis_title="Count",
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=DARK_BG
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Transaction size categories
    st.subheader("Transaction Size Categories")
    size_dist = data.groupby(['Size_Category', 'Currency']).size().reset_index(name='Count')
    
    fig = px.bar(size_dist, x='Size_Category', y='Count', color='Currency',
                barmode='group',
                category_orders={'Size_Category': ['<0.1', '0.1-1', '1-10', '10-100', '>100']},
                color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR},
                template='plotly_dark')
    fig.update_layout(
        xaxis_title="Transaction Size",
        yaxis_title="Count",
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=DARK_BG
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Hourly transaction patterns
    st.subheader("Hourly Transaction Patterns")
    hourly_data = data.groupby(['Hour', 'Currency']).size().reset_index(name='Count')
    
    fig = px.line(hourly_data, x='Hour', y='Count', color='Currency',
                 color_discrete_map={'BTC': BTC_COLOR, 'ETH': ETH_COLOR},
                 template='plotly_dark')
    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Transaction Count",
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=DARK_BG
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Raw data sample
    with st.expander("View Sample Data"):
        sample_size = st.slider("Select sample size", 5, 100, 20)
        st.dataframe(data.sample(sample_size), use_container_width=True)
    
    return volume_data, size_dist

def main():
    """Main function for CryptoInsight Analytics application"""
    # Logo and title
    st.sidebar.image("https://img.icons8.com/nolan/64/blockchain-technology.png", width=80)
    st.sidebar.title("CryptoInsight Analytics")
    st.sidebar.markdown("---")
    
    # Data input selection
    data_option = st.sidebar.radio(
        "Select Data Source",
        ["Demo Data", "Upload CSV"]
    )

    if data_option == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload transaction data (CSV)", type=["csv"])
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.sidebar.success("✅ File uploaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
                data = generate_demo_data()
                st.sidebar.warning("⚠️ Using demo data instead.")
        else:
            data = generate_demo_data()
            st.sidebar.info("ℹ️ Using demo data for preview.")
    else:
        data = generate_demo_data()
        st.sidebar.info("ℹ️ Using generated demo data.")

    # Preprocess data
    preprocessor = CryptoDataPreprocessor(data)
    data, btc_data, eth_data = preprocessor.get_processed_data()

    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose Analysis",
        ["Market Overview", "Transaction Overview", "Fee Analysis", 
         "Address Analysis", "Mining Pool Analysis", "Temporal Patterns"]
    )

    # Filter options
    st.sidebar.markdown("---")
    st.sidebar.title("Filters")
    
    # Date filter
    min_date = data['Timestamp'].min().date()
    max_date = data['Timestamp'].max().date()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Currency filter
    currencies = st.sidebar.multiselect(
        "Currencies",
        options=["BTC", "ETH"],
        default=["BTC", "ETH"]
    )
    
    # Status filter
    statuses = st.sidebar.multiselect(
        "Transaction Status",
        options=data['Transaction_Status'].unique(),
        default=data['Transaction_Status'].unique()
    )
    
    # Apply filters
    filtered_data = data.copy()
    if len(date_range) == 2:
        filtered_data = filtered_data[
            (filtered_data['Timestamp'].dt.date >= date_range[0]) &
            (filtered_data['Timestamp'].dt.date <= date_range[1])
        ]
    
    filtered_data = filtered_data[filtered_data['Currency'].isin(currencies)]
    filtered_data = filtered_data[filtered_data['Transaction_Status'].isin(statuses)]
    
    # Display selected page
    if page == "Market Overview":
        show_price_tab()
    
    elif page == "Transaction Overview":
        show_data_overview(filtered_data)
    
    elif page == "Fee Analysis":
        analyze_transaction_fees(filtered_data)
    
    elif page == "Address Analysis":
        analyze_address_activity(filtered_data)
    
    elif page == "Mining Pool Analysis":
        analyze_mining_pools(filtered_data)
    
    elif page == "Temporal Patterns":
        detect_temporal_patterns(filtered_data)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption(
        """
        CryptoInsight Analytics v1.0.0  
        © 2025 Crypto Analytics, Inc.
        """
    )

if __name__ == "__main__":
    main()
