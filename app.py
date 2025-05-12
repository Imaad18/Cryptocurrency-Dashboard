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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import networkx as nx
import requests
import json
import warnings
import io
import base64
from faker import Faker  # For generating realistic sample data

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('dark_background')
sns.set_style('darkgrid')
pd.set_option('display.max_columns', None)

# Set page config
st.set_page_config(
    page_title="CryptoInsight Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1E2130;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #FFFFFF;
        background-color: transparent;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #2C3040;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #4A5568;
        color: #64B5F6;
    }
    .metric-card {
        background-color: #1E2130;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: scale(1.05);
    }
    .stMarkdown {
        color: #E2E8F0;
    }
    .stDataFrame {
        background-color: #1E2130 !important;
    }
    .stAlert {
        background-color: #2C3040 !important;
    }
    </style>
    """, unsafe_allow_html=True)

class AdvancedCryptoAnalyzer:
    def __init__(self, data):
        self.original_data = data.copy()
        self.processed_data = None
        self.btc_data = None
        self.eth_data = None
        
    def preprocess_data(self):
    """Advanced data preprocessing with feature engineering"""
    st.write("🔍 Performing advanced data preprocessing...")
    
    # Convert timestamp
    self.processed_data = self.original_data.copy()
    self.processed_data['Timestamp'] = pd.to_datetime(self.processed_data['Timestamp'])
    
    # Feature engineering
    self.processed_data['Year'] = self.processed_data['Timestamp'].dt.year
    self.processed_data['Month'] = self.processed_data['Timestamp'].dt.month
    self.processed_data['Day'] = self.processed_data['Timestamp'].dt.day
    self.processed_data['Hour'] = self.processed_data['Timestamp'].dt.hour
    self.processed_data['DayOfWeek'] = self.processed_data['Timestamp'].dt.day_name()
    self.processed_data['DayOfWeekNum'] = self.processed_data['Timestamp'].dt.dayofweek
    
    # Normalize transaction amount
    scaler = MinMaxScaler()
    self.processed_data['Normalized_Amount'] = scaler.fit_transform(
        self.processed_data['Amount'].values.reshape(-1, 1)  # Fixed this line
    )
    
    # Sentiment proxy (based on transaction size and frequency)
    self.processed_data['Transaction_Sentiment'] = np.where(
        self.processed_data['Normalized_Amount'] > 0.7, 'High Confidence',
        np.where(self.processed_data['Normalized_Amount'] > 0.3, 'Moderate Confidence', 'Low Confidence')
    )
    
    # Separate BTC and ETH data
    self.btc_data = self.processed_data[self.processed_data['Currency'] == 'BTC']
    self.eth_data = self.processed_data[self.processed_data['Currency'] == 'ETH']
    
    st.success("✅ Advanced preprocessing completed!")
    return self.processed_data
    
    def advanced_clustering(self):
        """Perform advanced clustering analysis"""
        st.header("🌐 Advanced Transaction Clustering")
        
        # Prepare features for clustering
        features = ['Amount', 'Transaction_Fee']
        X = self.processed_data[features]
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Dimensionality reduction
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        self.processed_data['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Visualization
        fig = px.scatter(
            x=X_pca[:, 0], 
            y=X_pca[:, 1], 
            color=self.processed_data['Cluster'].astype(str),
            color_discrete_sequence=px.colors.qualitative.Pastel,
            title='Transaction Clusters (PCA Visualization)',
            labels={'x': 'First Principal Component', 'y': 'Second Principal Component'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster characteristics
        cluster_summary = self.processed_data.groupby('Cluster').agg({
            'Amount': ['mean', 'median'],
            'Transaction_Fee': ['mean', 'median'],
            'Transaction_ID': 'count'
        }).reset_index()
        cluster_summary.columns = ['Cluster', 'Avg Amount', 'Median Amount', 'Avg Fee', 'Median Fee', 'Transaction Count']
        
        st.subheader("Cluster Characteristics")
        st.dataframe(cluster_summary)
        
        return cluster_summary
    
    def predictive_risk_scoring(self):
        """Create a basic predictive risk scoring mechanism"""
        st.header("🎲 Transaction Risk Scoring")
        
        # Risk factors
        self.processed_data['Risk_Score'] = (
            (self.processed_data['Transaction_Fee'] / self.processed_data['Amount'] * 100) +  # Fee ratio
            (self.processed_data['Normalized_Amount'] * 100) +  # Transaction size
            (np.where(self.processed_data['Transaction_Status'] == 'Failed', 50, 0))  # Failed transaction penalty
        
        # Risk categorization
        def categorize_risk(score):
            if score < 10: return 'Low Risk'
            elif score < 30: return 'Medium Risk'
            else: return 'High Risk'
        
        self.processed_data['Risk_Category'] = self.processed_data['Risk_Score'].apply(categorize_risk)
        
        # Visualization
        risk_dist = self.processed_data['Risk_Category'].value_counts().reset_index()
        risk_dist.columns = ['Risk_Category', 'Count']
        
        fig = px.bar(
            risk_dist,
            x='Risk_Category', 
            y='Count', 
            title='Risk Distribution',
            labels={'x': 'Risk Category', 'y': 'Number of Transactions'},
            color='Risk_Category',
            color_discrete_map={
                'Low Risk': '#2ecc71',
                'Medium Risk': '#f39c12',
                'High Risk': '#e74c3c'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed risk insights
        st.subheader("Risk Category Insights")
        risk_insights = self.processed_data.groupby(['Currency', 'Risk_Category']).agg({
            'Amount': ['mean', 'median'],
            'Transaction_Fee': ['mean', 'median'],
            'Transaction_ID': 'count'
        }).reset_index()
        risk_insights.columns = ['Currency', 'Risk_Category', 'Avg Amount', 'Median Amount', 'Avg Fee', 'Median Fee', 'Transaction Count']
        st.dataframe(risk_insights)
        
        return self.processed_data, risk_insights
    
    def transaction_network_analysis(self):
        """Analyze transaction network patterns"""
        st.header("🕸️ Transaction Network Analysis")
        
        # Create a sample network (in a real app, you'd use actual sender/receiver data)
        sample_size = min(100, len(self.processed_data))
        sample_data = self.processed_data.sample(sample_size, random_state=42)
        
        G = nx.Graph()
        
        # Add nodes and edges
        for _, row in sample_data.iterrows():
            G.add_node(row['Sender_Address'], type='sender')
            G.add_node(row['Receiver_Address'], type='receiver')
            G.add_edge(row['Sender_Address'], row['Receiver_Address'], 
                      amount=row['Amount'], currency=row['Currency'])
        
        # Calculate network metrics
        degrees = dict(G.degree())
        betweenness = nx.betweenness_centrality(G)
        
        # Create network visualization
        pos = nx.spring_layout(G, seed=42)
        
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
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"Address: {node[:10]}...<br>Degree: {degrees.get(node, 0)}")
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))
        
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        title='Transaction Network Visualization',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Network statistics
        st.subheader("Network Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Number of Nodes", G.number_of_nodes())
        col2.metric("Number of Edges", G.number_of_edges())
        col3.metric("Average Degree", f"{sum(degrees.values())/len(degrees):.2f}")
        
        return G

def get_crypto_markets():
    """Enhanced cryptocurrency market data fetcher"""
    try:
        # Top 10 cryptocurrencies
        top_crypto_url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=10"
        response = requests.get(top_crypto_url)
        top_cryptos = response.json()
        
        return top_cryptos
    except Exception as e:
        st.error(f"Market data fetch error: {e}")
        return None

def display_market_overview(top_cryptos):
    """Display comprehensive market overview"""
    st.header("📈 Global Cryptocurrency Market Overview")
    
    if not top_cryptos:
        st.warning("Unable to fetch market data")
        return
    
    # Create columns for top cryptocurrencies
    columns = st.columns(5)
    
    for i, crypto in enumerate(top_cryptos[:5]):
        with columns[i]:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{crypto['symbol'].upper()}</h4>
                <p>Price: ${crypto['current_price']:,.2f}</p>
                <p>24h Change: 
                    <span style="color:{'green' if crypto['price_change_percentage_24h'] >= 0 else 'red'}">
                        {crypto['price_change_percentage_24h']:.2f}%
                    </span>
                </p>
                <p>Market Cap: ${crypto['market_cap']:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)

def generate_sample_data():
    """Generate realistic sample cryptocurrency transaction data"""
    fake = Faker()
    
    # Generate realistic timestamps
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = (end_date - start_date).days
    random_dates = [start_date + timedelta(days=np.random.randint(date_range), 
                                         hours=np.random.randint(24),
                  minutes=np.random.randint(60)) for _ in range(1000)]
    
    # Generate realistic amounts with different distributions for BTC and ETH
    btc_amounts = np.random.lognormal(mean=2, sigma=1.5, size=500)
    eth_amounts = np.random.lognormal(mean=1.5, sigma=1.2, size=500)
    
    sample_data = pd.DataFrame({
        'Transaction_ID': [f'TX{fake.unique.random_number(digits=8)}' for _ in range(1000)],
        'Timestamp': sorted(random_dates),
        'Currency': np.random.choice(['BTC', 'ETH'], 1000, p=[0.6, 0.4]),
        'Amount': np.concatenate([btc_amounts, eth_amounts]),
        'Transaction_Fee': np.random.uniform(0.001, 0.1, 1000) * np.concatenate([btc_amounts, eth_amounts]),
        'Sender_Address': [f'0x{fake.unique.sha256()[:40]}' for _ in range(1000)],
        'Receiver_Address': [f'0x{fake.unique.sha256()[:40]}' for _ in range(1000)],
        'Transaction_Status': np.random.choice(['Success', 'Failed'], 1000, p=[0.95, 0.05])
    })
    
    # Make some transactions between the same addresses
    for _ in range(50):
        idx = np.random.randint(0, 1000)
        sample_data.at[idx, 'Receiver_Address'] = sample_data.at[idx, 'Sender_Address']
    
    return sample_data

def display_transaction_trends(processed_data):
    """Display transaction trends by hour and day"""
    st.subheader("Transaction Trends by Time")
    
    # Create tabs for different time aggregations
    tab1, tab2 = st.tabs(["By Hour", "By Day of Week"])
    
    with tab1:
        # Transactions by hour
        hourly_data = processed_data.groupby(['Hour', 'Currency']).size().reset_index(name='Count')
        fig = px.line(
            hourly_data, 
            x='Hour', 
            y='Count', 
            color='Currency',
            title='Transaction Volume by Hour of Day',
            labels={'Hour': 'Hour of Day', 'Count': 'Transaction Count'}
        )
        fig.update_xaxes(tickvals=list(range(24)))
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Transactions by day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_data = processed_data.groupby(['DayOfWeek', 'Currency']).size().reset_index(name='Count')
        daily_data['DayOfWeek'] = pd.Categorical(daily_data['DayOfWeek'], categories=day_order, ordered=True)
        daily_data = daily_data.sort_values('DayOfWeek')
        
        fig = px.line(
            daily_data, 
            x='DayOfWeek', 
            y='Count', 
            color='Currency',
            title='Transaction Volume by Day of Week',
            labels={'DayOfWeek': 'Day of Week', 'Count': 'Transaction Count'}
        )
        st.plotly_chart(fig, use_container_width=True)

def display_whale_alert(processed_data):
    """Identify and display large transactions (whale alerts)"""
    st.header("🐋 Whale Alert: Large Transactions")
    
    # Define whale transactions (top 1% by amount)
    threshold = processed_data['Amount'].quantile(0.99)
    whale_txs = processed_data[processed_data['Amount'] >= threshold].sort_values('Amount', ascending=False)
    
    if len(whale_txs) > 0:
        st.warning(f"🚨 Detected {len(whale_txs)} large transactions (top 1%)")
        
        # Display top whale transactions
        cols = st.columns(4)
        cols[0].metric("Threshold Amount", f"{threshold:,.2f}")
        cols[1].metric("Largest Transaction", f"{whale_txs['Amount'].max():,.2f}")
        cols[2].metric("Average Whale TX", f"{whale_txs['Amount'].mean():,.2f}")
        cols[3].metric("Total Whale Volume", f"{whale_txs['Amount'].sum():,.2f}")
        
        # Show detailed table
        st.dataframe(whale_txs[['Timestamp', 'Currency', 'Amount', 'Sender_Address', 'Receiver_Address']])
        
        # Visualization of whale transactions over time
        fig = px.scatter(
            whale_txs,
            x='Timestamp',
            y='Amount',
            color='Currency',
            size='Amount',
            hover_data=['Sender_Address', 'Receiver_Address'],
            title='Large Transactions Over Time'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No whale transactions detected in this dataset")

def main():
    """Enhanced main function for CryptoInsight Pro"""
    st.title("🚀 CryptoInsight Pro: Advanced Transaction Analytics")
    
    # Sidebar for file upload and market overview
    st.sidebar.image('https://via.placeholder.com/150?text=CryptoInsight+Pro', width=200)
    st.sidebar.header("Market Overview")
    top_cryptos = get_crypto_markets()
    if top_cryptos:
        display_market_overview(top_cryptos)
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Cryptocurrency Transaction Data", 
        type=["csv", "xlsx"], 
        help="Upload a CSV or Excel file with cryptocurrency transaction data"
    )
    
    # Add some sample data options
    if st.sidebar.checkbox("Use Sample Data", help="Generate realistic sample data for demonstration"):
        sample_data = generate_sample_data()
        uploaded_file = io.BytesIO(sample_data.to_csv(index=False).encode())
        st.sidebar.success("Realistic sample data generated!")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            if uploaded_file.name.endswith('.csv'):
                raw_data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                raw_data = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
                return
                
            st.sidebar.success("Data loaded successfully!")
            
            # Initialize advanced analyzer
            analyzer = AdvancedCryptoAnalyzer(raw_data)
            processed_data = analyzer.preprocess_data()
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🔍 Data Overview", 
                "🌐 Clustering Analysis", 
                "🎲 Risk Assessment", 
                "🕸️ Network Analysis",
                "📊 Advanced Insights"
            ])
            
            with tab1:
                st.header("Data Overview")
                
                # Quick stats
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Transactions", len(processed_data))
                col2.metric("Total Volume", f"{processed_data['Amount'].sum():,.2f}")
                col3.metric("Unique Addresses", 
                           f"{len(set(processed_data['Sender_Address'].unique()) | set(processed_data['Receiver_Address'].unique()))}")
                
                st.write("Processed Data Sample:")
                st.dataframe(processed_data.head())
                
                # Transaction distribution
                display_transaction_trends(processed_data)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Transactions by Currency")
                    currency_dist = processed_data['Currency'].value_counts()
                    fig = px.pie(
                        values=currency_dist.values, 
                        names=currency_dist.index, 
                        title='Currency Distribution',
                        hole=0.3
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Transaction Status")
                    status_dist = processed_data['Transaction_Status'].value_counts()
                    fig = px.bar(
                        x=status_dist.index, 
                        y=status_dist.values, 
                        title='Transaction Status Distribution',
                        color=status_dist.index,
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                analyzer.advanced_clustering()
            
            with tab3:
                analyzer.predictive_risk_scoring()
            
            with tab4:
                analyzer.transaction_network_analysis()
            
            with tab5:
                st.header("📊 Advanced Transaction Insights")
                
                # Advanced time series analysis
                st.subheader("Transaction Volume Over Time")
                time_series = processed_data.groupby([processed_data['Timestamp'].dt.date, 'Currency'])['Amount'].sum().reset_index()
                fig = px.line(
                    time_series, 
                    x='Timestamp', 
                    y='Amount', 
                    color='Currency',
                    title='Daily Transaction Volume'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation heatmap
                st.subheader("Feature Correlation")
                numeric_cols = ['Amount', 'Transaction_Fee', 'Normalized_Amount']
                corr_matrix = processed_data[numeric_cols].corr()
                fig = px.imshow(
                    corr_matrix, 
                    text_auto=True, 
                    title='Feature Correlation Heatmap',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Whale alert section
                display_whale_alert(processed_data)
                
                # Address activity analysis
                st.subheader("Address Activity Analysis")
                top_senders = processed_data['Sender_Address'].value_counts().head(10).reset_index()
                top_senders.columns = ['Address', 'Count']
                
                fig = px.bar(
                    top_senders,
                    x='Address',
                    y='Count',
                    title='Most Active Sender Addresses',
                    color='Count',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
    else:
        st.info("📤 Please upload a CSV/Excel file or use sample data to begin analysis")

if __name__ == "__main__":
    main()
