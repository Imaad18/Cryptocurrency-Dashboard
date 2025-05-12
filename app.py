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
        self.processed_data['DayOfWeek'] = self.processed_data['Timestamp'].dt.day_name()
        
        # Normalize transaction amount
        scaler = MinMaxScaler()
        self.processed_data['Normalized_Amount'] = scaler.fit_transform(
            self.processed_data['Amount'].values.reshape(-1, 1)
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
        )
        
        # Risk categorization
        def categorize_risk(score):
            if score < 10: return 'Low Risk'
            elif score < 30: return 'Medium Risk'
            else: return 'High Risk'
        
        self.processed_data['Risk_Category'] = self.processed_data['Risk_Score'].apply(categorize_risk)
        
        # Visualization
        risk_dist = self.processed_data.groupby(['Currency', 'Risk_Category']).size().unstack(fill_value=0)
        
        fig = px.bar(
            x=risk_dist.index, 
            y=risk_dist.values, 
            title='Risk Distribution by Cryptocurrency',
            labels={'x': 'Currency', 'y': 'Number of Transactions'},
            color_discrete_sequence=px.colors.qualitative.Pastel
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
        type=["csv"], 
        help="Upload a CSV file with cryptocurrency transaction data"
    )
    
    # Add some sample data options
    if st.sidebar.checkbox("Use Sample Data"):
        # Create a sample dataset
        sample_data = pd.DataFrame({
            'Timestamp': pd.date_range(start='2023-01-01', periods=1000),
            'Currency': np.random.choice(['BTC', 'ETH'], 1000),
            'Amount': np.random.uniform(10, 10000, 1000),
            'Transaction_Fee': np.random.uniform(0.1, 100, 1000),
            'Sender_Address': [f'0x{np.random.randint(0, 2**32):08x}' for _ in range(1000)],
            'Receiver_Address': [f'0x{np.random.randint(0, 2**32):08x}' for _ in range(1000)],
            'Transaction_Status': np.random.choice(['Success', 'Failed'], 1000)
        })
        uploaded_file = io.BytesIO(sample_data.to_csv(index=False).encode())
        st.sidebar.success("Sample data generated!")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            raw_data = pd.read_csv(uploaded_file)
            st.sidebar.success("Data loaded successfully!")
            
            # Initialize advanced analyzer
            analyzer = AdvancedCryptoAnalyzer(raw_data)
            processed_data = analyzer.preprocess_data()
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4 = st.tabs([
                "🔍 Data Overview", 
                "🌐 Clustering Analysis", 
                "🎲 Risk Assessment", 
                "📊 Advanced Insights"
            ])
            
            with tab1:
                st.header("Data Overview")
                st.write(processed_data.describe())
                
                # Transaction distribution
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Transactions by Currency")
                    currency_dist = processed_data['Currency'].value_counts()
                    fig = px.pie(
                        values=currency_dist.values, 
                        names=currency_dist.index, 
                        title='Currency Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Transactions by Day of Week")
                    day_dist = processed_data['DayOfWeek'].value_counts()
                    fig = px.bar(
                        x=day_dist.index, 
                        y=day_dist.values, 
                        title='Transactions by Day'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                analyzer.advanced_clustering()
            
            with tab3:
                analyzer.predictive_risk_scoring()
            
            with tab4:
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
                    title='Feature Correlation Heatmap'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing data: {e}")
    else:
        st.info("📤 Please upload a CSV file or use sample data to begin analysis")

if __name__ == "__main__":
    main()
