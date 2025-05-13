import streamlit as st
from pages_overview import show_overview_page
from pages_price_analysis import show_price_analysis_page

# Title and introduction are in crypto_dashboard.py
# Import the base dashboard setup
from crypto_dashboard import *

# Route to the correct page based on sidebar selection
if page == "Overview":
    show_overview_page()
elif page == "Price Analysis":
    show_price_analysis_page()
elif page == "Comparison":
    st.markdown("<h2 class='subheader'>Cryptocurrency Comparison</h2>", unsafe_allow_html=True)
    st.info("This feature is coming soon! Check back later for updates.")
    
    # Placeholder for comparison feature
    st.markdown("""
    The Comparison feature will allow you to:
    - Compare multiple cryptocurrencies side by side
    - Analyze correlation between different coins
    - View comparative performance metrics
    - Generate comparison charts and reports
    """)
elif page == "Portfolio Tracker":
    st.markdown("<h2 class='subheader'>Portfolio Tracker</h2>", unsafe_allow_html=True)
    st.info("This feature is coming soon! Check back later for updates.")
    
    # Placeholder for portfolio tracker
    st.markdown("""
    The Portfolio Tracker feature will allow you to:
    - Track your cryptocurrency holdings
    - Monitor portfolio performance
    - Set price alerts
    - Analyze portfolio diversification
    - Generate performance reports
    """)

# Footer
st.markdown("<div class='footer'>", unsafe_allow_html=True)
st.markdown("Cryptocurrency Dashboard created with Streamlit • Data provided by CoinGecko API", unsafe_allow_html=True)
st.markdown("© 2023 CryptoTrack", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
