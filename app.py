import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fraud_detector import FraudDetector
from data_processor import DataProcessor
from visualizations import Visualizations
from transaction_analyzer import TransactionAnalyzer
from utils import Utils

# Configure page
st.set_page_config(
    page_title="Payment Fraud Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Initialize session state
    if 'transactions' not in st.session_state:
        st.session_state.transactions = None
    if 'fraud_detector' not in st.session_state:
        st.session_state.fraud_detector = FraudDetector()
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = DataProcessor()
    if 'visualizations' not in st.session_state:
        st.session_state.visualizations = Visualizations()
    if 'transaction_analyzer' not in st.session_state:
        st.session_state.transaction_analyzer = TransactionAnalyzer()

    # Header
    st.title("🔒 Payment Fraud Detection System")
    st.markdown("---")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Dashboard", "Transaction Analysis", "Manual Review", "Historical Data", "Settings"]
    )

    # Data upload section
    st.sidebar.markdown("### Data Management")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Transaction Data (CSV)",
        type=['csv'],
        help="Upload a CSV file containing transaction data"
    )

    # Process uploaded data
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.transactions = st.session_state.data_processor.process_transactions(df)
            st.sidebar.success(f"✅ Loaded {len(df)} transactions")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading data: {str(e)}")

    # Check if data is available
    if st.session_state.transactions is None or st.session_state.transactions.empty:
        st.warning("⚠️ No transaction data available. Please upload a CSV file with transaction data.")
        st.markdown("""
        ### Expected CSV Format:
        The CSV file should contain the following columns:
        - `transaction_id`: Unique identifier for each transaction
        - `amount`: Transaction amount (numeric)
        - `merchant`: Merchant name or ID
        - `location`: Transaction location (city, country, or coordinates)
        - `timestamp`: Transaction date and time
        - `user_id`: Customer/user identifier
        - `card_type`: Type of payment card used
        - `merchant_category`: Category of merchant
        """)
        return

    # Route to selected page
    if page == "Dashboard":
        show_dashboard()
    elif page == "Transaction Analysis":
        show_transaction_analysis()
    elif page == "Manual Review":
        show_manual_review()
    elif page == "Historical Data":
        show_historical_data()
    elif page == "Settings":
        show_settings()

def show_dashboard():
    st.header("📊 Fraud Detection Dashboard")
    
    transactions = st.session_state.transactions
    fraud_detector = st.session_state.fraud_detector
    visualizations = st.session_state.visualizations
    
    # Run fraud detection
    transactions_with_scores = fraud_detector.detect_fraud(transactions)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_transactions = len(transactions_with_scores)
    flagged_transactions = len(transactions_with_scores[transactions_with_scores['is_fraud'] == True])
    high_risk_transactions = len(transactions_with_scores[transactions_with_scores['risk_score'] > 0.7])
    total_amount = transactions_with_scores['amount'].sum()
    
    with col1:
        st.metric("Total Transactions", f"{total_transactions:,}")
    with col2:
        st.metric("Flagged as Fraud", f"{flagged_transactions:,}")
    with col3:
        st.metric("High Risk", f"{high_risk_transactions:,}")
    with col4:
        st.metric("Total Amount", f"${total_amount:,.2f}")
    
    # Risk distribution
    st.subheader("Risk Score Distribution")
    fig_risk = visualizations.create_risk_distribution(transactions_with_scores)
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # Time series analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transactions Over Time")
        fig_time = visualizations.create_time_series(transactions_with_scores)
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        st.subheader("Fraud by Location")
        fig_location = visualizations.create_location_analysis(transactions_with_scores)
        st.plotly_chart(fig_location, use_container_width=True)
    
    # Recent high-risk transactions
    st.subheader("⚠️ Recent High-Risk Transactions")
    high_risk = transactions_with_scores[
        transactions_with_scores['risk_score'] > 0.6
    ].sort_values('timestamp', ascending=False).head(10)
    
    if not high_risk.empty:
        st.dataframe(
            high_risk[['transaction_id', 'amount', 'merchant', 'location', 'risk_score', 'fraud_reasons']],
            use_container_width=True
        )
    else:
        st.info("No high-risk transactions detected recently.")

def show_transaction_analysis():
    st.header("🔍 Transaction Analysis")
    
    transactions = st.session_state.transactions
    visualizations = st.session_state.visualizations
    
    # Filters
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        amount_range = st.slider(
            "Amount Range",
            min_value=float(transactions['amount'].min()),
            max_value=float(transactions['amount'].max()),
            value=(float(transactions['amount'].min()), float(transactions['amount'].max()))
        )
    
    with col2:
        locations = st.multiselect(
            "Locations",
            options=transactions['location'].unique(),
            default=list(transactions['location'].unique())[:5]
        )
    
    with col3:
        date_range = st.date_input(
            "Date Range",
            value=(transactions['timestamp'].min().date(), transactions['timestamp'].max().date()),
            min_value=transactions['timestamp'].min().date(),
            max_value=transactions['timestamp'].max().date()
        )
    
    # Apply filters
    filtered_transactions = transactions[
        (transactions['amount'] >= amount_range[0]) &
        (transactions['amount'] <= amount_range[1]) &
        (transactions['location'].isin(locations)) &
        (transactions['timestamp'].dt.date >= date_range[0]) &
        (transactions['timestamp'].dt.date <= date_range[1])
    ]
    
    st.write(f"Showing {len(filtered_transactions)} transactions")
    
    # Analysis charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Amount Distribution")
        fig_amount = visualizations.create_amount_distribution(filtered_transactions)
        st.plotly_chart(fig_amount, use_container_width=True)
    
    with col2:
        st.subheader("Merchant Category Analysis")
        fig_merchant = visualizations.create_merchant_analysis(filtered_transactions)
        st.plotly_chart(fig_merchant, use_container_width=True)
    
    # Hourly patterns
    st.subheader("Transaction Patterns by Hour")
    fig_hourly = visualizations.create_hourly_patterns(filtered_transactions)
    st.plotly_chart(fig_hourly, use_container_width=True)

def show_manual_review():
    st.header("👤 Manual Transaction Review")
    
    transactions = st.session_state.transactions
    fraud_detector = st.session_state.fraud_detector
    transaction_analyzer = st.session_state.transaction_analyzer
    
    # Transaction lookup
    st.subheader("Transaction Lookup")
    transaction_id = st.text_input("Enter Transaction ID for review:")
    
    if transaction_id:
        transaction = transactions[transactions['transaction_id'] == transaction_id]
        
        if not transaction.empty:
            tx = transaction.iloc[0]
            
            # Display transaction details
            st.subheader("Transaction Details")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Amount:** ${tx['amount']:,.2f}")
                st.write(f"**Merchant:** {tx['merchant']}")
                st.write(f"**Location:** {tx['location']}")
            
            with col2:
                st.write(f"**Time:** {tx['timestamp']}")
                st.write(f"**User ID:** {tx['user_id']}")
                st.write(f"**Card Type:** {tx['card_type']}")
            
            with col3:
                st.write(f"**Merchant Category:** {tx['merchant_category']}")
            
            # Calculate risk score
            risk_analysis = transaction_analyzer.analyze_single_transaction(tx, transactions)
            
            # Display risk analysis
            st.subheader("Risk Analysis")
            risk_score = risk_analysis['risk_score']
            
            if risk_score > 0.7:
                st.error(f"🚨 HIGH RISK: {risk_score:.2%}")
            elif risk_score > 0.4:
                st.warning(f"⚠️ MEDIUM RISK: {risk_score:.2%}")
            else:
                st.success(f"✅ LOW RISK: {risk_score:.2%}")
            
            # Risk factors
            if risk_analysis['risk_factors']:
                st.subheader("Risk Factors Identified:")
                for factor in risk_analysis['risk_factors']:
                    st.write(f"• {factor}")
            
            # Manual decision
            st.subheader("Manual Decision")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("✅ Approve Transaction", type="primary"):
                    st.success("Transaction approved and whitelisted")
            
            with col2:
                if st.button("❌ Block Transaction", type="secondary"):
                    st.error("Transaction blocked and flagged as fraud")
            
            with col3:
                if st.button("⏸️ Hold for Review", type="secondary"):
                    st.warning("Transaction placed on hold for further review")
            
            # Historical patterns for this user
            st.subheader("User Transaction History")
            user_history = transactions[transactions['user_id'] == tx['user_id']].sort_values('timestamp', ascending=False)
            
            if len(user_history) > 1:
                st.dataframe(
                    user_history[['transaction_id', 'amount', 'merchant', 'location', 'timestamp']].head(10),
                    use_container_width=True
                )
            else:
                st.info("No previous transactions found for this user.")
        
        else:
            st.error("Transaction ID not found.")
    
    # Bulk review section
    st.markdown("---")
    st.subheader("Bulk Review Queue")
    
    # Get flagged transactions
    flagged_transactions = fraud_detector.detect_fraud(transactions)
    high_risk_queue = flagged_transactions[
        flagged_transactions['risk_score'] > 0.6
    ].sort_values('risk_score', ascending=False).head(20)
    
    if not high_risk_queue.empty:
        st.write(f"**{len(high_risk_queue)} transactions requiring review**")
        
        for idx, (_, tx) in enumerate(high_risk_queue.iterrows()):
            with st.expander(f"Transaction {tx['transaction_id']} - Risk: {tx['risk_score']:.2%}"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.write(f"Amount: ${tx['amount']:,.2f}")
                    st.write(f"Location: {tx['location']}")
                
                with col2:
                    st.write(f"Merchant: {tx['merchant']}")
                    st.write(f"Time: {tx['timestamp']}")
                
                with col3:
                    st.write(f"Risk Factors:")
                    if pd.notna(tx['fraud_reasons']):
                        for reason in tx['fraud_reasons'].split(', '):
                            st.write(f"• {reason}")
                
                with col4:
                    st.button(f"Review {tx['transaction_id']}", key=f"review_{idx}")
    else:
        st.info("No transactions currently in review queue.")

def show_historical_data():
    st.header("📈 Historical Data Analysis")
    
    transactions = st.session_state.transactions
    visualizations = st.session_state.visualizations
    
    # Time period selector
    period = st.selectbox(
        "Select Analysis Period",
        ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"]
    )
    
    # Filter data based on period
    end_date = transactions['timestamp'].max()
    if period == "Last 7 Days":
        start_date = end_date - timedelta(days=7)
    elif period == "Last 30 Days":
        start_date = end_date - timedelta(days=30)
    elif period == "Last 90 Days":
        start_date = end_date - timedelta(days=90)
    else:
        start_date = transactions['timestamp'].min()
    
    filtered_data = transactions[
        (transactions['timestamp'] >= start_date) &
        (transactions['timestamp'] <= end_date)
    ]
    
    # Summary statistics
    st.subheader("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", f"{len(filtered_data):,}")
    with col2:
        st.metric("Total Volume", f"${filtered_data['amount'].sum():,.2f}")
    with col3:
        st.metric("Average Amount", f"${filtered_data['amount'].mean():.2f}")
    with col4:
        st.metric("Unique Merchants", f"{filtered_data['merchant'].nunique():,}")
    
    # Trend analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction Volume Trend")
        fig_volume = visualizations.create_volume_trend(filtered_data)
        st.plotly_chart(fig_volume, use_container_width=True)
    
    with col2:
        st.subheader("Average Amount Trend")
        fig_amount_trend = visualizations.create_amount_trend(filtered_data)
        st.plotly_chart(fig_amount_trend, use_container_width=True)
    
    # Geographic analysis
    st.subheader("Geographic Distribution")
    fig_geo = visualizations.create_geographic_analysis(filtered_data)
    st.plotly_chart(fig_geo, use_container_width=True)
    
    # Export functionality
    st.subheader("Export Data")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export to CSV"):
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"transactions_{period.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📈 Export Summary Report"):
            summary_report = Utils.generate_summary_report(filtered_data)
            st.download_button(
                label="Download Report",
                data=summary_report,
                file_name=f"fraud_analysis_report_{period.lower().replace(' ', '_')}.txt",
                mime="text/plain"
            )

def show_settings():
    st.header("⚙️ Settings")
    
    st.subheader("Fraud Detection Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Thresholds")
        high_risk_threshold = st.slider("High Risk Threshold", 0.0, 1.0, 0.7, 0.05)
        medium_risk_threshold = st.slider("Medium Risk Threshold", 0.0, 1.0, 0.4, 0.05)
        
        st.subheader("Amount Thresholds")
        unusual_amount_threshold = st.number_input("Unusual Amount Threshold ($)", value=10000.0)
        micro_transaction_threshold = st.number_input("Micro Transaction Threshold ($)", value=1.0)
    
    with col2:
        st.subheader("Time-based Rules")
        off_hours_start = st.time_input("Off Hours Start", value=datetime.strptime("22:00", "%H:%M").time())
        off_hours_end = st.time_input("Off Hours End", value=datetime.strptime("06:00", "%H:%M").time())
        
        st.subheader("Velocity Rules")
        max_transactions_per_hour = st.number_input("Max Transactions per Hour", value=10, min_value=1)
        max_amount_per_day = st.number_input("Max Amount per Day ($)", value=50000.0)
    
    if st.button("💾 Save Settings"):
        # Update fraud detector settings
        settings = {
            'high_risk_threshold': high_risk_threshold,
            'medium_risk_threshold': medium_risk_threshold,
            'unusual_amount_threshold': unusual_amount_threshold,
            'micro_transaction_threshold': micro_transaction_threshold,
            'off_hours_start': off_hours_start,
            'off_hours_end': off_hours_end,
            'max_transactions_per_hour': max_transactions_per_hour,
            'max_amount_per_day': max_amount_per_day
        }
        
        st.session_state.fraud_detector.update_settings(settings)
        st.success("✅ Settings saved successfully!")
    
    # System information
    st.markdown("---")
    st.subheader("System Information")
    st.write(f"**Application Version:** 1.0.0")
    st.write(f"**Last Data Update:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if st.session_state.transactions is not None:
        st.write(f"**Total Transactions Loaded:** {len(st.session_state.transactions):,}")
        st.write(f"**Data Date Range:** {st.session_state.transactions['timestamp'].min()} to {st.session_state.transactions['timestamp'].max()}")

if __name__ == "__main__":
    main()
