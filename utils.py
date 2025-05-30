import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import csv
from io import StringIO

class Utils:
    @staticmethod
    def generate_summary_report(transactions_df: pd.DataFrame) -> str:
        """Generate a comprehensive summary report"""
        report = StringIO()
        
        # Header
        report.write("FRAUD DETECTION ANALYSIS REPORT\n")
        report.write("=" * 50 + "\n")
        report.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Basic Statistics
        report.write("BASIC STATISTICS\n")
        report.write("-" * 20 + "\n")
        report.write(f"Total Transactions: {len(transactions_df):,}\n")
        report.write(f"Date Range: {transactions_df['timestamp'].min()} to {transactions_df['timestamp'].max()}\n")
        report.write(f"Total Amount: ${transactions_df['amount'].sum():,.2f}\n")
        report.write(f"Average Amount: ${transactions_df['amount'].mean():.2f}\n")
        report.write(f"Median Amount: ${transactions_df['amount'].median():.2f}\n")
        report.write(f"Unique Users: {transactions_df['user_id'].nunique():,}\n")
        report.write(f"Unique Merchants: {transactions_df['merchant'].nunique():,}\n")
        report.write(f"Unique Locations: {transactions_df['location'].nunique():,}\n\n")
        
        # Fraud Analysis (if fraud detection has been run)
        if 'risk_score' in transactions_df.columns:
            fraud_transactions = transactions_df[transactions_df.get('is_fraud', False)]
            high_risk = transactions_df[transactions_df['risk_score'] > 0.7]
            medium_risk = transactions_df[
                (transactions_df['risk_score'] > 0.4) & 
                (transactions_df['risk_score'] <= 0.7)
            ]
            
            report.write("FRAUD ANALYSIS\n")
            report.write("-" * 15 + "\n")
            report.write(f"High Risk Transactions: {len(high_risk):,} ({len(high_risk)/len(transactions_df)*100:.1f}%)\n")
            report.write(f"Medium Risk Transactions: {len(medium_risk):,} ({len(medium_risk)/len(transactions_df)*100:.1f}%)\n")
            report.write(f"Flagged as Fraud: {len(fraud_transactions):,} ({len(fraud_transactions)/len(transactions_df)*100:.1f}%)\n")
            report.write(f"Average Risk Score: {transactions_df['risk_score'].mean():.3f}\n")
            
            if len(fraud_transactions) > 0:
                report.write(f"Fraudulent Amount: ${fraud_transactions['amount'].sum():,.2f}\n")
                report.write(f"Fraud Amount Rate: {fraud_transactions['amount'].sum()/transactions_df['amount'].sum()*100:.1f}%\n")
            report.write("\n")
        
        # Top Merchants by Transaction Count
        report.write("TOP MERCHANTS BY TRANSACTION COUNT\n")
        report.write("-" * 35 + "\n")
        top_merchants = transactions_df['merchant'].value_counts().head(10)
        for merchant, count in top_merchants.items():
            report.write(f"{merchant}: {count:,} transactions\n")
        report.write("\n")
        
        # Top Locations by Transaction Count
        report.write("TOP LOCATIONS BY TRANSACTION COUNT\n")
        report.write("-" * 35 + "\n")
        top_locations = transactions_df['location'].value_counts().head(10)
        for location, count in top_locations.items():
            report.write(f"{location}: {count:,} transactions\n")
        report.write("\n")
        
        # Transaction Distribution by Hour
        report.write("TRANSACTION DISTRIBUTION BY HOUR\n")
        report.write("-" * 33 + "\n")
        hourly_dist = transactions_df.groupby(transactions_df['timestamp'].dt.hour).size()
        for hour, count in hourly_dist.items():
            report.write(f"{hour:02d}:00 - {count:,} transactions\n")
        report.write("\n")
        
        # Amount Distribution
        report.write("AMOUNT DISTRIBUTION\n")
        report.write("-" * 19 + "\n")
        amount_ranges = [
            (0, 10, "$0-$10"),
            (10, 50, "$10-$50"),
            (50, 100, "$50-$100"),
            (100, 500, "$100-$500"),
            (500, 1000, "$500-$1,000"),
            (1000, 5000, "$1,000-$5,000"),
            (5000, float('inf'), "$5,000+")
        ]
        
        for min_amt, max_amt, label in amount_ranges:
            if max_amt == float('inf'):
                count = len(transactions_df[transactions_df['amount'] >= min_amt])
            else:
                count = len(transactions_df[
                    (transactions_df['amount'] >= min_amt) & 
                    (transactions_df['amount'] < max_amt)
                ])
            percentage = count / len(transactions_df) * 100
            report.write(f"{label}: {count:,} transactions ({percentage:.1f}%)\n")
        
        return report.getvalue()
    
    @staticmethod
    def export_risk_analysis(transactions_df: pd.DataFrame, filename: Optional[str] = None) -> str:
        """Export detailed risk analysis to CSV format"""
        if 'risk_score' not in transactions_df.columns:
            raise ValueError("Risk analysis not available. Run fraud detection first.")
        
        # Select relevant columns for export
        export_columns = [
            'transaction_id', 'timestamp', 'amount', 'merchant', 'location',
            'user_id', 'card_type', 'merchant_category', 'risk_score', 'is_fraud', 'fraud_reasons'
        ]
        
        available_columns = [col for col in export_columns if col in transactions_df.columns]
        export_df = transactions_df[available_columns].copy()
        
        # Sort by risk score descending
        export_df = export_df.sort_values('risk_score', ascending=False)
        
        return export_df.to_csv(index=False)
    
    @staticmethod
    def validate_transaction_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate transaction data and return validation results"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        required_columns = ['transaction_id', 'amount', 'merchant', 'location', 'timestamp', 'user_id']
        
        # Check for missing required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
        
        # Check for empty dataframe
        if len(df) == 0:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Dataset is empty")
            return validation_results
        
        # Check data types and values
        if 'amount' in df.columns:
            # Check for negative amounts
            negative_amounts = df['amount'] < 0
            if negative_amounts.any():
                validation_results['warnings'].append(f"{negative_amounts.sum()} transactions have negative amounts")
            
            # Check for zero amounts
            zero_amounts = df['amount'] == 0
            if zero_amounts.any():
                validation_results['warnings'].append(f"{zero_amounts.sum()} transactions have zero amounts")
            
            # Check for extremely large amounts
            large_amounts = df['amount'] > 1000000
            if large_amounts.any():
                validation_results['warnings'].append(f"{large_amounts.sum()} transactions have amounts > $1M")
        
        # Check for duplicate transaction IDs
        if 'transaction_id' in df.columns:
            duplicate_ids = df['transaction_id'].duplicated()
            if duplicate_ids.any():
                validation_results['warnings'].append(f"{duplicate_ids.sum()} duplicate transaction IDs found")
        
        # Check timestamp format
        if 'timestamp' in df.columns:
            try:
                pd.to_datetime(df['timestamp'])
            except:
                validation_results['errors'].append("Invalid timestamp format detected")
                validation_results['is_valid'] = False
        
        # Check for missing values
        missing_summary = df.isnull().sum()
        critical_missing = missing_summary[missing_summary > 0]
        if len(critical_missing) > 0:
            validation_results['warnings'].append(f"Missing values detected: {critical_missing.to_dict()}")
        
        # Generate summary
        validation_results['summary'] = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'missing_values': missing_summary.sum(),
            'duplicate_transaction_ids': df['transaction_id'].duplicated().sum() if 'transaction_id' in df.columns else 0
        }
        
        return validation_results
    
    @staticmethod
    def calculate_fraud_metrics(transactions_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate various fraud detection metrics"""
        if 'is_fraud' not in transactions_df.columns:
            raise ValueError("Fraud labels not available")
        
        total_transactions = len(transactions_df)
        fraud_transactions = transactions_df['is_fraud'].sum()
        normal_transactions = total_transactions - fraud_transactions
        
        # Basic metrics
        fraud_rate = fraud_transactions / total_transactions if total_transactions > 0 else 0
        
        # Amount-based metrics
        total_amount = transactions_df['amount'].sum()
        fraud_amount = transactions_df[transactions_df['is_fraud']]['amount'].sum()
        fraud_amount_rate = fraud_amount / total_amount if total_amount > 0 else 0
        
        # Average amounts
        avg_fraud_amount = transactions_df[transactions_df['is_fraud']]['amount'].mean() if fraud_transactions > 0 else 0
        avg_normal_amount = transactions_df[~transactions_df['is_fraud']]['amount'].mean() if normal_transactions > 0 else 0
        
        return {
            'fraud_rate': fraud_rate,
            'fraud_amount_rate': fraud_amount_rate,
            'total_transactions': total_transactions,
            'fraud_transactions': fraud_transactions,
            'normal_transactions': normal_transactions,
            'total_amount': total_amount,
            'fraud_amount': fraud_amount,
            'normal_amount': total_amount - fraud_amount,
            'avg_fraud_amount': avg_fraud_amount,
            'avg_normal_amount': avg_normal_amount,
            'fraud_to_normal_ratio': avg_fraud_amount / avg_normal_amount if avg_normal_amount > 0 else 0
        }
    
    @staticmethod
    def format_currency(amount: float) -> str:
        """Format amount as currency string"""
        return f"${amount:,.2f}"
    
    @staticmethod
    def format_percentage(value: float) -> str:
        """Format value as percentage string"""
        return f"{value:.1%}"
    
    @staticmethod
    def get_risk_color(risk_score: float) -> str:
        """Get color code based on risk score"""
        if risk_score >= 0.7:
            return "#d62728"  # Red
        elif risk_score >= 0.4:
            return "#ff9800"  # Orange
        elif risk_score >= 0.2:
            return "#ffc107"  # Yellow
        else:
            return "#2ca02c"  # Green
    
    @staticmethod
    def create_data_sample(num_records: int = 1000) -> pd.DataFrame:
        """Create sample transaction data for testing (only when explicitly requested)"""
        # This method should only be used when explicitly requested by user
        # and not for production use
        
        np.random.seed(42)  # For reproducible results
        
        # Generate sample data
        transaction_ids = [f"TXN_{i:06d}" for i in range(1, num_records + 1)]
        
        # Generate realistic amounts with some outliers
        amounts = np.random.lognormal(mean=3, sigma=1, size=num_records)
        amounts = np.round(amounts, 2)
        
        # Add some outliers
        outlier_indices = np.random.choice(num_records, size=int(num_records * 0.05), replace=False)
        amounts[outlier_indices] = np.random.uniform(10000, 50000, size=len(outlier_indices))
        
        # Generate merchants
        merchants = [
            "Amazon", "Walmart", "Target", "Best Buy", "Home Depot", "Starbucks",
            "McDonald's", "Shell Gas", "Costco", "CVS Pharmacy", "Uber", "Netflix",
            "PayPal", "Apple Store", "Google Play", "Microsoft Store"
        ]
        merchant_list = np.random.choice(merchants, size=num_records)
        
        # Generate locations
        locations = [
            "New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX",
            "Phoenix, AZ", "Philadelphia, PA", "San Antonio, TX", "San Diego, CA",
            "Dallas, TX", "San Jose, CA", "Austin, TX", "Jacksonville, FL",
            "Fort Worth, TX", "Columbus, OH", "Charlotte, NC", "San Francisco, CA"
        ]
        location_list = np.random.choice(locations, size=num_records)
        
        # Generate users
        user_ids = [f"USER_{i:04d}" for i in np.random.randint(1, 501, size=num_records)]
        
        # Generate timestamps (last 90 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        timestamps = [
            start_date + timedelta(
                seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))
            ) for _ in range(num_records)
        ]
        
        # Generate card types
        card_types = np.random.choice(
            ["Visa", "Mastercard", "American Express", "Discover"],
            size=num_records,
            p=[0.5, 0.3, 0.15, 0.05]
        )
        
        # Generate merchant categories
        categories = [
            "Retail", "Gas Station", "Restaurant", "Grocery", "Online",
            "Entertainment", "Travel", "Healthcare", "Automotive", "Utilities"
        ]
        category_list = np.random.choice(categories, size=num_records)
        
        # Create DataFrame
        sample_df = pd.DataFrame({
            'transaction_id': transaction_ids,
            'amount': amounts,
            'merchant': merchant_list,
            'location': location_list,
            'timestamp': timestamps,
            'user_id': user_ids,
            'card_type': card_types,
            'merchant_category': category_list
        })
        
        return sample_df.sort_values('timestamp')
