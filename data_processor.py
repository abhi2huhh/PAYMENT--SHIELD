import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import re

class DataProcessor:
    def __init__(self):
        self.required_columns = [
            'transaction_id', 'amount', 'merchant', 'location', 
            'timestamp', 'user_id', 'card_type', 'merchant_category'
        ]
    
    def process_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw transaction data and prepare it for fraud detection
        """
        # Create a copy to avoid modifying original data
        processed_df = df.copy()
        
        # Validate required columns
        processed_df = self._validate_columns(processed_df)
        
        # Clean and standardize data
        processed_df = self._clean_data(processed_df)
        
        # Parse and validate timestamps
        processed_df = self._process_timestamps(processed_df)
        
        # Standardize amounts
        processed_df = self._process_amounts(processed_df)
        
        # Clean text fields
        processed_df = self._clean_text_fields(processed_df)
        
        # Add derived features
        processed_df = self._add_derived_features(processed_df)
        
        # Remove invalid records
        processed_df = self._remove_invalid_records(processed_df)
        
        return processed_df
    
    def _validate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that all required columns are present"""
        missing_columns = []
        
        for col in self.required_columns:
            if col not in df.columns:
                missing_columns.append(col)
        
        if missing_columns:
            # Try to map common alternative column names
            column_mapping = {
                'id': 'transaction_id',
                'txn_id': 'transaction_id',
                'transaction_amount': 'amount',
                'amt': 'amount',
                'value': 'amount',
                'store': 'merchant',
                'shop': 'merchant',
                'vendor': 'merchant',
                'city': 'location',
                'country': 'location',
                'place': 'location',
                'datetime': 'timestamp',
                'date': 'timestamp',
                'time': 'timestamp',
                'customer_id': 'user_id',
                'client_id': 'user_id',
                'payment_method': 'card_type',
                'card': 'card_type',
                'category': 'merchant_category',
                'type': 'merchant_category'
            }
            
            # Apply column mapping
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns and new_name in missing_columns:
                    df = df.rename(columns={old_name: new_name})
                    missing_columns.remove(new_name)
            
            # Create missing columns with default values
            for col in missing_columns:
                if col == 'transaction_id':
                    df[col] = range(1, len(df) + 1)
                elif col == 'amount':
                    df[col] = 0.0
                elif col == 'merchant':
                    df[col] = 'Unknown Merchant'
                elif col == 'location':
                    df[col] = 'Unknown Location'
                elif col == 'timestamp':
                    df[col] = datetime.now()
                elif col == 'user_id':
                    df[col] = 'Unknown User'
                elif col == 'card_type':
                    df[col] = 'Unknown'
                elif col == 'merchant_category':
                    df[col] = 'Unknown'
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the data"""
        # Remove duplicate transactions
        df = df.drop_duplicates(subset=['transaction_id'])
        
        # Handle missing values
        df['merchant'] = df['merchant'].fillna('Unknown Merchant')
        df['location'] = df['location'].fillna('Unknown Location')
        df['card_type'] = df['card_type'].fillna('Unknown')
        df['merchant_category'] = df['merchant_category'].fillna('Unknown')
        df['user_id'] = df['user_id'].fillna('Unknown User')
        
        return df
    
    def _process_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and validate timestamps"""
        # Convert timestamp to datetime
        if df['timestamp'].dtype == 'object':
            # Try different datetime formats
            formats_to_try = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%d',
                '%m/%d/%Y %H:%M:%S',
                '%m/%d/%Y',
                '%d/%m/%Y %H:%M:%S',
                '%d/%m/%Y'
            ]
            
            timestamp_converted = False
            for fmt in formats_to_try:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format=fmt)
                    timestamp_converted = True
                    break
                except (ValueError, TypeError):
                    continue
            
            if not timestamp_converted:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                except:
                    # If all fails, use current time with incremental seconds
                    base_time = datetime.now()
                    df['timestamp'] = [base_time + timedelta(seconds=i) for i in range(len(df))]
        
        # Ensure timestamps are in the past
        future_timestamps = df['timestamp'] > datetime.now()
        if future_timestamps.any():
            # Set future timestamps to current time
            df.loc[future_timestamps, 'timestamp'] = datetime.now()
        
        return df
    
    def _process_amounts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and validate transaction amounts"""
        # Convert amount to numeric
        if df['amount'].dtype == 'object':
            # Remove currency symbols and commas
            df['amount'] = df['amount'].astype(str).str.replace(r'[$,€£¥]', '', regex=True)
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Handle negative amounts (make them positive)
        df['amount'] = df['amount'].abs()
        
        # Fill NaN amounts with 0
        df['amount'] = df['amount'].fillna(0.0)
        
        # Remove extremely large amounts (likely data errors)
        max_reasonable_amount = 1000000  # $1M
        df.loc[df['amount'] > max_reasonable_amount, 'amount'] = max_reasonable_amount
        
        return df
    
    def _clean_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize text fields"""
        text_columns = ['merchant', 'location', 'card_type', 'merchant_category']
        
        for col in text_columns:
            if col in df.columns:
                # Strip whitespace and convert to title case
                df[col] = df[col].astype(str).str.strip().str.title()
                
                # Replace empty strings with 'Unknown'
                df.loc[df[col].isin(['', 'Nan', 'None', 'Null']), col] = 'Unknown'
                
                # Clean special characters from merchant names
                if col == 'merchant':
                    df[col] = df[col].str.replace(r'[^\w\s-]', '', regex=True)
                
                # Standardize location format
                if col == 'location':
                    # Simple cleaning - remove extra spaces and standardize separators
                    df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
                    df[col] = df[col].str.replace(r'[,;]', ',', regex=True)
                
                # Standardize card types
                if col == 'card_type':
                    card_type_mapping = {
                        'visa': 'Visa',
                        'mastercard': 'Mastercard',
                        'master card': 'Mastercard',
                        'amex': 'American Express',
                        'american express': 'American Express',
                        'discover': 'Discover',
                        'debit': 'Debit',
                        'credit': 'Credit'
                    }
                    
                    df[col] = df[col].str.lower()
                    for old_val, new_val in card_type_mapping.items():
                        df.loc[df[col].str.contains(old_val, na=False), col] = new_val
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features that might be useful for fraud detection"""
        # Add time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Add amount-based features
        df['amount_rounded'] = (df['amount'] % 1 == 0)  # Check if amount is rounded
        df['amount_log'] = np.log1p(df['amount'])  # Log transform for amount
        
        # Add merchant features
        df['merchant_length'] = df['merchant'].str.len()
        df['merchant_word_count'] = df['merchant'].str.split().str.len()
        
        return df
    
    def _remove_invalid_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove records that are clearly invalid"""
        initial_count = len(df)
        
        # Remove records with zero amounts (unless they're legitimate)
        df = df[df['amount'] > 0]
        
        # Remove records with invalid user IDs
        df = df[df['user_id'].notna()]
        df = df[df['user_id'] != '']
        
        # Remove records with invalid timestamps
        df = df[df['timestamp'].notna()]
        
        final_count = len(df)
        removed_count = initial_count - final_count
        
        if removed_count > 0:
            print(f"Removed {removed_count} invalid records during processing")
        
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Validate data quality and return quality metrics"""
        quality_metrics = {
            'total_records': len(df),
            'missing_values': {},
            'data_types': {},
            'unique_values': {},
            'potential_issues': []
        }
        
        # Check for missing values
        for col in df.columns:
            missing_count = df[col].isna().sum()
            quality_metrics['missing_values'][col] = missing_count
            
            if missing_count > 0:
                quality_metrics['potential_issues'].append(f"{col}: {missing_count} missing values")
        
        # Check data types
        for col in df.columns:
            quality_metrics['data_types'][col] = str(df[col].dtype)
        
        # Check unique values for categorical columns
        categorical_columns = ['merchant', 'location', 'card_type', 'merchant_category']
        for col in categorical_columns:
            if col in df.columns:
                quality_metrics['unique_values'][col] = df[col].nunique()
        
        # Check for potential data issues
        if df['amount'].min() < 0:
            quality_metrics['potential_issues'].append("Negative amounts detected")
        
        if df['amount'].max() > 100000:
            quality_metrics['potential_issues'].append("Very large amounts detected")
        
        # Check timestamp range
        timestamp_range = df['timestamp'].max() - df['timestamp'].min()
        if timestamp_range > timedelta(days=365*5):
            quality_metrics['potential_issues'].append("Very large timestamp range (>5 years)")
        
        return quality_metrics
    
    def export_processed_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """Export processed data to CSV format"""
        if filename is None:
            filename = f"processed_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Select columns for export
        export_columns = [
            'transaction_id', 'timestamp', 'amount', 'merchant', 'location',
            'user_id', 'card_type', 'merchant_category'
        ]
        
        export_df = df[export_columns].copy()
        csv_data = export_df.to_csv(index=False)
        
        return csv_data
