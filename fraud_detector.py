import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class FraudDetector:
    def __init__(self):
        self.settings = {
            'high_risk_threshold': 0.7,
            'medium_risk_threshold': 0.4,
            'unusual_amount_threshold': 10000.0,
            'micro_transaction_threshold': 1.0,
            'off_hours_start': datetime.strptime("22:00", "%H:%M").time(),
            'off_hours_end': datetime.strptime("06:00", "%H:%M").time(),
            'max_transactions_per_hour': 10,
            'max_amount_per_day': 50000.0
        }
    
    def update_settings(self, new_settings: Dict):
        """Update fraud detection settings"""
        self.settings.update(new_settings)
    
    def detect_fraud(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Main fraud detection function that applies multiple rules and returns
        transactions with fraud scores and flags
        """
        df = transactions.copy()
        
        # Initialize fraud detection columns
        df['risk_score'] = 0.0
        df['is_fraud'] = False
        df['fraud_reasons'] = ''
        
        # Apply various fraud detection rules
        df = self._check_amount_anomalies(df)
        df = self._check_time_patterns(df)
        df = self._check_velocity_patterns(df)
        df = self._check_location_patterns(df)
        df = self._check_merchant_patterns(df)
        df = self._calculate_final_score(df)
        
        return df
    
    def _check_amount_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check for unusual transaction amounts"""
        # Statistical outliers
        amount_mean = df['amount'].mean()
        amount_std = df['amount'].std()
        z_threshold = 3
        
        # Flag transactions with extreme amounts
        extreme_amounts = np.abs((df['amount'] - amount_mean) / amount_std) > z_threshold
        df.loc[extreme_amounts, 'risk_score'] += 0.3
        df.loc[extreme_amounts, 'fraud_reasons'] += 'Extreme amount, '
        
        # Flag very large transactions
        large_amounts = df['amount'] > self.settings['unusual_amount_threshold']
        df.loc[large_amounts, 'risk_score'] += 0.2
        df.loc[large_amounts, 'fraud_reasons'] += 'Large amount, '
        
        # Flag micro transactions (potential testing)
        micro_amounts = df['amount'] < self.settings['micro_transaction_threshold']
        df.loc[micro_amounts, 'risk_score'] += 0.1
        df.loc[micro_amounts, 'fraud_reasons'] += 'Micro transaction, '
        
        # Flag round amounts (potential fraud indicator)
        round_amounts = (df['amount'] % 100 == 0) & (df['amount'] > 100)
        df.loc[round_amounts, 'risk_score'] += 0.05
        df.loc[round_amounts, 'fraud_reasons'] += 'Round amount, '
        
        return df
    
    def _check_time_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check for suspicious time patterns"""
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Off-hours transactions
        off_hours_start = self.settings['off_hours_start'].hour
        off_hours_end = self.settings['off_hours_end'].hour
        
        if off_hours_start > off_hours_end:  # Spans midnight
            off_hours = (df['hour'] >= off_hours_start) | (df['hour'] <= off_hours_end)
        else:
            off_hours = (df['hour'] >= off_hours_start) & (df['hour'] <= off_hours_end)
        
        df.loc[off_hours, 'risk_score'] += 0.15
        df.loc[off_hours, 'fraud_reasons'] += 'Off-hours transaction, '
        
        # Weekend transactions (higher risk for certain merchant types)
        weekend_transactions = df['day_of_week'].isin([5, 6])  # Saturday, Sunday
        df.loc[weekend_transactions, 'risk_score'] += 0.05
        df.loc[weekend_transactions, 'fraud_reasons'] += 'Weekend transaction, '
        
        return df
    
    def _check_velocity_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check for transaction velocity anomalies"""
        df_sorted = df.sort_values('timestamp')
        
        # Check for rapid-fire transactions from same user
        for user_id in df['user_id'].unique():
            user_transactions = df_sorted[df_sorted['user_id'] == user_id].copy()
            
            if len(user_transactions) > 1:
                # Time differences between consecutive transactions
                user_transactions['time_diff'] = user_transactions['timestamp'].diff()
                
                # Flag transactions within 1 minute of each other
                rapid_transactions = user_transactions['time_diff'] < timedelta(minutes=1)
                rapid_indices = user_transactions[rapid_transactions].index
                
                df.loc[rapid_indices, 'risk_score'] += 0.25
                df.loc[rapid_indices, 'fraud_reasons'] += 'Rapid consecutive transactions, '
                
                # Check hourly transaction count
                user_transactions['hour_group'] = user_transactions['timestamp'].dt.floor('H')
                hourly_counts = user_transactions.groupby('hour_group').size()
                excessive_hourly = hourly_counts > self.settings['max_transactions_per_hour']
                
                if excessive_hourly.any():
                    excessive_hours = hourly_counts[excessive_hourly].index
                    for hour in excessive_hours:
                        hour_transactions = user_transactions[
                            user_transactions['hour_group'] == hour
                        ].index
                        df.loc[hour_transactions, 'risk_score'] += 0.2
                        df.loc[hour_transactions, 'fraud_reasons'] += 'High hourly velocity, '
                
                # Check daily amount limits
                user_transactions['date'] = user_transactions['timestamp'].dt.date
                daily_amounts = user_transactions.groupby('date')['amount'].sum()
                excessive_daily = daily_amounts > self.settings['max_amount_per_day']
                
                if excessive_daily.any():
                    excessive_dates = daily_amounts[excessive_daily].index
                    for date in excessive_dates:
                        date_transactions = user_transactions[
                            user_transactions['date'] == date
                        ].index
                        df.loc[date_transactions, 'risk_score'] += 0.15
                        df.loc[date_transactions, 'fraud_reasons'] += 'High daily amount, '
        
        return df
    
    def _check_location_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check for suspicious location patterns"""
        # Check for location velocity (impossible travel)
        df_sorted = df.sort_values('timestamp')
        
        for user_id in df['user_id'].unique():
            user_transactions = df_sorted[df_sorted['user_id'] == user_id].copy()
            
            if len(user_transactions) > 1:
                # Simple location change detection
                user_transactions['location_changed'] = (
                    user_transactions['location'] != user_transactions['location'].shift(1)
                )
                user_transactions['time_diff'] = user_transactions['timestamp'].diff()
                
                # Flag rapid location changes (within 1 hour)
                rapid_location_change = (
                    user_transactions['location_changed'] & 
                    (user_transactions['time_diff'] < timedelta(hours=1))
                )
                
                rapid_indices = user_transactions[rapid_location_change].index
                df.loc[rapid_indices, 'risk_score'] += 0.3
                df.loc[rapid_indices, 'fraud_reasons'] += 'Rapid location change, '
        
        # Check for high-risk locations (simplified - could be enhanced with real data)
        # This would typically use a database of known high-risk locations
        high_risk_keywords = ['unknown', 'test', 'temp', 'null', '']
        for keyword in high_risk_keywords:
            high_risk_locations = df['location'].str.lower().str.contains(keyword, na=False)
            df.loc[high_risk_locations, 'risk_score'] += 0.1
            df.loc[high_risk_locations, 'fraud_reasons'] += 'High-risk location, '
        
        return df
    
    def _check_merchant_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check for suspicious merchant patterns"""
        # Check for new/unknown merchants
        merchant_counts = df['merchant'].value_counts()
        rare_merchants = merchant_counts[merchant_counts == 1].index
        
        rare_merchant_mask = df['merchant'].isin(rare_merchants)
        df.loc[rare_merchant_mask, 'risk_score'] += 0.1
        df.loc[rare_merchant_mask, 'fraud_reasons'] += 'New/rare merchant, '
        
        # Check for high-risk merchant categories
        high_risk_categories = ['gambling', 'adult', 'cryptocurrency', 'cash_advance']
        for category in high_risk_categories:
            high_risk_mask = df['merchant_category'].str.lower().str.contains(category, na=False)
            df.loc[high_risk_mask, 'risk_score'] += 0.15
            df.loc[high_risk_mask, 'fraud_reasons'] += f'High-risk category ({category}), '
        
        # Check for merchant name anomalies
        suspicious_keywords = ['test', 'temp', 'fake', 'dummy']
        for keyword in suspicious_keywords:
            suspicious_mask = df['merchant'].str.lower().str.contains(keyword, na=False)
            df.loc[suspicious_mask, 'risk_score'] += 0.2
            df.loc[suspicious_mask, 'fraud_reasons'] += 'Suspicious merchant name, '
        
        return df
    
    def _calculate_final_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate final risk score and fraud flag"""
        # Cap risk score at 1.0
        df['risk_score'] = np.minimum(df['risk_score'], 1.0)
        
        # Set fraud flag based on threshold
        df['is_fraud'] = df['risk_score'] >= self.settings['high_risk_threshold']
        
        # Clean up fraud reasons (remove trailing commas and spaces)
        df['fraud_reasons'] = df['fraud_reasons'].str.rstrip(', ')
        
        # Drop temporary columns
        columns_to_drop = ['hour', 'day_of_week']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        return df
    
    def get_fraud_statistics(self, transactions: pd.DataFrame) -> Dict:
        """Generate fraud detection statistics"""
        transactions_with_scores = self.detect_fraud(transactions)
        
        total_transactions = len(transactions_with_scores)
        fraud_transactions = len(transactions_with_scores[transactions_with_scores['is_fraud']])
        high_risk_transactions = len(transactions_with_scores[transactions_with_scores['risk_score'] > 0.7])
        medium_risk_transactions = len(transactions_with_scores[
            (transactions_with_scores['risk_score'] > 0.4) & 
            (transactions_with_scores['risk_score'] <= 0.7)
        ])
        
        fraud_amount = transactions_with_scores[transactions_with_scores['is_fraud']]['amount'].sum()
        total_amount = transactions_with_scores['amount'].sum()
        
        return {
            'total_transactions': total_transactions,
            'fraud_transactions': fraud_transactions,
            'fraud_rate': fraud_transactions / total_transactions if total_transactions > 0 else 0,
            'high_risk_transactions': high_risk_transactions,
            'medium_risk_transactions': medium_risk_transactions,
            'fraud_amount': fraud_amount,
            'total_amount': total_amount,
            'fraud_amount_rate': fraud_amount / total_amount if total_amount > 0 else 0,
            'average_risk_score': transactions_with_scores['risk_score'].mean()
        }
