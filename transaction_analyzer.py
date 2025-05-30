import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import statistics

class TransactionAnalyzer:
    def __init__(self):
        self.analysis_cache = {}
    
    def analyze_single_transaction(self, transaction: pd.Series, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze a single transaction and provide detailed risk assessment
        """
        risk_factors = []
        risk_score = 0.0
        
        # User behavior analysis
        user_analysis = self._analyze_user_behavior(transaction, historical_data)
        risk_score += user_analysis['risk_contribution']
        risk_factors.extend(user_analysis['factors'])
        
        # Amount analysis
        amount_analysis = self._analyze_amount_patterns(transaction, historical_data)
        risk_score += amount_analysis['risk_contribution']
        risk_factors.extend(amount_analysis['factors'])
        
        # Temporal analysis
        temporal_analysis = self._analyze_temporal_patterns(transaction, historical_data)
        risk_score += temporal_analysis['risk_contribution']
        risk_factors.extend(temporal_analysis['factors'])
        
        # Location analysis
        location_analysis = self._analyze_location_patterns(transaction, historical_data)
        risk_score += location_analysis['risk_contribution']
        risk_factors.extend(location_analysis['factors'])
        
        # Merchant analysis
        merchant_analysis = self._analyze_merchant_patterns(transaction, historical_data)
        risk_score += merchant_analysis['risk_contribution']
        risk_factors.extend(merchant_analysis['factors'])
        
        # Cap risk score at 1.0
        risk_score = min(risk_score, 1.0)
        
        return {
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'user_analysis': user_analysis,
            'amount_analysis': amount_analysis,
            'temporal_analysis': temporal_analysis,
            'location_analysis': location_analysis,
            'merchant_analysis': merchant_analysis,
            'recommendation': self._get_recommendation(risk_score, risk_factors)
        }
    
    def _analyze_user_behavior(self, transaction: pd.Series, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze user behavior patterns"""
        user_id = transaction['user_id']
        user_transactions = historical_data[historical_data['user_id'] == user_id]
        
        risk_contribution = 0.0
        factors = []
        analysis_details = {}
        
        if len(user_transactions) == 0:
            # New user - higher risk
            risk_contribution += 0.3
            factors.append("New user - no transaction history")
            analysis_details['is_new_user'] = True
            analysis_details['transaction_count'] = 0
        else:
            analysis_details['is_new_user'] = False
            analysis_details['transaction_count'] = len(user_transactions)
            
            # Calculate user statistics
            avg_amount = user_transactions['amount'].mean()
            std_amount = user_transactions['amount'].std()
            
            analysis_details['avg_amount'] = avg_amount
            analysis_details['std_amount'] = std_amount
            
            # Check if current transaction is unusual for this user
            if std_amount > 0:
                z_score = abs((transaction['amount'] - avg_amount) / std_amount)
                analysis_details['amount_z_score'] = z_score
                
                if z_score > 3:
                    risk_contribution += 0.25
                    factors.append(f"Amount highly unusual for user (Z-score: {z_score:.2f})")
                elif z_score > 2:
                    risk_contribution += 0.15
                    factors.append(f"Amount somewhat unusual for user (Z-score: {z_score:.2f})")
            
            # Check transaction frequency
            last_transaction = user_transactions['timestamp'].max()
            time_since_last = transaction['timestamp'] - last_transaction
            
            analysis_details['time_since_last_transaction'] = time_since_last
            
            if time_since_last < timedelta(minutes=5):
                risk_contribution += 0.2
                factors.append("Very recent transaction from same user")
            
            # Check for unusual spending patterns
            recent_transactions = user_transactions[
                user_transactions['timestamp'] > (transaction['timestamp'] - timedelta(days=1))
            ]
            
            if len(recent_transactions) > 0:
                daily_spending = recent_transactions['amount'].sum() + transaction['amount']
                avg_daily_spending = user_transactions.groupby(
                    user_transactions['timestamp'].dt.date
                )['amount'].sum().mean()
                
                analysis_details['daily_spending'] = daily_spending
                analysis_details['avg_daily_spending'] = avg_daily_spending
                
                if daily_spending > avg_daily_spending * 3:
                    risk_contribution += 0.2
                    factors.append("Daily spending significantly above average")
        
        return {
            'risk_contribution': risk_contribution,
            'factors': factors,
            'details': analysis_details
        }
    
    def _analyze_amount_patterns(self, transaction: pd.Series, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze transaction amount patterns"""
        amount = transaction['amount']
        risk_contribution = 0.0
        factors = []
        analysis_details = {}
        
        # Calculate global statistics
        global_stats = {
            'mean': historical_data['amount'].mean(),
            'std': historical_data['amount'].std(),
            'median': historical_data['amount'].median(),
            'q75': historical_data['amount'].quantile(0.75),
            'q95': historical_data['amount'].quantile(0.95),
            'q99': historical_data['amount'].quantile(0.99)
        }
        
        analysis_details['global_stats'] = global_stats
        analysis_details['amount_percentile'] = (historical_data['amount'] < amount).mean()
        
        # Very large amounts
        if amount > global_stats['q99']:
            risk_contribution += 0.3
            factors.append(f"Amount in top 1% of all transactions (${amount:,.2f})")
        elif amount > global_stats['q95']:
            risk_contribution += 0.15
            factors.append(f"Amount in top 5% of all transactions (${amount:,.2f})")
        
        # Very small amounts (potential testing)
        if amount < 1.0:
            risk_contribution += 0.15
            factors.append(f"Micro transaction amount (${amount:.2f})")
        
        # Round amounts (potential fraud indicator)
        if amount >= 100 and amount % 100 == 0:
            risk_contribution += 0.1
            factors.append(f"Round amount (${amount:,.0f})")
        elif amount >= 10 and amount % 10 == 0:
            risk_contribution += 0.05
            factors.append(f"Round amount to nearest $10 (${amount:,.0f})")
        
        # Statistical outlier check
        if global_stats['std'] > 0:
            z_score = abs((amount - global_stats['mean']) / global_stats['std'])
            analysis_details['z_score'] = z_score
            
            if z_score > 3:
                risk_contribution += 0.2
                factors.append(f"Statistical outlier (Z-score: {z_score:.2f})")
        
        return {
            'risk_contribution': risk_contribution,
            'factors': factors,
            'details': analysis_details
        }
    
    def _analyze_temporal_patterns(self, transaction: pd.Series, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns"""
        timestamp = transaction['timestamp']
        risk_contribution = 0.0
        factors = []
        analysis_details = {}
        
        # Extract time features
        hour = timestamp.hour
        day_of_week = timestamp.weekday()  # 0=Monday, 6=Sunday
        
        analysis_details['hour'] = hour
        analysis_details['day_of_week'] = day_of_week
        analysis_details['is_weekend'] = day_of_week >= 5
        
        # Off-hours transactions (10 PM to 6 AM)
        if hour >= 22 or hour <= 6:
            risk_contribution += 0.15
            factors.append(f"Off-hours transaction ({hour:02d}:00)")
            analysis_details['is_off_hours'] = True
        else:
            analysis_details['is_off_hours'] = False
        
        # Weekend transactions
        if day_of_week >= 5:
            risk_contribution += 0.05
            factors.append("Weekend transaction")
        
        # Holiday check (simplified - would need holiday calendar in production)
        # For now, check if it's January 1st or December 25th
        if (timestamp.month == 1 and timestamp.day == 1) or (timestamp.month == 12 and timestamp.day == 25):
            risk_contribution += 0.1
            factors.append("Holiday transaction")
            analysis_details['is_holiday'] = True
        else:
            analysis_details['is_holiday'] = False
        
        # Analyze historical patterns for this hour
        hour_transactions = historical_data[historical_data['timestamp'].dt.hour == hour]
        analysis_details['hour_transaction_count'] = len(hour_transactions)
        
        if len(hour_transactions) < len(historical_data) * 0.01:  # Less than 1% of transactions
            risk_contribution += 0.1
            factors.append(f"Unusual hour for transactions (only {len(hour_transactions)} historical transactions)")
        
        return {
            'risk_contribution': risk_contribution,
            'factors': factors,
            'details': analysis_details
        }
    
    def _analyze_location_patterns(self, transaction: pd.Series, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze location patterns"""
        location = transaction['location']
        user_id = transaction['user_id']
        risk_contribution = 0.0
        factors = []
        analysis_details = {}
        
        # User's location history
        user_transactions = historical_data[historical_data['user_id'] == user_id]
        user_locations = user_transactions['location'].value_counts()
        
        analysis_details['user_location_count'] = len(user_locations)
        analysis_details['is_new_location'] = location not in user_locations.index
        
        if len(user_locations) > 0:
            analysis_details['most_common_location'] = user_locations.index[0]
            analysis_details['location_frequency'] = user_locations.get(location, 0)
            
            # New location for user
            if location not in user_locations.index:
                risk_contribution += 0.2
                factors.append("New location for this user")
            elif user_locations[location] == 1:
                risk_contribution += 0.1
                factors.append("Rarely used location for this user")
        
        # Global location analysis
        global_locations = historical_data['location'].value_counts()
        analysis_details['global_location_frequency'] = global_locations.get(location, 0)
        
        # Uncommon location globally
        if location not in global_locations.index:
            risk_contribution += 0.15
            factors.append("New location globally")
        elif global_locations[location] < 5:
            risk_contribution += 0.1
            factors.append("Rarely used location globally")
        
        # Check for suspicious location names
        suspicious_keywords = ['test', 'temp', 'fake', 'unknown', 'null', '']
        if any(keyword.lower() in location.lower() for keyword in suspicious_keywords):
            risk_contribution += 0.25
            factors.append("Suspicious location name")
        
        return {
            'risk_contribution': risk_contribution,
            'factors': factors,
            'details': analysis_details
        }
    
    def _analyze_merchant_patterns(self, transaction: pd.Series, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze merchant patterns"""
        merchant = transaction['merchant']
        merchant_category = transaction['merchant_category']
        user_id = transaction['user_id']
        risk_contribution = 0.0
        factors = []
        analysis_details = {}
        
        # User's merchant history
        user_transactions = historical_data[historical_data['user_id'] == user_id]
        user_merchants = user_transactions['merchant'].value_counts()
        user_categories = user_transactions['merchant_category'].value_counts()
        
        analysis_details['user_merchant_count'] = len(user_merchants)
        analysis_details['user_category_count'] = len(user_categories)
        
        # New merchant for user
        if len(user_merchants) > 0:
            if merchant not in user_merchants.index:
                risk_contribution += 0.1
                factors.append("New merchant for this user")
            
            # New category for user
            if merchant_category not in user_categories.index:
                risk_contribution += 0.15
                factors.append("New merchant category for this user")
        
        # Global merchant analysis
        global_merchants = historical_data['merchant'].value_counts()
        global_categories = historical_data['merchant_category'].value_counts()
        
        analysis_details['global_merchant_frequency'] = global_merchants.get(merchant, 0)
        
        # New merchant globally
        if merchant not in global_merchants.index:
            risk_contribution += 0.15
            factors.append("New merchant globally")
        elif global_merchants[merchant] < 5:
            risk_contribution += 0.1
            factors.append("Rarely used merchant")
        
        # High-risk merchant categories
        high_risk_categories = ['gambling', 'adult', 'cryptocurrency', 'cash advance', 'money transfer']
        if any(risk_cat.lower() in merchant_category.lower() for risk_cat in high_risk_categories):
            risk_contribution += 0.2
            factors.append(f"High-risk merchant category: {merchant_category}")
        
        # Suspicious merchant names
        suspicious_keywords = ['test', 'temp', 'fake', 'unknown', 'null']
        if any(keyword.lower() in merchant.lower() for keyword in suspicious_keywords):
            risk_contribution += 0.25
            factors.append("Suspicious merchant name")
        
        return {
            'risk_contribution': risk_contribution,
            'factors': factors,
            'details': analysis_details
        }
    
    def _get_recommendation(self, risk_score: float, risk_factors: List[str]) -> Dict[str, Any]:
        """Generate recommendation based on risk analysis"""
        if risk_score >= 0.8:
            action = "BLOCK"
            confidence = "HIGH"
            reasoning = "Multiple high-risk indicators detected"
        elif risk_score >= 0.6:
            action = "REVIEW"
            confidence = "MEDIUM"
            reasoning = "Several risk factors present, manual review recommended"
        elif risk_score >= 0.3:
            action = "MONITOR"
            confidence = "LOW"
            reasoning = "Some risk factors present, continue monitoring"
        else:
            action = "APPROVE"
            confidence = "HIGH"
            reasoning = "Low risk transaction"
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': reasoning,
            'risk_level': self._get_risk_level(risk_score),
            'primary_concerns': risk_factors[:3] if risk_factors else []
        }
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level"""
        if risk_score >= 0.7:
            return "HIGH"
        elif risk_score >= 0.4:
            return "MEDIUM"
        elif risk_score >= 0.2:
            return "LOW"
        else:
            return "VERY LOW"
    
    def generate_user_profile(self, user_id: str, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive user profile for fraud analysis"""
        user_transactions = historical_data[historical_data['user_id'] == user_id]
        
        if len(user_transactions) == 0:
            return {'error': 'No transactions found for user'}
        
        profile = {
            'user_id': user_id,
            'total_transactions': len(user_transactions),
            'date_range': {
                'first_transaction': user_transactions['timestamp'].min(),
                'last_transaction': user_transactions['timestamp'].max(),
                'account_age_days': (user_transactions['timestamp'].max() - user_transactions['timestamp'].min()).days
            },
            'spending_patterns': {
                'total_amount': user_transactions['amount'].sum(),
                'average_amount': user_transactions['amount'].mean(),
                'median_amount': user_transactions['amount'].median(),
                'std_amount': user_transactions['amount'].std(),
                'min_amount': user_transactions['amount'].min(),
                'max_amount': user_transactions['amount'].max()
            },
            'location_patterns': {
                'unique_locations': user_transactions['location'].nunique(),
                'most_common_location': user_transactions['location'].mode().iloc[0] if not user_transactions['location'].mode().empty else 'Unknown',
                'location_distribution': user_transactions['location'].value_counts().to_dict()
            },
            'merchant_patterns': {
                'unique_merchants': user_transactions['merchant'].nunique(),
                'unique_categories': user_transactions['merchant_category'].nunique(),
                'most_common_merchant': user_transactions['merchant'].mode().iloc[0] if not user_transactions['merchant'].mode().empty else 'Unknown',
                'category_distribution': user_transactions['merchant_category'].value_counts().to_dict()
            },
            'temporal_patterns': {
                'transactions_by_hour': user_transactions.groupby(user_transactions['timestamp'].dt.hour).size().to_dict(),
                'transactions_by_day': user_transactions.groupby(user_transactions['timestamp'].dt.day_name()).size().to_dict(),
                'average_transactions_per_day': len(user_transactions) / max(1, (user_transactions['timestamp'].max() - user_transactions['timestamp'].min()).days)
            }
        }
        
        return profile
