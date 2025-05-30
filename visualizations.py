import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class Visualizations:
    def __init__(self):
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'danger': '#d62728',
            'warning': '#ff9800',
            'success': '#2ca02c',
            'info': '#17a2b8'
        }
    
    def create_risk_distribution(self, transactions_df: pd.DataFrame) -> go.Figure:
        """Create risk score distribution chart"""
        fig = go.Figure()
        
        # Risk score histogram
        fig.add_trace(go.Histogram(
            x=transactions_df['risk_score'],
            nbinsx=20,
            name='Risk Score Distribution',
            marker_color=self.color_scheme['primary'],
            opacity=0.7
        ))
        
        # Add vertical lines for thresholds
        fig.add_vline(
            x=0.4, 
            line_dash="dash", 
            line_color=self.color_scheme['warning'],
            annotation_text="Medium Risk"
        )
        fig.add_vline(
            x=0.7, 
            line_dash="dash", 
            line_color=self.color_scheme['danger'],
            annotation_text="High Risk"
        )
        
        fig.update_layout(
            title="Transaction Risk Score Distribution",
            xaxis_title="Risk Score",
            yaxis_title="Number of Transactions",
            showlegend=False,
            height=400
        )
        
        return fig
    
    def create_time_series(self, transactions_df: pd.DataFrame) -> go.Figure:
        """Create time series chart of transactions"""
        # Group by day
        daily_transactions = transactions_df.groupby(
            transactions_df['timestamp'].dt.date
        ).agg({
            'transaction_id': 'count',
            'amount': 'sum',
            'risk_score': 'mean'
        }).reset_index()
        
        daily_transactions.columns = ['date', 'transaction_count', 'total_amount', 'avg_risk_score']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Transaction Count', 'Average Risk Score'),
            vertical_spacing=0.1
        )
        
        # Transaction count
        fig.add_trace(
            go.Scatter(
                x=daily_transactions['date'],
                y=daily_transactions['transaction_count'],
                mode='lines+markers',
                name='Transaction Count',
                line=dict(color=self.color_scheme['primary'])
            ),
            row=1, col=1
        )
        
        # Average risk score
        fig.add_trace(
            go.Scatter(
                x=daily_transactions['date'],
                y=daily_transactions['avg_risk_score'],
                mode='lines+markers',
                name='Avg Risk Score',
                line=dict(color=self.color_scheme['danger'])
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Transaction Trends Over Time",
            height=500,
            showlegend=False
        )
        
        return fig
    
    def create_location_analysis(self, transactions_df: pd.DataFrame) -> go.Figure:
        """Create location-based fraud analysis"""
        location_stats = transactions_df.groupby('location').agg({
            'transaction_id': 'count',
            'amount': 'sum',
            'risk_score': 'mean',
            'is_fraud': lambda x: (x == True).sum()
        }).reset_index()
        
        location_stats.columns = ['location', 'transaction_count', 'total_amount', 'avg_risk_score', 'fraud_count']
        location_stats['fraud_rate'] = location_stats['fraud_count'] / location_stats['transaction_count']
        location_stats = location_stats.sort_values('fraud_rate', ascending=False).head(10)
        
        fig = go.Figure(data=go.Bar(
            x=location_stats['location'],
            y=location_stats['fraud_rate'],
            marker=dict(
                color=location_stats['avg_risk_score'],
                colorscale='Reds'
            ),
            text=[f"{rate:.1%}" for rate in location_stats['fraud_rate']],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Fraud Rate by Location (Top 10)",
            xaxis_title="Location",
            yaxis_title="Fraud Rate",
            height=400
        )
        
        return fig
    
    def create_amount_distribution(self, transactions_df: pd.DataFrame) -> go.Figure:
        """Create amount distribution chart"""
        fig = go.Figure()
        
        # Normal transactions
        normal_transactions = transactions_df[transactions_df['is_fraud'] == False]
        fraud_transactions = transactions_df[transactions_df['is_fraud'] == True]
        
        fig.add_trace(go.Histogram(
            x=normal_transactions['amount'],
            name='Normal Transactions',
            opacity=0.7,
            marker_color=self.color_scheme['success'],
            nbinsx=30
        ))
        
        if not fraud_transactions.empty:
            fig.add_trace(go.Histogram(
                x=fraud_transactions['amount'],
                name='Fraudulent Transactions',
                opacity=0.7,
                marker_color=self.color_scheme['danger'],
                nbinsx=30
            ))
        
        fig.update_layout(
            title="Transaction Amount Distribution",
            xaxis_title="Amount ($)",
            yaxis_title="Number of Transactions",
            barmode='overlay',
            height=400
        )
        
        return fig
    
    def create_merchant_analysis(self, transactions_df: pd.DataFrame) -> go.Figure:
        """Create merchant category analysis"""
        merchant_stats = transactions_df.groupby('merchant_category').agg({
            'transaction_id': 'count',
            'amount': 'sum',
            'is_fraud': lambda x: (x == True).sum()
        }).reset_index()
        
        merchant_stats.columns = ['merchant_category', 'transaction_count', 'total_amount', 'fraud_count']
        merchant_stats['fraud_rate'] = merchant_stats['fraud_count'] / merchant_stats['transaction_count']
        merchant_stats = merchant_stats.sort_values('transaction_count', ascending=True).tail(10)
        
        fig = go.Figure(data=go.Bar(
            y=merchant_stats['merchant_category'],
            x=merchant_stats['transaction_count'],
            orientation='h',
            marker=dict(
                color=merchant_stats['fraud_rate'],
                colorscale='Reds'
            ),
            text=[f"Fraud: {rate:.1%}" for rate in merchant_stats['fraud_rate']],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Transaction Count by Merchant Category",
            xaxis_title="Number of Transactions",
            yaxis_title="Merchant Category",
            height=400
        )
        
        return fig
    
    def create_hourly_patterns(self, transactions_df: pd.DataFrame) -> go.Figure:
        """Create hourly transaction patterns"""
        hourly_stats = transactions_df.groupby(transactions_df['timestamp'].dt.hour).agg({
            'transaction_id': 'count',
            'amount': 'sum',
            'risk_score': 'mean',
            'is_fraud': lambda x: (x == True).sum()
        }).reset_index()
        
        hourly_stats.columns = ['hour', 'transaction_count', 'total_amount', 'avg_risk_score', 'fraud_count']
        hourly_stats['fraud_rate'] = hourly_stats['fraud_count'] / hourly_stats['transaction_count']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Transaction Volume by Hour', 'Fraud Rate by Hour'),
            vertical_spacing=0.15
        )
        
        # Transaction volume
        fig.add_trace(
            go.Bar(
                x=hourly_stats['hour'],
                y=hourly_stats['transaction_count'],
                name='Transaction Count',
                marker_color=self.color_scheme['primary']
            ),
            row=1, col=1
        )
        
        # Fraud rate
        fig.add_trace(
            go.Scatter(
                x=hourly_stats['hour'],
                y=hourly_stats['fraud_rate'],
                mode='lines+markers',
                name='Fraud Rate',
                line=dict(color=self.color_scheme['danger']),
                marker=dict(size=8)
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
        fig.update_yaxes(title_text="Transaction Count", row=1, col=1)
        fig.update_yaxes(title_text="Fraud Rate", row=2, col=1)
        
        fig.update_layout(
            title="Transaction Patterns by Hour of Day",
            height=500,
            showlegend=False
        )
        
        return fig
    
    def create_volume_trend(self, transactions_df: pd.DataFrame) -> go.Figure:
        """Create transaction volume trend"""
        daily_volume = transactions_df.groupby(
            transactions_df['timestamp'].dt.date
        )['transaction_id'].count().reset_index()
        
        daily_volume.columns = ['date', 'transaction_count']
        
        fig = go.Figure(data=go.Scatter(
            x=daily_volume['date'],
            y=daily_volume['transaction_count'],
            mode='lines+markers',
            line=dict(color=self.color_scheme['primary'], width=3),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="Daily Transaction Volume",
            xaxis_title="Date",
            yaxis_title="Number of Transactions",
            height=400
        )
        
        return fig
    
    def create_amount_trend(self, transactions_df: pd.DataFrame) -> go.Figure:
        """Create average amount trend"""
        daily_amount = transactions_df.groupby(
            transactions_df['timestamp'].dt.date
        )['amount'].mean().reset_index()
        
        daily_amount.columns = ['date', 'avg_amount']
        
        fig = go.Figure(data=go.Scatter(
            x=daily_amount['date'],
            y=daily_amount['avg_amount'],
            mode='lines+markers',
            line=dict(color=self.color_scheme['secondary'], width=3),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="Daily Average Transaction Amount",
            xaxis_title="Date",
            yaxis_title="Average Amount ($)",
            height=400
        )
        
        return fig
    
    def create_geographic_analysis(self, transactions_df: pd.DataFrame) -> go.Figure:
        """Create geographic distribution analysis"""
        location_stats = transactions_df.groupby('location').agg({
            'transaction_id': 'count',
            'amount': 'sum',
            'is_fraud': lambda x: (x == True).sum()
        }).reset_index()
        
        location_stats.columns = ['location', 'transaction_count', 'total_amount', 'fraud_count']
        location_stats['fraud_rate'] = location_stats['fraud_count'] / location_stats['transaction_count']
        location_stats = location_stats.sort_values('transaction_count', ascending=False).head(15)
        
        fig = go.Figure(data=go.Scatter(
            x=location_stats['transaction_count'],
            y=location_stats['total_amount'],
            mode='markers',
            marker=dict(
                size=location_stats['fraud_count'] * 2 + 10,
                color=location_stats['fraud_rate'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Fraud Rate")
            ),
            text=location_stats['location'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Transactions: %{x}<br>' +
                         'Total Amount: $%{y:,.0f}<br>' +
                         'Fraud Count: %{customdata}<br>' +
                         '<extra></extra>',
            customdata=location_stats['fraud_count']
        ))
        
        fig.update_layout(
            title="Geographic Analysis: Transaction Volume vs Total Amount",
            xaxis_title="Number of Transactions",
            yaxis_title="Total Amount ($)",
            height=500
        )
        
        return fig
    
    def create_risk_score_heatmap(self, transactions_df: pd.DataFrame) -> go.Figure:
        """Create risk score heatmap by hour and day of week"""
        # Create hour and day of week columns
        df_temp = transactions_df.copy()
        df_temp['hour'] = df_temp['timestamp'].dt.hour
        df_temp['day_of_week'] = df_temp['timestamp'].dt.day_name()
        
        # Calculate average risk score for each hour-day combination
        heatmap_data = df_temp.groupby(['day_of_week', 'hour'])['risk_score'].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='risk_score')
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_pivot = heatmap_pivot.reindex(day_order)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_pivot.values,
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            colorscale='Reds',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Risk Score Heatmap by Day and Hour",
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            height=400
        )
        
        return fig
