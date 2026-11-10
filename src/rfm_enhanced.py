import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
import seaborn as sns
import matplotlib.pyplot as plt
import os
from typing import Tuple

def winsorize_column(series: pd.Series, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> pd.Series:
    """Apply winsorization to handle outliers."""
    lower = series.quantile(lower_quantile)
    upper = series.quantile(upper_quantile)
    return np.clip(series, lower, upper)

def calculate_rfm_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    """Calculate enhanced RFM scores with winsorization and log transforms."""
    # Make a copy to avoid modifying the original
    rfm = rfm.copy()
    
    # Preserve observed positive-sale revenue for reports and concentration metrics.
    rfm['Monetary_raw'] = rfm['Monetary']
    # Use clipped monetary values only for scores and clustering.
    rfm['Monetary'] = winsorize_column(rfm['Monetary'])
    
    # Cap recency at 700 days (consider older as churned)
    rfm['Recency'] = np.minimum(rfm['Recency'], 700)
    
    # Calculate scores with rank-based approach to handle duplicates
    def rank_based_score(series, n_bins=5):
        # Use rank with method='first' to handle ties
        ranks = series.rank(method='first', pct=True)
        # Convert to 1-5 scale
        return pd.cut(ranks, bins=n_bins, labels=range(1, n_bins+1), include_lowest=True)
    
    # Apply rank-based scoring
    rfm['R_score'] = rank_based_score(-rfm['Recency'])
    rfm['F_score'] = rank_based_score(np.log1p(rfm['Frequency']))
    rfm['M_score'] = rank_based_score(np.log1p(rfm['Monetary']))
    
    # Convert scores to numeric
    for col in ['R_score', 'F_score', 'M_score']:
        rfm[col] = pd.to_numeric(rfm[col])
    
    # Calculate overall RFM score (weighted if needed)
    rfm['RFM_Score'] = rfm['R_score'] * 0.5 + rfm['F_score'] * 0.25 + rfm['M_score'] * 0.25
    
    return rfm

def segment_customers_enhanced(rfm: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Enhanced customer segmentation with business-friendly naming."""
    # Make a copy to avoid modifying the original
    rfm = rfm.copy()
    
    # Scale features with robust scaling
    features = rfm[['Recency', 'Frequency', 'Monetary']].copy()
    
    # Apply log transformation to reduce skew
    features = np.log1p(features)
    
    # Robust scaling (less sensitive to outliers)
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Determine optimal number of clusters using Elbow method
    distortions = []
    K = range(2, min(6, len(rfm)))
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features_scaled)
        distortions.append(kmeans.inertia_)
    
    # Find the elbow point (simplified approach)
    n_clusters = 5  # Default
    if len(distortions) > 1:
        # Calculate the rate of decrease of distortion
        deltas = np.diff(distortions)
        if len(deltas) > 0:
            # Find the point where the decrease slows down the most
            relative_deltas = np.abs(deltas) / np.max(np.abs(deltas))
            n_clusters = np.argmax(relative_deltas) + 2  # +2 because we started from k=2
            n_clusters = min(max(3, n_clusters), 5)
    n_clusters = min(n_clusters, len(rfm))
    
    # Final clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['Segment'] = kmeans.fit_predict(features_scaled)
    
    # Sort segments by monetary value (descending)
    segment_means = rfm.groupby('Segment')['Monetary'].mean().sort_values(ascending=False)
    segment_names = [
        'Champions',
        'Loyal',
        'Potential Loyalists',
        'At Risk',
        'Big Spenders',
    ][:n_clusters]
    
    # Create mapping from segment number to name, ordered by monetary value
    segment_map = {int(segment_means.index[i]): name for i, name in enumerate(segment_names)}
    rfm['Segment_Name'] = rfm['Segment'].map(segment_map)
    
    return rfm, segment_map

def generate_segment_analysis(rfm: pd.DataFrame) -> None:
    """Generate and save segment analysis visualizations."""
    os.makedirs("../visualizations", exist_ok=True)
    
    # Segment size distribution
    plt.figure(figsize=(10, 6))
    segment_counts = rfm['Segment_Name'].value_counts()
    sns.barplot(x=segment_counts.index, y=segment_counts.values)
    plt.title('Customer Segment Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../visualizations/segment_distribution.png')
    plt.close()
    
    # Monetary value by segment
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Segment_Name', y='Monetary', data=rfm)
    plt.title('Monetary Value by Segment')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../visualizations/monetary_by_segment.png')
    plt.close()

def generate_segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics for each segment."""
    summary = rfm.groupby('Segment_Name').agg({
        'CustomerID': 'count',
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'sum']
    }).round(2)
    
    # Calculate percentages
    summary[('CustomerID', 'percentage')] = (summary[('CustomerID', 'count')] / len(rfm) * 100).round(1)
    summary[('Monetary', 'revenue_share')] = (summary[('Monetary', 'sum')] / rfm['Monetary'].sum() * 100).round(1)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    # Save to CSV
    summary.to_csv('../visualizations/segment_summary.csv')
    
    return summary
