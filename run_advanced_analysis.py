import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.advanced_analysis import AdvancedRFMAnalysis
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_rfm_features
from src.rfm_enhanced import calculate_rfm_scores, segment_customers_enhanced

def main():
    print("🚀 Starting Advanced RFM Analysis...")
    
    # Ensure directories exist
    os.makedirs("visualizations", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    try:
        # Load and preprocess data
        print("\n📊 Loading and cleaning data...")
        df = load_and_clean_data("data/online_retail_II.csv")
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
        
        # Create RFM features
        print("\n🔧 Creating RFM features...")
        rfm = create_rfm_features(df)
        
        # Calculate RFM scores
        print("\n🎯 Calculating enhanced RFM scores...")
        rfm_scored = calculate_rfm_scores(rfm)
        
        # Perform customer segmentation
        print("\n👥 Performing customer segmentation...")
        rfm_segmented, _ = segment_customers_enhanced(rfm_scored)
        
        # Run advanced analysis
        print("\n📊 Running advanced analysis...")
        analyzer = AdvancedRFMAnalysis(rfm_segmented)
        results = analyzer.run_analysis()
        
        # Generate summary report
        generate_summary_report(results)
        
        print("\n✅ Analysis complete! Check the visualizations/ and reports/ folders for results.")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

def generate_summary_report(results: dict) -> None:
    """Generate a markdown summary report of the analysis."""
    metrics = results['metrics']
    action_plan = results['action_plan']
    
    # Calculate key metrics
    total_customers = metrics['Customer_Count'].sum()
    total_revenue = metrics['Total_Revenue'].sum()
    top_segment_rev_share = metrics['Revenue_Share'].max()
    gini = 1 - 2 * np.trapz(
        np.sort(metrics['Total_Revenue'] / metrics['Total_Revenue'].sum()),
        np.linspace(0, 1, len(metrics))
    )
    
    # Create markdown content
    report = """# RFM Analysis Report

## Executive Summary

### Customer Segmentation Overview
"""
    
    # Add segment metrics
    report += "\n### Segment Metrics\n"
    report += metrics[['Customer_Count', 'Revenue_Share', 'ARPU', 'Median_Recency']].to_markdown()
    
    # Add key findings
    report += "\n\n## Key Findings\n"
    report += f"1. **Revenue Concentration**: The top segment generates {top_segment_rev_share:.1%} of total revenue.\n"
    report += f"2. **Customer Distribution**: {metrics.loc['Champions', 'Customer_Count']/total_customers:.1%} of customers are 'Champions' but generate {metrics.loc['Champions', 'Revenue_Share']:.1%} of revenue.\n"
    report += f"3. **Gini Coefficient**: {gini:.3f}, indicating {'high' if gini > 0.6 else 'moderate'} revenue concentration.\n"
    
    # Add recommendations
    report += "\n## Strategic Recommendations\n"
    report += "### 1. Champion Nurturing\n"
    report += "- Implement VIP program with exclusive benefits\n"
    report += "- Offer early access to new products\n"
    
    report += "\n### 2. Loyalty Building\n"
    report += "- Create personalized win-back campaigns for at-risk segments\n"
    report += "- Develop targeted cross-sell/upsell opportunities\n"
    
    report += "\n### 3. Revenue Diversification\n"
    report += f"- Focus on increasing share of wallet in the top {len(metrics[metrics['Revenue_Share'] > 0.1])} segments\n"
    report += "- Implement customer referral program\n"
    
    # Save report
    with open('reports/rfm_analysis_report.md', 'w') as f:
        f.write(report)
    
    print("\n📝 Generated summary report: reports/rfm_analysis_report.md")

if __name__ == "__main__":
    main()
