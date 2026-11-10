import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import hashlib
import json
from pathlib import Path
from src.advanced_analysis import AdvancedRFMAnalysis
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_rfm_features
from src.rfm_enhanced import calculate_rfm_scores, segment_customers_enhanced

def main(data_path='data/online_retail_II.csv'):
    print("🚀 Starting Advanced RFM Analysis...")
    
    # Ensure directories exist
    os.makedirs("visualizations", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Missing {data_path}. See README.md for dataset acquisition or use --data with a CSV path.')
    try:
        # Load and preprocess data
        print("\n📊 Loading and cleaning data...")
        df = load_and_clean_data(data_path)
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
        with open(data_path, 'rb') as source:
            input_sha256 = hashlib.file_digest(source, 'sha256').hexdigest()
        manifest_path = Path(data_path).with_suffix('.manifest.json')
        source_sha256 = None
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('output_sha256') == input_sha256:
                source_sha256 = manifest.get('source_sha256')
        generate_summary_report(results, {
            'input_name': Path(data_path).name,
            'input_sha256': input_sha256,
            'workbook_sha256': source_sha256,
            'included_rows': len(df),
            'reference_date': str((df['InvoiceDate'].max() + pd.Timedelta(days=1)).date()),
        })
        
        print("\n✅ Analysis complete! Check the visualizations/ and reports/ folders for results.")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def generate_summary_report(results: dict, snapshot: dict) -> None:
    """Generate a markdown summary report of the analysis."""
    metrics = results['metrics']
    
    # Calculate key metrics
    total_customers = metrics['Customer_Count'].sum()
    top_segment_rev_share = metrics['Revenue_Share'].max()
    gini = results['customer_gini']
    
    # Create markdown content
    report = f"""# RFM Analysis Report

## Input snapshot

- Converted CSV: `{snapshot['input_name']}`
- CSV SHA-256: `{snapshot['input_sha256']}`
- Source workbook SHA-256: `{snapshot['workbook_sha256'] or 'not recorded'}`
- Included positive-sale rows: {snapshot['included_rows']:,}
- RFM reference date: {snapshot['reference_date']}
- Monetary unit: source currency units; no conversion applied

## Executive Summary

### Customer Segmentation Overview
"""
    
    # Add segment metrics
    report += "\n### Segment Metrics\n"
    report += metrics[['Customer_Count', 'Revenue_Share', 'ARPU', 'Median_Recency']].to_markdown()
    
    # Add key findings
    report += "\n\n## Key Findings\n"
    report += f"1. **Observed Positive-Sale Revenue**: The top segment accounts for {top_segment_rev_share:.1%} of included positive-sale revenue. Returns and nonpositive-price adjustments are excluded.\n"
    report += f"2. **Customer Distribution**: {metrics.loc['Champions', 'Customer_Count']/total_customers:.1%} of customers are 'Champions' but generate {metrics.loc['Champions', 'Revenue_Share']:.1%} of revenue.\n"
    report += f"3. **Customer-Level Gini Coefficient**: {gini:.3f} for included positive-sale revenue.\n"
    
    report += "\n## Interpretation limits\n"
    report += "These labels are descriptive cluster names, not validated loyalty or churn states. "
    report += "The analysis does not predict CLV, prove treatment effects, or justify a specific campaign. "
    report += "Test any proposed intervention separately with a defined outcome and comparison group.\n"
    
    # Save report
    with open('reports/rfm_analysis_report.md', 'w') as f:
        f.write(report)
    
    print("\n📝 Generated summary report: reports/rfm_analysis_report.md")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/online_retail_II.csv')
    main(parser.parse_args().data)
