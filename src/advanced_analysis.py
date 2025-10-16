import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from typing import Dict, Tuple
import os

class AdvancedRFMAnalysis:
    def __init__(self, rfm: pd.DataFrame):
        """Initialize with RFM DataFrame."""
        self.rfm = rfm.copy()
        # Define consistent segment names and colors
        self.segment_names = [
            'Champions', 
            'Loyal', 
            'Potential Loyalists', 
            'At Risk', 
            'Big Spenders'
        ]
        
        self.segment_colors = {
            'Champions': '#2ecc71',
            'Loyal': '#3498db',
            'Potential Loyalists': '#f1c40f',
            'At Risk': '#e74c3c',
            'Big Spenders': '#9b59b6'
        }
        
        # Set style for all plots
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'seaborn')
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'figure.figsize': (10, 6)
        })
        
        # Ensure output directories exist
        os.makedirs('visualizations', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        
    def preprocess_monetary(self) -> None:
        """Apply winsorization and log transform to monetary values."""
        # Winsorize to handle outliers
        self.rfm['Monetary_w'] = winsorize(
            self.rfm['Monetary'], 
            limits=[0, 0.01]
        )
        # Log transform for visualization
        self.rfm['log_Monetary'] = np.log1p(self.rfm['Monetary_w'])
    
    def calculate_segment_metrics(self) -> pd.DataFrame:
        """
        Calculate comprehensive metrics for each segment.
        Returns:
            DataFrame with segment metrics including customer count, revenue, ARPU, AOV, etc.
        """
        # Ensure consistent segment ordering
        self.rfm['Segment_Name'] = pd.Categorical(
            self.rfm['Segment_Name'], 
            categories=self.segment_names,
            ordered=True
        )
        
        # Calculate metrics
        metrics = self.rfm.groupby('Segment_Name', observed=True).agg({
            'CustomerID': 'nunique',
            'Frequency': ['sum', 'mean'],  # Using Frequency for total orders (sum) and average frequency (mean)
            'Monetary_w': ['sum', 'median', 'mean'],
            'Recency': 'median'
        })
        
        # Flatten multi-index columns
        metrics.columns = ['_'.join(col).strip() for col in metrics.columns.values]
        metrics = metrics.rename(columns={
            'CustomerID_nunique': 'Customer_Count',
            'Frequency_sum': 'Total_Orders',  # Total number of orders
            'Frequency_mean': 'Avg_Frequency',
            'Monetary_w_sum': 'Total_Revenue',
            'Monetary_w_median': 'Median_Revenue',
            'Monetary_w_mean': 'Mean_Revenue',
            'Recency_median': 'Median_Recency'
        })
        
        # Calculate derived metrics
        metrics['Revenue_Share'] = metrics['Total_Revenue'] / metrics['Total_Revenue'].sum()
        metrics['ARPU'] = metrics['Total_Revenue'] / metrics['Customer_Count']
        metrics['AOV'] = metrics['Total_Revenue'] / metrics['Total_Orders']
        metrics['Orders_per_Customer'] = metrics['Total_Orders'] / metrics['Customer_Count']
        
        # Reorder columns for better readability
        metrics = metrics[[
            'Customer_Count', 'Total_Orders', 'Total_Revenue', 
            'Revenue_Share', 'ARPU', 'AOV', 'Median_Revenue',
            'Mean_Revenue', 'Avg_Frequency', 'Median_Recency',
            'Orders_per_Customer'
        ]]
        
        # Save detailed metrics to CSV
        metrics.to_csv('visualizations/segment_kpi_detailed.csv')
        
        # Create and save simplified summary
        summary = metrics[['Customer_Count', 'Revenue_Share', 'ARPU', 'Median_Recency']]
        summary.to_csv('visualizations/segment_kpi_summary.csv')
        
        return metrics
    
    def plot_segment_monetaries(self) -> None:
        """
        Create enhanced boxplot of segment monetaries with improved styling.
        Shows winsorized, log-transformed monetary values by segment.
        """
        plt.figure(figsize=(10, 6))
        
        # Create boxplot with enhanced styling
        ax = sns.boxplot(
            x='Segment_Name',
            y='log_Monetary',
            data=self.rfm,
            palette=self.segment_colors,
            order=self.segment_names,
            showfliers=False,  # Hide outliers for cleaner visualization
            width=0.6,
            linewidth=1.5
        )
        
        # Add median value annotations
        medians = self.rfm.groupby('Segment_Name')['log_Monetary'].median().reindex(self.segment_names)
        for i, median in enumerate(medians):
            if not np.isnan(median):  # Only add annotation if median is not NaN
                ax.text(i, median, f'${int(np.expm1(median)):,}', 
                       horizontalalignment='center',
                       size=9, color='black', weight='semibold')
        
        # Format y-axis with dollar amounts
        y_ticks = np.linspace(
            self.rfm['log_Monetary'].min(),
            self.rfm['log_Monetary'].max(),
            num=6
        )
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'${int(np.expm1(y)):,}' for y in y_ticks])
        
        # Enhance plot appearance
        plt.title('Customer Segment Monetaries\n(Winsorized & Log-Transformed)', 
                 fontsize=14, pad=20, weight='bold')
        plt.xlabel('')
        plt.ylabel('Spending per Customer (log scale)')
        plt.xticks(rotation=45, ha='right')
        
        # Add grid for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add descriptive caption
        plt.figtext(0.5, -0.15, 
                   "Note: Monetary values are winsorized at the 99th percentile and log-transformed for visualization.",
                   ha="center", fontsize=8, style='italic')
        
        plt.tight_layout()
        plt.savefig('visualizations/final_segment_monetaries.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_segment_metrics(self, metrics: pd.DataFrame) -> None:
        """
        Create a comprehensive segment metrics dashboard with enhanced visualizations.
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(3, 4)
        
        # 1. Customer Count (Top Left)
        ax1 = fig.add_subplot(gs[0, :2])
        sns.barplot(
            x=metrics.index,
            y='Customer_Count',
            data=metrics,
            palette=self.segment_colors,
            ax=ax1
        )
        ax1.set_title('Customer Count by Segment', fontweight='bold')
        ax1.set_xlabel('')
        ax1.set_ylabel('Number of Customers')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on top of bars
        for p in ax1.patches:
            ax1.annotate(f'{int(p.get_height()):,}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5),
                        textcoords='offset points')
        
        # 2. Revenue Share (Top Right)
        ax2 = fig.add_subplot(gs[0, 2:])
        explode = [0.05] * len(metrics)
        wedges, texts, autotexts = ax2.pie(
            metrics['Revenue_Share'],
            labels=metrics.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=[self.segment_colors[s] for s in metrics.index],
            explode=explode,
            wedgeprops=dict(width=0.6, edgecolor='w'),
            pctdistance=0.85
        )
        ax2.set_title('Revenue Share by Segment', fontweight='bold')
        
        # 3. ARPU (Middle Row)
        ax3 = fig.add_subplot(gs[1, :])
        sns.barplot(
            x=metrics.index,
            y='ARPU',
            data=metrics,
            palette=self.segment_colors,
            ax=ax3
        )
        ax3.set_title('Average Revenue Per User (ARPU)', fontweight='bold')
        ax3.set_xlabel('')
        ax3.set_ylabel('ARPU ($)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Format y-axis as currency
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f'${x:,.0f}'))
        
        # 4. Key Metrics Table (Bottom)
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # Prepare table data
        table_data = metrics[['Customer_Count', 'Revenue_Share', 'ARPU', 'AOV', 'Median_Recency']].copy()
        table_data['Revenue_Share'] = table_data['Revenue_Share'].apply(lambda x: f'{x:.1%}')
        table_data['ARPU'] = table_data['ARPU'].apply(lambda x: f'${x:,.2f}')
        table_data['AOV'] = table_data['AOV'].apply(lambda x: f'${x:,.2f}')
        table_data['Median_Recency'] = table_data['Median_Recency'].apply(lambda x: f'{int(x)} days')
        
        # Create table
        table = ax4.table(
            cellText=table_data.values,
            rowLabels=table_data.index,
            colLabels=['Customers', 'Rev. Share', 'ARPU', 'AOV', 'Recency'],
            cellLoc='center',
            loc='center',
            colColours=['#f3f3f3'] * 5,
            rowColours=[self.segment_colors[s] for s in metrics.index]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Add title for the table
        plt.figtext(0.5, 0.1, 'Segment Performance Metrics', 
                   ha='center', fontsize=12, fontweight='bold')
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.suptitle('Customer Segment Analysis Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Add timestamp
        plt.figtext(0.5, 0.02, f'Generated on {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}', 
                   ha='center', fontsize=8, style='italic')
        
        plt.savefig('visualizations/segment_metrics_dashboard.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def calculate_gini(self, values: np.ndarray) -> float:
        """
        Calculate Gini coefficient for revenue concentration.
        
        Args:
            values: Array of monetary values
            
        Returns:
            float: Gini coefficient (0 = perfect equality, 1 = maximum inequality)
        """
        # Sort values if not already sorted
        values = np.sort(values)
        n = len(values)
        
        # Calculate cumulative values and areas
        cum_values = np.cumsum(values)
        cum_share = cum_values / cum_values[-1]  # Cumulative share of total value
        
        # Calculate Gini using the area under the Lorenz curve
        # Gini = (A) / (A + B) where A + B = 0.5
        # So Gini = 1 - 2 * (area under Lorenz curve)
        lorenz_area = np.trapz(cum_share, dx=1/n)
        gini = 1 - 2 * lorenz_area
        
        return gini
    
    def plot_revenue_concentration(self) -> None:
        """
        Plot enhanced Lorenz curve for revenue concentration analysis.
        Shows the distribution of revenue across customers and highlights key metrics.
        """
        # Sort customers by monetary value (ascending)
        rfm_sorted = self.rfm.sort_values('Monetary_w')
        
        # Calculate cumulative percentages
        n = len(rfm_sorted)
        rfm_sorted['cumulative_pct_customers'] = np.arange(1, n + 1) / n
        rfm_sorted['cumulative_pct_revenue'] = rfm_sorted['Monetary_w'].cumsum() / rfm_sorted['Monetary_w'].sum()
        
        # Calculate Gini coefficient
        gini = self.calculate_gini(rfm_sorted['Monetary_w'])
        
        # Find the point where 80% of revenue is reached
        pct_customers_at_80pct_revenue = rfm_sorted.loc[
            rfm_sorted['cumulative_pct_revenue'] >= 0.8, 
            'cumulative_pct_customers'
        ].iloc[0]
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Plot perfect equality line
        plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.7, 
                label='Line of Perfect Equality')
        
        # Plot the Lorenz curve
        plt.plot(rfm_sorted['cumulative_pct_customers'], 
                rfm_sorted['cumulative_pct_revenue'],
                linewidth=2.5, color='#3498db',
                label=f'Actual Revenue (Gini = {gini:.3f})')
        
        # Add 80/20 line and annotation
        plt.axvline(x=pct_customers_at_80pct_revenue, color='r', linestyle=':', alpha=0.7)
        plt.axhline(y=0.8, color='r', linestyle=':', alpha=0.7)
        plt.scatter(pct_customers_at_80pct_revenue, 0.8, 
                   color='red', zorder=5,
                   label=f'Top {pct_customers_at_80pct_revenue:.1%} generate 80% revenue')
        
        # Add 50/50 line if it's meaningful
        if (rfm_sorted['cumulative_pct_revenue'] >= 0.5).any():
            pct_at_50 = rfm_sorted.loc[
                rfm_sorted['cumulative_pct_revenue'] >= 0.5,
                'cumulative_pct_customers'
            ].iloc[0]
            plt.scatter(pct_at_50, 0.5, color='green', zorder=5,
                       label=f'Top {pct_at_50:.1%} generate 50% revenue')
        
        # Enhance plot appearance
        plt.title('Revenue Concentration Analysis\n(Lorenz Curve)', 
                 fontsize=14, pad=20, fontweight='bold')
        plt.xlabel('Cumulative Percentage of Customers', fontsize=12)
        plt.ylabel('Cumulative Percentage of Revenue', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Add legend and annotations
        plt.legend(loc='upper left', frameon=True, framealpha=0.9)
        
        # Add Gini coefficient explanation
        plt.figtext(0.5, -0.1,
                   f"Gini Coefficient: {gini:.3f} (0 = perfect equality, 1 = maximum inequality)",
                   ha='center', fontsize=10, style='italic')
        
        # Ensure proper layout and save
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    def create_action_plan(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a comprehensive action plan for each customer segment.
        
        Args:
            metrics: DataFrame containing segment metrics
            
        Returns:
            DataFrame with detailed action plans for each segment
        """
        action_plan = {
            'Champions': {
                'Segment Description': 'Your best customers who buy often and spend the most',
                'Key Characteristics': 'High recency, frequency, and monetary values',
                'Strategic Priority': 'Retention & Loyalty',
                'Primary Goal': 'Increase share of wallet and lifetime value',
                'Recommended Actions': [
                    'Implement VIP program with exclusive benefits',
                    'Offer early access to new products',
                    'Create personalized product recommendations',
                    'Develop a dedicated loyalty program tier'
                ],
                'Channel Strategy': [
                    'Personalized email campaigns',
                    'Dedicated account management',
                    'Exclusive member portal',
                    'Priority customer support'
                ],
                'Offer Strategy': [
                    'Free expedited shipping',
                    'Exclusive product bundles',
                    'Birthday/anniversary rewards',
                    'Invitation to VIP events'
                ],
                'Success Metrics': [
                    'Repeat purchase rate',
                    'Average order value',
                    'Customer lifetime value',
                    'Net Promoter Score (NPS)'
                ]
            },
            'Loyal': {
                'Segment Description': 'Frequent buyers with moderate spending',
                'Key Characteristics': 'Good purchase frequency but lower average order value',
                'Strategic Priority': 'Upsell & Cross-sell',
                'Primary Goal': 'Increase average order value and frequency',
                'Recommended Actions': [
                    'Upsell to premium products',
                    'Create product bundles',
                    'Implement volume discounts',
                    'Develop subscription options'
                ],
                'Channel Strategy': [
                    'Targeted email campaigns',
                    'Retargeting ads',
                    'In-app recommendations',
                    'Personalized offers'
                ],
                'Offer Strategy': [
                    'Bundle discounts',
                    'Free shipping thresholds',
                    'Limited-time promotions',
                    'Exclusive member pricing'
                ],
                'Success Metrics': [
                    'Average order value growth',
                    'Purchase frequency',
                    'Upsell conversion rate',
                    'Customer retention rate'
                ]
            },
            'Potential Loyalists': {
                'Segment Description': 'New or occasional customers showing potential',
                'Key Characteristics': 'Recent first purchase or few purchases',
                'Strategic Priority': 'Engagement & Conversion',
                'Primary Goal': 'Increase purchase frequency and loyalty',
                'Recommended Actions': [
                    'Welcome series for new customers',
                    'Post-purchase follow-up',
                    'Educational content about products',
                    'Customer feedback collection'
                ],
                'Channel Strategy': [
                    'Drip email campaigns',
                    'Social media engagement',
                    'SMS marketing',
                    'Retargeting campaigns'
                ],
                'Offer Strategy': [
                    'First-purchase discount',
                    'Referral bonuses',
                    'Free samples with next purchase',
                    'Limited-time welcome offers'
                ],
                'Success Metrics': [
                    'Repeat purchase rate',
                    'Email open/click rates',
                    'Conversion rate',
                    'Customer satisfaction score'
                ]
            },
            'At Risk': {
                'Segment Description': 'Previously active but haven\'t purchased recently',
                'Key Characteristics': 'Declining purchase frequency',
                'Strategic Priority': 'Win-back & Reactivation',
                'Primary Goal': 'Regain engagement and prevent churn',
                'Recommended Actions': [
                    'Win-back email sequence',
                    'Special reactivation offers',
                    'Customer feedback surveys',
                    'Personalized recommendations'
                ],
                'Channel Strategy': [
                    'Targeted win-back emails',
                    'Retargeting ads',
                    'Direct mail',
                    'SMS re-engagement'
                ],
                'Offer Strategy': [
                    'Exclusive comeback discount',
                    'Free shipping on next order',
                    'Limited-time reactivation offer',
                    'Loyalty points bonus'
                ],
                'Success Metrics': [
                    'Reactivation rate',
                    'Email engagement',
                    'Repeat purchase rate',
                    'Customer lifetime value'
                ]
            },
            'Big Spenders': {
                'Segment Description': 'High-value but infrequent customers',
                'Key Characteristics': 'High average order value, low frequency',
                'Strategic Priority': 'Frequency & Engagement',
                'Primary Goal': 'Increase purchase frequency',
                'Recommended Actions': [
                    'Develop subscription options',
                    'Create product replenishment reminders',
                    'Offer exclusive early access',
                    'Provide bulk purchase incentives'
                ],
                'Channel Strategy': [
                    'Personalized email outreach',
                    'Account-based marketing',
                    'Direct sales contact',
                    'Exclusive event invitations'
                ],
                'Offer Strategy': [
                    'Volume discounts',
                    'Subscription benefits',
                    'Exclusive product bundles',
                    'Complimentary premium services'
                ],
                'Success Metrics': [
                    'Purchase frequency',
                    'Customer lifetime value',
                    'Repeat purchase rate',
                    'Engagement with communications'
                ]
            }
        }
            
        # Convert to DataFrame and reindex to match segment order
        action_plan_df = pd.DataFrame.from_dict(action_plan, orient='index')
            
        # Ensure all segments in metrics have an action plan
        for segment in metrics.index:
            if segment not in action_plan_df.index:
                action_plan_df.loc[segment] = {
                    'Segment Description': 'Segment not found in action plan',
                    'Key Characteristics': 'N/A',
                    'Strategic Priority': 'N/A',
                    'Primary Goal': 'N/A',
                    'Recommended Actions': [],
                    'Channel Strategy': [],
                    'Offer Strategy': [],
                    'Success Metrics': []
                }
            
        # Reindex to match the order in metrics
        action_plan_df = action_plan_df.reindex(metrics.index)
            
        # Save to CSV and Excel for better formatting
        action_plan_df.to_csv('visualizations/segment_action_plan.csv')
        
        # Create a more readable version for Excel with line breaks
        excel_df = action_plan_df.copy()
        for col in ['Recommended Actions', 'Channel Strategy', 
                   'Offer Strategy', 'Success Metrics']:
            excel_df[col] = excel_df[col].apply(
                lambda x: '\n• ' + '\n• '.join(x) if isinstance(x, list) else x
            )
        
        excel_df.to_excel('reports/segment_action_plan.xlsx', 
                         sheet_name='Segment Action Plan',
                         index_label='Segment')
        
        return action_plan_df

    def run_analysis(self) -> Dict[str, pd.DataFrame]:
        """Run complete analysis pipeline."""
        # Ensure visualizations directory exists
        os.makedirs('visualizations', exist_ok=True)
        
        # Preprocess data
        self.preprocess_monetary()
        
        # Calculate metrics
        metrics = self.calculate_segment_metrics()
        
        # Generate visualizations
        self.plot_segment_monetaries()
        self.plot_segment_metrics(metrics)
        self.plot_revenue_concentration()
        
        # Create action plan
        action_plan = self.create_action_plan(metrics)
        
        # Save metrics
        metrics.to_csv('visualizations/segment_metrics_detailed.csv')
        
        return {
            'metrics': metrics,
            'action_plan': action_plan
        }
