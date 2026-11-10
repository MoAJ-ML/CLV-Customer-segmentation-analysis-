# RFM Analysis Report

## Input snapshot

- Converted CSV: `online_retail_II.csv`
- CSV SHA-256: `32569a66f3842a82b0d8c4d63b263c5d98a76bde5d1f65c6c01bf457e541d3a9`
- Source workbook SHA-256: `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`
- Included positive-sale rows: 805,549
- RFM reference date: 2011-12-10
- Monetary unit: source currency units; no conversion applied

## Executive Summary

### Customer Segmentation Overview

### Segment Metrics
| Segment_Name        |   Customer_Count |   Revenue_Share |      ARPU |   Median_Recency |
|:--------------------|-----------------:|----------------:|----------:|-----------------:|
| Champions           |             1305 |       0.772456  | 10502.7   |               17 |
| Loyal               |             2356 |       0.18705   |  1408.71  |               71 |
| Potential Loyalists |             2217 |       0.0404936 |   324.084 |              393 |

## Key Findings
1. **Observed Positive-Sale Revenue**: The top segment accounts for 77.2% of included positive-sale revenue. Returns and nonpositive-price adjustments are excluded.
2. **Customer Distribution**: 22.2% of customers are 'Champions' but generate 77.2% of revenue.
3. **Customer-Level Gini Coefficient**: 0.742 for included positive-sale revenue.

## Interpretation limits
These labels are descriptive cluster names, not validated loyalty or churn states. The analysis does not predict CLV, prove treatment effects, or justify a specific campaign. Test any proposed intervention separately with a defined outcome and comparison group.
