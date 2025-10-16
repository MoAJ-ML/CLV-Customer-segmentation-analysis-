# Customer Lifetime Value (CLV) Analysis

This project performs advanced RFM (Recency, Frequency, Monetary) analysis to segment customers and predict customer lifetime value.

## Features

- Data preprocessing and cleaning
- RFM feature engineering
- Customer segmentation using K-means clustering
- Advanced visualization of customer segments
- Revenue concentration analysis
- Cohort retention analysis
- Actionable insights and recommendations

## Getting Started

### Prerequisites

- Python 3.8+
- Required Python packages (see `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd clv_project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

Run the analysis:
```bash
python run_advanced_analysis.py
```

## Project Structure

```
clv_project/
├── data/                   # Raw data files
├── src/                    # Source code
│   ├── advanced_analysis.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── rfm_enhanced.py
├── visualizations/         # Generated visualizations
├── reports/                # Analysis reports
├── .gitignore
├── README.md
└── requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
