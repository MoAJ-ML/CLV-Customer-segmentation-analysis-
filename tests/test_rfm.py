import unittest
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from prepare_data import convert_workbook
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_rfm_features
from src.rfm_enhanced import calculate_rfm_scores
from src.advanced_analysis import AdvancedRFMAnalysis


class RFMTests(unittest.TestCase):
    def test_nonpositive_price_adjustments_are_excluded(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'transactions.csv'
            pd.DataFrame({
                'Invoice': ['A', 'B', 'C'],
                'Quantity': [1, 1, 1],
                'InvoiceDate': ['2010-01-01'] * 3,
                'Price': [5, 0, -2],
                'Customer ID': [1, 2, 3],
            }).to_csv(path, index=False)
            clean = load_and_clean_data(path)
        self.assertEqual(len(clean), 1)
        self.assertEqual(clean['TotalPrice'].sum(), 5)

    def test_workbook_conversion_clean_run(self):
        with TemporaryDirectory() as folder:
            source = Path(folder) / 'retail.xlsx'
            output = Path(folder) / 'retail.csv'
            sample = pd.read_csv('tests/fixtures/retail_sample.csv')
            with pd.ExcelWriter(source) as writer:
                sample.iloc[:3].to_excel(writer, sheet_name='Year 1', index=False)
                sample.iloc[3:].to_excel(writer, sheet_name='Year 2', index=False)
            self.assertEqual(convert_workbook(source, output), len(sample))
            self.assertEqual(len(pd.read_csv(output)), len(sample))
            self.assertTrue(output.with_suffix('.manifest.json').exists())

    def test_clean_clone_fixture_and_invoice_frequency(self):
        data = load_and_clean_data(Path('tests/fixtures/retail_sample.csv'))
        self.assertEqual(len(data), 4)
        rfm = create_rfm_features(data).set_index('CustomerID')
        self.assertEqual(rfm.loc[1001, 'Frequency'], 2)
        self.assertEqual(rfm.loc[1002, 'Frequency'], 1)
        self.assertEqual(rfm.loc[1001, 'Monetary'], 17)

    def test_report_revenue_uses_observed_sales_not_winsorized_scores(self):
        frame = pd.DataFrame({
            'CustomerID': [1, 2, 3],
            'Recency': [1, 2, 3],
            'Frequency': [1, 2, 3],
            'Monetary': [10.0, 20.0, 1000.0],
            'Segment_Name': ['Champions', 'Loyal', 'At Risk'],
        })
        scored = calculate_rfm_scores(frame)
        self.assertEqual(scored['Monetary_raw'].sum(), 1030.0)
        with TemporaryDirectory() as folder:
            previous = os.getcwd()
            os.chdir(folder)
            try:
                analyzer = AdvancedRFMAnalysis(scored)
                analyzer.preprocess_monetary()
                metrics = analyzer.calculate_segment_metrics()
            finally:
                os.chdir(previous)
        self.assertEqual(metrics['Total_Revenue'].sum(), 1030.0)


if __name__ == '__main__':
    unittest.main()
