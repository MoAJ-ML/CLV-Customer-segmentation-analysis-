import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from prepare_data import convert_workbook
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_rfm_features


class RFMTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
