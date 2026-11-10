# Online Retail II RFM Segmentation

This project cleans transaction data, computes recency, invoice frequency and monetary value per customer, clusters those features, and describes segment revenue. It does **not** predict customer lifetime value or calculate a time-based retention cohort. The committed reports and plots are historical outputs until reproduced with the input snapshot that generated them.

## Data

Download [UCI Online Retail II, dataset 502](https://archive.ics.uci.edu/dataset/502/online%2Bretail%2Bii), specifically `online_retail_II.xlsx`. UCI credits Daqing Chen and lists the dataset under CC BY 4.0; retain the citation and check the publisher page before redistribution. The large workbook/CSV is intentionally absent from this repository. Do not substitute the distinct Online Retail dataset 352.

```bash
pip install -r requirements.txt
python prepare_data.py /path/to/online_retail_II.xlsx
python run_advanced_analysis.py --data data/online_retail_II.csv
python -m unittest discover -s tests
```

`prepare_data.py` combines workbook sheets and normalizes the required column names. It writes a JSON manifest with SHA-256 hashes of the workbook and output CSV; record the retrieval date separately. The analysis drops missing customer IDs and nonpositive quantities, then uses the last observed transaction date plus one day as the RFM reference date. Returns/cancellations are excluded; this can overstate net revenue. Segments are descriptive clusters with business-friendly labels, not validated behavioral or predictive categories.

A synthetic fixture is included under `tests/fixtures` for a clean-clone preprocessing check; `python run_advanced_analysis.py --data tests/fixtures/retail_sample.csv` also passed end to end. The fixture is not a substitute for the UCI workbook or for rerunning the full report. The existing report's previous Gini formula used segment totals; the script now calculates it over customer monetary values. A real-workbook rerun was not completed because the public archive download timed out.

No software license has been granted for this repository. The UCI dataset's CC BY 4.0 license covers the data, not necessarily this project's code. Suggested repository description: “RFM customer segmentation on UCI Online Retail II with reproducible data conversion and descriptive segment reports.”
