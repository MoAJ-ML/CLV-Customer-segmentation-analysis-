# Online Retail II RFM Segmentation

This project cleans transaction data, computes recency, invoice frequency and monetary value per customer, clusters those features, and describes segment revenue. It does **not** predict customer lifetime value or calculate a time-based retention cohort. The report and charts were regenerated on 10 November 2026 using the workbook snapshot identified below; the old cohort plot was removed.

## Data

Download [UCI Online Retail II, dataset 502](https://archive.ics.uci.edu/dataset/502/online%2Bretail%2Bii), specifically `online_retail_II.xlsx`. UCI credits Daqing Chen and lists the dataset under CC BY 4.0; retain the citation and check the publisher page before redistribution. The large workbook/CSV is intentionally absent from this repository. Do not substitute the distinct Online Retail dataset 352.

```bash
pip install -r requirements.txt
mkdir -p data
curl -fL --max-time 600 'https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip' -o data/online_retail_ii.zip
unzip -o data/online_retail_ii.zip -d data
python prepare_data.py data/online_retail_II.xlsx
python run_advanced_analysis.py --data data/online_retail_II.csv
python -m unittest discover -s tests
```

`prepare_data.py` combines workbook sheets and normalizes the required column names. It writes a JSON manifest with SHA-256 hashes of the workbook and output CSV; record the retrieval date separately. The analysis drops missing customer IDs, nonpositive quantities, and nonpositive prices, then uses the last observed transaction date plus one day as the RFM reference date. Returns/cancellations are excluded; reported revenue is positive observed sales, not net revenue. Monetary values are clipped for scoring and clustering, while revenue, ARPU, and Gini use the original positive-sale amounts. Segments are descriptive clusters with business-friendly labels, not validated behavioral or predictive categories.

A synthetic fixture is included under `tests/fixtures` for a clean-clone preprocessing check; `python run_advanced_analysis.py --data tests/fixtures/retail_sample.csv` also passed end to end. The fixture is not a substitute for the UCI workbook. The real-workbook rerun used workbook SHA-256 `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980` and converted CSV SHA-256 `32569a66f3842a82b0d8c4d63b263c5d98a76bde5d1f65c6c01bf457e541d3a9`. It retained 805,549 positive-sale rows across 5,878 customers, totaling 17,743,429.18 in source currency units; the report's segment totals matched that cleaned amount exactly. The customer-level Gini was 0.742. The workbook and CSV are not redistributed here.

The generated action-plan files contain **unvalidated ideas**, not measured treatment effects or recommended interventions. Cluster labels such as “Champions” are descriptive names based partly on spend and do not establish loyalty or churn status.

No software license has been granted for this repository. The UCI dataset's CC BY 4.0 license covers the data, not necessarily this project's code. Suggested repository description: “RFM customer segmentation on UCI Online Retail II with reproducible data conversion and descriptive segment reports.”
