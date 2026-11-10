"""Convert the UCI Online Retail II workbook to the schema used by this project."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REQUIRED = {'Invoice', 'Quantity', 'InvoiceDate', 'Price', 'Customer ID'}


def convert_workbook(source, destination):
    source = Path(source)
    sheets = pd.read_excel(source, sheet_name=None)
    data = pd.concat(sheets.values(), ignore_index=True)
    data = data.rename(columns={'InvoiceNo': 'Invoice', 'UnitPrice': 'Price', 'CustomerID': 'Customer ID'})
    missing = REQUIRED - set(data.columns)
    if missing:
        raise ValueError(f'Workbook lacks required columns: {sorted(missing)}')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(destination, index=False)
    manifest = {
        'source_filename': source.name,
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'output_filename': destination.name,
        'output_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
        'rows': len(data),
        'converted_utc': datetime.now(timezone.utc).isoformat(),
    }
    destination.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return len(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workbook', help='online_retail_II.xlsx downloaded from UCI dataset 502')
    parser.add_argument('--output', default='data/online_retail_II.csv')
    args = parser.parse_args()
    print(f'Wrote {convert_workbook(args.workbook, args.output)} rows to {args.output}')
