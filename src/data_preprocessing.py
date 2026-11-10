import pandas as pd

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, encoding='unicode_escape')

    # Drop missing customer IDs
    df = df.dropna(subset=['Customer ID']).copy()

    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Remove negative quantities (returns)
    df = df[df['Quantity'] > 0]

    # Create TotalPrice column
    df['TotalPrice'] = df['Quantity'] * df['Price']

    return df
