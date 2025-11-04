import os
import pandas as pd
import csv

def load_crane_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Loads outreach and height CSVs, auto-detects delimiters (; , or tab),
    and returns a single tidy DataFrame with 'Outreach [m]' and 'Height [m]'.
    """

    def read_csv_safely(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        # Auto-detect delimiter
        with open(path, "r", newline="") as f:
            sample = f.read(2048)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter if ";" in sample or "," in sample else ";"

        df = pd.read_csv(path, sep=delimiter, engine="python", header=None)

        # If it’s a single row (semicolon-separated numbers), transpose
        if df.shape[0] == 1 and df.shape[1] > 1:
            df = df.T

        # Take first column and drop NaNs
        series = df.iloc[:, 0].dropna().astype(str).str.replace(",", ".")
        return pd.to_numeric(series, errors="coerce").dropna().reset_index(drop=True)

    outreach_path = os.path.join(data_dir, "outreach.csv")
    height_path = os.path.join(data_dir, "height.csv")

    outreach_series = read_csv_safely(outreach_path)
    height_series = read_csv_safely(height_path)

    # Equalize length
    n = min(len(outreach_series), len(height_series))
    outreach_series = outreach_series.iloc[:n]
    height_series = height_series.iloc[:n]

    df = pd.DataFrame({
        "Outreach [m]": outreach_series,
        "Height [m]": height_series,
    })

    return df
