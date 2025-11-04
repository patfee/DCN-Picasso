import os
import pandas as pd

def load_crane_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Loads outreach and height CSVs and returns a single DataFrame with columns:
    'Outreach [m]' and 'Height [m]'.

    CSVs can be one-column lists or have named columns; we just take the first column.
    """
    outreach_path = os.path.join(data_dir, "outreach.csv")
    height_path   = os.path.join(data_dir, "height.csv")

    if not os.path.exists(outreach_path):
        raise FileNotFoundError(f"Missing file: {outreach_path}")
    if not os.path.exists(height_path):
        raise FileNotFoundError(f"Missing file: {height_path}")

    outreach_df = pd.read_csv(outreach_path)
    height_df   = pd.read_csv(height_path)

    # Use first column in each CSV
    outreach_series = outreach_df.iloc[:, 0]
    height_series   = height_df.iloc[:, 0]

    # Trim to equal length (keeps things robust for now)
    n = min(len(outreach_series), len(height_series))
    outreach_series = outreach_series.iloc[:n].reset_index(drop=True)
    height_series   = height_series.iloc[:n].reset_index(drop=True)

    df = pd.DataFrame({
        "Outreach [m]": outreach_series.astype(float),
        "Height [m]": height_series.astype(float),
    })

    return df
