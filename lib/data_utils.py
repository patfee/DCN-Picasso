import pandas as pd
import os

def load_crane_data(data_dir="data"):
    """
    Load outreach and height data from CSVs and merge into one DataFrame.
    Each CSV should contain equal-length numeric columns.
    """
    outreach_path = os.path.join(data_dir, "outreach.csv")
    height_path = os.path.join(data_dir, "height.csv")

    outreach_df = pd.read_csv(outreach_path)
    height_df = pd.read_csv(height_path)

    # Normalize to same shape
    min_len = min(len(outreach_df), len(height_df))
    outreach_df = outreach_df.head(min_len)
    height_df = height_df.head(min_len)

    df = pd.DataFrame({
        "Outreach [m]": outreach_df.iloc[:, 0],
        "Height [m]": height_df.iloc[:, 0]
    })

    return df
