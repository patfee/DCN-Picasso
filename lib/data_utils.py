from functools import lru_cache
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

@lru_cache(maxsize=32)
def load_csv(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame({"info": [f"CSV not found: {filename}"]})
    return pd.read_csv(path, encoding="utf-8")

def _is_numberlike(s: str) -> bool:
    try:
        float(str(s).strip().replace("°",""))
        return True
    except Exception:
        return False

@lru_cache(maxsize=32)
def load_matrix_csv(filename: str) -> pd.DataFrame:
    """
    Load a matrix CSV from /data where:
      - Columns are Main-jib angles (header row)
      - Rows are Folding-jib angles (first column)
    Returns a DataFrame indexed by Folding-jib angle (float),
    with columns as Main-jib angle (float).
    """
    df = load_csv(filename)
    if df.empty:
        return df

    # First column = folding angles; header row = main angles
    # Try to coerce both to floats (strip degree symbols if present)
    # Handle cases where CSV includes an unnamed angle column name
    df = df.copy()
    # Find the first column that looks like angle list (row-wise)
    angle_col = df.columns[0]
    # Clean folding angles
    df[angle_col] = df[angle_col].astype(str).str.replace("°","", regex=False).str.strip()
    df = df[_is_numberlike(df[angle_col])] if callable(getattr(df, "__getitem__", None)) else df
    df.index = df[angle_col].astype(float)
    df = df.drop(columns=[angle_col])

    # Clean main angles (column headers)
    clean_cols = []
    for c in df.columns:
        cs = str(c).replace("°","").strip()
        clean_cols.append(float(cs) if _is_numberlike(cs) else cs)
    df.columns = clean_cols

    # Ensure numeric values
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.sort_index(axis=0).sort_index(axis=1)

def stack_height_outreach(height_df: pd.DataFrame, outreach_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine height and outreach matrices into a long table:
    columns: main_deg, folding_deg, height_m, outreach_m
    Only rows with finite numbers are kept.
    """
    # Align on the same angle grids (outer join to be safe)
    # Reindex both to the union of angles
    f_angles = sorted(set(height_df.index).union(set(outreach_df.index)))
    m_angles = sorted(set(height_df.columns).union(set(outreach_df.columns)))
    H = height_df.reindex(index=f_angles, columns=m_angles)
    R = outreach_df.reindex(index=f_angles, columns=m_angles)

    # Stack to long form
    h_long = H.stack().rename("height_m").reset_index(names=["folding_deg","main_deg"])
    r_long = R.stack().rename("outreach_m").reset_index(names=["folding_deg","main_deg"])
    df = pd.merge(h_long, r_long, on=["folding_deg","main_deg"], how="outer")
    # Keep finite pairs
    df = df[np.isfinite(df["height_m"]) & np.isfinite(df["outreach_m"])]
    return df.sort_values(["main_deg","folding_deg"]).reset_index(drop=True)
