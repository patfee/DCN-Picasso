import os
import csv
import pandas as pd


def _read_angle_grid(path: str) -> pd.DataFrame:
    """
    Read a CSV grid where:
      - Row 0, Col 1..N = main-jib angles (deg)
      - Col 0, Row 1..M = folding-jib angles (deg)
      - Body = numeric values (either Outreach [m] or Height [m])
    Auto-detects ';' or ',' delimiter and converts decimal commas.
    Returns a numeric DataFrame with:
      rows   = folding-jib angles
      cols   = main-jib angles
      values = the matrix values
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    # Auto-detect delimiter
    with open(path, "r", newline="") as f:
        sample = f.read(2048)
        if ";" in sample:
            delim = ";"
        elif "," in sample:
            delim = ","
        else:
            delim = "\t"

    df_raw = pd.read_csv(path, sep=delim, header=None, engine="python")

    # Header row (main-jib angles), header col (folding-jib angles)
    main_angles = (
        df_raw.iloc[0, 1:]
        .astype(str).str.replace(",", ".", regex=False)
        .astype(float)
        .tolist()
    )
    folding_angles = (
        df_raw.iloc[1:, 0]
        .astype(str).str.replace(",", ".", regex=False)
        .astype(float)
        .tolist()
    )

    # Body -> numeric
    body = df_raw.iloc[1:, 1:].applymap(
        lambda v: float(str(v).replace(",", ".")) if pd.notna(v) else float("nan")
    )
    body.index = folding_angles
    body.columns = main_angles
    body = body.dropna(axis=0, how="all").dropna(axis=1, how="all")

    return body


def load_crane_grids(data_dir: str = "data"):
    """
    Loads outreach and height angle grids.
    Returns:
      outreach_grid: DataFrame [folding_deg x main_deg] -> Outreach [m]
      height_grid  : DataFrame [folding_deg x main_deg] -> Height [m]
    """
    outreach_grid = _read_angle_grid(os.path.join(data_dir, "outreach.csv"))
    height_grid   = _read_angle_grid(os.path.join(data_dir, "height.csv"))

    # Safety: align on common angles
    common_rows = outreach_grid.index.intersection(height_grid.index)
    common_cols = outreach_grid.columns.intersection(height_grid.columns)
    outreach_grid = outreach_grid.loc[common_rows, common_cols]
    height_grid   = height_grid.loc[common_rows, common_cols]

    return outreach_grid, height_grid


def load_crane_points(data_dir: str = "data") -> pd.DataFrame:
    """
    Flattens the two angle grids into a tidy table of paired (Outreach, Height).
    Columns:
      - folding_deg
