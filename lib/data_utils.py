import os
import csv
import pandas as pd
from functools import lru_cache

# =========================
# Core CSV/Angle-grid loader
# =========================

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


@lru_cache(maxsize=1)
def load_crane_grids(data_dir: str = "data"):
    """
    Loads outreach and height angle grids (cached).
    Returns:
      outreach_grid: DataFrame [folding_deg x main_deg] -> Outreach [m]
      height_grid  : DataFrame [folding_deg x main_deg] -> Height [m] (above pedestal flange)
    """
    outreach_grid = _read_angle_grid(os.path.join(data_dir, "outreach.csv"))
    height_grid   = _read_angle_grid(os.path.join(data_dir, "height.csv"))

    # Align on common angles
    common_rows = outreach_grid.index.intersection(height_grid.index)
    common_cols = outreach_grid.columns.intersection(height_grid.columns)
    outreach_grid = outreach_grid.loc[common_rows, common_cols]
    height_grid   = height_grid.loc[common_rows, common_cols]
    return outreach_grid, height_grid


def _flatten_grids(outreach_grid: pd.DataFrame, height_grid: pd.DataFrame) -> pd.DataFrame:
    """
    Pair the two grids into a tidy table.
    """
    X_long = outreach_grid.stack().rename("Outreach [m]").reset_index()
    X_long.columns = ["folding_deg", "main_deg", "Outreach [m]"]

    Y_long = height_grid.stack().rename("Height [m]").reset_index()
    Y_long.columns = ["folding_deg", "main_deg", "Height [m]"]

    df = pd.merge(X_long, Y_long, on=["folding_deg", "main_deg"], how="inner")
    return df.dropna(subset=["Outreach [m]", "Height [m]"]).reset_index(drop=True)


# ======================================
# Global "effective dataset" construction
# ======================================

def get_crane_points(config: dict | None = None, data_dir: str = "data") -> pd.DataFrame:
    """
    Returns the *effective* dataset for the whole application, based on a config dict:
      config = {
        "include_pedestal": bool,
        "pedestal_height": float,   # meters
      }

    The returned DataFrame ALWAYS exposes the column name "Height [m]" so downstream
    code doesn't need to change. If pedestal is included, we add it to the baseline
    heights before returning.
    """
    outreach_grid, height_grid = load_crane_grids(data_dir)
    df = _flatten_grids(outreach_grid, height_grid)

    include = bool(config.get("include_pedestal", False)) if config else False
    pedestal = float(config.get("pedestal_height", 6.0)) if config else 6.0

    if include:
        df = df.copy()
        df["Height [m]"] = df["Height [m]"] + pedestal

    return df
