import os
import csv
import numpy as pd
import pandas as pd
from functools import lru_cache
import numpy as np


def _read_angle_grid(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r", newline="") as f:
        sample = f.read(2048)
        if ";" in sample:
            delim = ";"
        elif "," in sample:
            delim = ","
        else:
            delim = "\t"

    df_raw = pd.read_csv(path, sep=delim, header=None, engine="python")

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

    body = df_raw.iloc[1:, 1:].applymap(
        lambda v: float(str(v).replace(",", ".")) if pd.notna(v) else float("nan")
    )
    body.index = folding_angles
    body.columns = main_angles
    body = body.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return body


@lru_cache(maxsize=1)
def load_crane_grids(data_dir: str = "data"):
    outreach_grid = _read_angle_grid(os.path.join(data_dir, "outreach.csv"))
    height_grid   = _read_angle_grid(os.path.join(data_dir, "height.csv"))

    # align on common angles
    common_rows = outreach_grid.index.intersection(height_grid.index)
    common_cols = outreach_grid.columns.intersection(height_grid.columns)
    outreach_grid = outreach_grid.loc[common_rows, common_cols]
    height_grid   = height_grid.loc[common_rows, common_cols]
    return outreach_grid, height_grid


def _subdivide_angles(orig: np.ndarray, factor: int) -> np.ndarray:
    """
    Create a densified angle vector where each original interval [ai, ai+1]
    is split into `factor` equal sub-intervals. factor=1 returns the original.
    """
    orig = np.asarray(orig, dtype=float)
    if factor <= 1 or orig.size <= 1:
        return orig.copy()

    pieces = []
    for i in range(len(orig) - 1):
        a, b = orig[i], orig[i + 1]
        # generate factor points including left endpoint, excluding right endpoint
        seg = np.linspace(a, b, factor, endpoint=False)
        pieces.append(seg)
    pieces.append(np.array([orig[-1]]))
    return np.concatenate(pieces)


def _interp_grid(values: pd.DataFrame, new_main: np.ndarray, new_folding: np.ndarray) -> pd.DataFrame:
    """
    Bilinear (separable) interpolation using numpy:
      1) interpolate each row along main-axis to new_main
      2) interpolate each column along folding-axis to new_folding
    Assumes both original angle axes are strictly increasing.
    NaNs are prefilled by simple linear interpolation along axes.
    """
    # Fill NaNs roughly so np.interp works
    tmp = values.copy()
    tmp = tmp.interpolate(axis=1, limit_direction="both")
    tmp = tmp.interpolate(axis=0, limit_direction="both")
    tmp = tmp.fillna(method="ffill", axis=1).fillna(method="bfill", axis=1)
    tmp = tmp.fillna(method="ffill", axis=0).fillna(method="bfill", axis=0)

    orig_folding = tmp.index.to_numpy(dtype=float)
    orig_main    = tmp.columns.to_numpy(dtype=float)
    mat = tmp.to_numpy(dtype=float)  # shape (F, M)

    # Step 1: interpolate along main for each folding row
    interp_rows = np.empty((mat.shape[0], len(new_main)), dtype=float)
    for i in range(mat.shape[0]):
        interp_rows[i, :] = np.interp(new_main, orig_main, mat[i, :])

    # Step 2: interpolate along folding for each main column
    result = np.empty((len(new_folding), interp_rows.shape[1]), dtype=float)
    for j in range(interp_rows.shape[1]):
        result[:, j] = np.interp(new_folding, orig_folding, interp_rows[:, j])

    df_new = pd.DataFrame(result, index=new_folding, columns=new_main)
    return df_new


def _flatten(outreach_grid: pd.DataFrame, height_grid: pd.DataFrame) -> pd.DataFrame:
    X_long = outreach_grid.stack().rename("Outreach [m]").reset_index()
    X_long.columns = ["folding_deg", "main_deg", "Outreach [m]"]

    Y_long = height_grid.stack().rename("Height [m]").reset_index()
    Y_long.columns = ["folding_deg", "main_deg", "Height [m]"]

    df = pd.merge(X_long, Y_long, on=["folding_deg", "main_deg"], how="inner")
    return df.dropna(subset=["Outreach [m]", "Height [m]"]).reset_index(drop=True)


def get_crane_points(config: dict | None = None, data_dir: str = "data") -> pd.DataFrame:
    """
    Returns the effective (possibly interpolated) dataset for the whole app.
    Config keys supported:
      - include_pedestal: bool
      - pedestal_height: float (m)
      - main_factor: int in {1,2,4,8,16}
      - folding_factor: int in {1,2,4,8,16,32}
    """
    outreach_grid, height_grid = load_crane_grids(data_dir)

    # Read config
    include = bool(config.get("include_pedestal", False)) if config else False
    pedestal = float(config.get("pedestal_height", 6.0)) if config else 6.0
    main_factor = int(config.get("main_factor", 1)) if config else 1
    folding_factor = int(config.get("folding_factor", 1)) if config else 1

    # New angle axes
    orig_main = outreach_grid.columns.to_numpy(dtype=float)
    orig_fold = outreach_grid.index.to_numpy(dtype=float)
    new_main  = _subdivide_angles(orig_main, main_factor)
    new_fold  = _subdivide_angles(orig_fold, folding_factor)

    # Interpolate both grids onto the new axes
    outreach_i = _interp_grid(outreach_grid, new_main, new_fold)
    height_i   = _interp_grid(height_grid,   new_main, new_fold)

    # Pedestal inclusion (keeps the same column name)
    if include:
        height_i = height_i + pedestal

    # Flatten
    df = _flatten(outreach_i, height_i)
    return df
