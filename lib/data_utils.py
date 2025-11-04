import os
import numpy as np
import pandas as pd
from functools import lru_cache


# ----------------------------------------------------------------------
# Read & prepare angle-based CSV grids
# ----------------------------------------------------------------------
def _read_angle_grid(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    # Detect delimiter
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
        .astype(float).tolist()
    )
    folding_angles = (
        df_raw.iloc[1:, 0]
        .astype(str).str.replace(",", ".", regex=False)
        .astype(float).tolist()
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
    common_rows   = outreach_grid.index.intersection(height_grid.index)
    common_cols   = outreach_grid.columns.intersection(height_grid.columns)
    outreach_grid = outreach_grid.loc[common_rows, common_cols]
    height_grid   = height_grid.loc[common_rows, common_cols]
    return outreach_grid, height_grid


# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def _subdivide_angles(orig: np.ndarray, factor: int) -> np.ndarray:
    orig = np.asarray(orig, dtype=float)
    if factor <= 1 or len(orig) < 2:
        return orig.copy()
    segs = []
    for i in range(len(orig) - 1):
        a, b = orig[i], orig[i + 1]
        segs.append(np.linspace(a, b, factor, endpoint=False))
    segs.append(np.array([orig[-1]]))
    return np.concatenate(segs)


def _flatten(outreach_grid: pd.DataFrame, height_grid: pd.DataFrame) -> pd.DataFrame:
    X = outreach_grid.stack().rename("Outreach [m]").reset_index()
    X.columns = ["folding_deg", "main_deg", "Outreach [m]"]
    Y = height_grid.stack().rename("Height [m]").reset_index()
    Y.columns = ["folding_deg", "main_deg", "Height [m]"]
    df = pd.merge(X, Y, on=["folding_deg", "main_deg"], how="inner")
    return df.dropna(subset=["Outreach [m]", "Height [m]"]).reset_index(drop=True)


def _interp_fill(values: pd.DataFrame) -> pd.DataFrame:
    tmp = values.copy()
    tmp = tmp.interpolate(axis=1, limit_direction="both")
    tmp = tmp.interpolate(axis=0, limit_direction="both")
    tmp = tmp.fillna(method="ffill", axis=1).fillna(method="bfill", axis=1)
    tmp = tmp.fillna(method="ffill", axis=0).fillna(method="bfill", axis=0)
    return tmp


def _interp_grid_linear(values: pd.DataFrame, new_main: np.ndarray, new_folding: np.ndarray) -> pd.DataFrame:
    tmp = _interp_fill(values)
    orig_folding = tmp.index.to_numpy(dtype=float)
    orig_main    = tmp.columns.to_numpy(dtype=float)
    mat = tmp.to_numpy(dtype=float)

    r = np.empty((mat.shape[0], len(new_main)), float)
    for i in range(mat.shape[0]):
        r[i, :] = np.interp(new_main, orig_main, mat[i, :])

    out = np.empty((len(new_folding), r.shape[1]), float)
    for j in range(r.shape[1]):
        out[:, j] = np.interp(new_folding, orig_folding, r[:, j])

    return pd.DataFrame(out, index=new_folding, columns=new_main)


def _interp_grid_spline(values: pd.DataFrame, new_main: np.ndarray, new_folding: np.ndarray) -> pd.DataFrame:
    tmp = values.copy().interpolate(axis=1, method="spline", order=3, limit_direction="both")
    tmp = tmp.interpolate(axis=0, method="spline", order=3, limit_direction="both")
    tmp = _interp_fill(tmp)

    orig_folding = tmp.index.to_numpy(dtype=float)
    orig_main    = tmp.columns.to_numpy(dtype=float)
    mat = tmp.to_numpy(dtype=float)

    r = np.empty((mat.shape[0], len(new_main)), float)
    for i in range(mat.shape[0]):
        r[i, :] = np.interp(new_main, orig_main, mat[i, :])
    out = np.empty((len(new_folding), r.shape[1]), float)
    for j in range(r.shape[1]):
        out[:, j] = np.interp(new_folding, orig_folding, r[:, j])

    return pd.DataFrame(out, index=new_folding, columns=new_main)


# ----------------------------------------------------------------------
# Main entry: get_crane_points (Linear/Spline only)
# ----------------------------------------------------------------------
def get_crane_points(config: dict | None = None, data_dir: str = "data") -> pd.DataFrame:
    outreach_grid, height_grid = load_crane_grids(data_dir)

    include = bool(config.get("include_pedestal", False)) if config else False
    pedestal = float(config.get("pedestal_height", 6.0)) if config else 6.0
    main_factor = int(config.get("main_factor", 1)) if config else 1
    folding_factor = int(config.get("folding_factor", 1)) if config else 1
    mode = (config.get("interp_mode") or "linear").lower() if config else "linear"
    if mode not in {"linear", "spline"}:  # coerce any legacy value (e.g., 'kinematic')
        mode = "linear"

    orig_main = outreach_grid.columns.to_numpy(dtype=float)
    orig_fold = outreach_grid.index.to_numpy(dtype=float)
    new_main = _subdivide_angles(orig_main, main_factor)
    new_fold = _subdivide_angles(orig_fold, folding_factor)

    if mode == "spline":
        Xgrid = _interp_grid_spline(outreach_grid, new_main, new_fold)
        Ygrid = _interp_grid_spline(height_grid, new_main, new_fold)
    else:
        Xgrid = _interp_grid_linear(outreach_grid, new_main, new_fold)
        Ygrid = _interp_grid_linear(height_grid, new_main, new_fold)

    if include:
        Ygrid = Ygrid + pedestal

    return _flatten(Xgrid, Ygrid)
