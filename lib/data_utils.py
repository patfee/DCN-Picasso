import os
import csv
import numpy as np
import pandas as pd
from functools import lru_cache


# ------------------------------
# CSV → angle grids
# ------------------------------
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


# ------------------------------
# Helpers: subdivision, flatten
# ------------------------------
def _subdivide_angles(orig: np.ndarray, factor: int) -> np.ndarray:
    orig = np.asarray(orig, dtype=float)
    if factor <= 1 or orig.size <= 1:
        return orig.copy()
    pieces = []
    for i in range(len(orig) - 1):
        a, b = orig[i], orig[i + 1]
        seg = np.linspace(a, b, factor, endpoint=False)  # include left, exclude right
        pieces.append(seg)
    pieces.append(np.array([orig[-1]]))
    return np.concatenate(pieces)


def _flatten(outreach_grid: pd.DataFrame, height_grid: pd.DataFrame) -> pd.DataFrame:
    X = outreach_grid.stack().rename("Outreach [m]").reset_index()
    X.columns = ["folding_deg", "main_deg", "Outreach [m]"]
    Y = height_grid.stack().rename("Height [m]").reset_index()
    Y.columns = ["folding_deg", "main_deg", "Height [m]"]
    df = pd.merge(X, Y, on=["folding_deg", "main_deg"], how="inner")
    return df.dropna(subset=["Outreach [m]", "Height [m]"]).reset_index(drop=True)


# ------------------------------
# Interpolation: linear & spline
# ------------------------------
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
    mat = tmp.to_numpy(dtype=float)  # (F, M)

    # Along main
    r = np.empty((mat.shape[0], len(new_main)), float)
    for i in range(mat.shape[0]):
        r[i, :] = np.interp(new_main, orig_main, mat[i, :])

    # Along folding
    out = np.empty((len(new_folding), r.shape[1]), float)
    for j in range(r.shape[1]):
        out[:, j] = np.interp(new_folding, orig_folding, r[:, j])

    return pd.DataFrame(out, index=new_folding, columns=new_main)


def _interp_grid_spline(values: pd.DataFrame, new_main: np.ndarray, new_folding: np.ndarray) -> pd.DataFrame:
    """
    Cubic spline along each axis (PCHIP-like behavior via pandas interpolate then np.interp).
    Keeps monotonicity better than plain cubic; no SciPy dependency.
    """
    # 1) cubic along main with dense original grid, then interpolate
    tmp = values.copy().interpolate(axis=1, method="spline", order=3, limit_direction="both")
    tmp = tmp.interpolate(axis=0, method="spline", order=3, limit_direction="both")
    tmp = _interp_fill(tmp)  # final clean-up

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


# ------------------------------
# Kinematic (2-link) model
# ------------------------------
def _solve_2link_params(df_points: pd.DataFrame):
    """
    Solve for L1, L2, x0, y0 via linear least squares.
    df_points columns: 'main_deg', 'folding_deg', 'Outreach [m]', 'Height [m]' (no pedestal added!)
    Model:
      x = L1 cosθ + L2 cos(θ+φ) + x0
      y = L1 sinθ + L2 sin(θ+φ) + y0
    """
    th = np.deg2rad(df_points["main_deg"].to_numpy())
    ph = np.deg2rad(df_points["folding_deg"].to_numpy())
    x  = df_points["Outreach [m]"].to_numpy()
    y  = df_points["Height [m]"].to_numpy()

    cth, sth = np.cos(th), np.sin(th)
    ctp, stp = np.cos(th + ph), np.sin(th + ph)

    # Build linear system A @ [L1, L2, x0, y0] = b
    # Stack x-equations and y-equations
    Ax = np.column_stack([cth,  ctp,  np.ones_like(x), np.zeros_like(x)])
    Ay = np.column_stack([sth,  stp,  np.zeros_like(y), np.ones_like(y)])
    A  = np.vstack([Ax, Ay])
    b  = np.concatenate([x, y])

    params, *_ = np.linalg.lstsq(A, b, rcond=None)
    L1, L2, x0, y0 = params
    return float(L1), float(L2), float(x0), float(y0)


def _forward_2link(main_deg: np.ndarray, fold_deg: np.ndarray, L1: float, L2: float, x0: float, y0: float):
    TH, PH = np.meshgrid(main_deg, fold_deg)  # cols: main, rows: folding
    th = np.deg2rad(TH)
    ph = np.deg2rad(PH)
    x = L1 * np.cos(th) + L2 * np.cos(th + ph) + x0
    y = L1 * np.sin(th) + L2 * np.sin(th + ph) + y0
    return pd.DataFrame(x, index=fold_deg, columns=main_deg), pd.DataFrame(y, index=fold_deg, columns=main_deg)


# ------------------------------
# Public: get_crane_points
# ------------------------------
def get_crane_points(config: dict | None = None, data_dir: str = "data") -> pd.DataFrame:
    """
    Build the effective dataset for the app based on config:
      config = {
        "include_pedestal": bool,
        "pedestal_height": float,
        "main_factor": int in {1,2,4,8,16},
        "folding_factor": int in {1,2,4,8,16,32},
        "interp_mode": "linear" | "spline" | "kinematic"
      }
    Returns DataFrame with columns: folding_deg, main_deg, Outreach [m], Height [m]
    """
    outreach_grid, height_grid = load_crane_grids(data_dir)

    include       = bool(config.get("include_pedestal", False)) if config else False
    pedestal      = float(config.get("pedestal_height", 6.0))    if config else 6.0
    main_factor   = int(config.get("main_factor", 1))            if config else 1
    folding_factor= int(config.get("folding_factor", 1))         if config else 1
    mode          = (config.get("interp_mode") or "linear")       if config else "linear"
    mode = mode.lower()

    orig_main = outreach_grid.columns.to_numpy(dtype=float)
    orig_fold = outreach_grid.index.to_numpy(dtype=float)
    new_main  = _subdivide_angles(orig_main,  main_factor)
    new_fold  = _subdivide_angles(orig_fold, folding_factor)

    if mode == "kinematic":
        # Fit parameters using the ORIGINAL (no pedestal) points
        df_orig = _flatten(outreach_grid, height_grid)
        L1, L2, x0, y0 = _solve_2link_params(df_orig)

        Xgrid, Ygrid = _forward_2link(new_main, new_fold, L1, L2, x0, y0)

        # Apply pedestal afterwards (preserves meaning of y0 as mechanical offset)
        if include:
            Ygrid = Ygrid + pedestal

        df = _flatten(Xgrid, Ygrid)
        return df

    # Grid interpolation modes
    if mode == "spline":
        Xgrid = _interp_grid_spline(outreach_grid, new_main, new_fold)
        Ygrid = _interp_grid_spline(height_grid,   new_main, new_fold)
    else:  # "linear"
        Xgrid = _interp_grid_linear(outreach_grid, new_main, new_fold)
        Ygrid = _interp_grid_linear(height_grid,   new_main, new_fold)

    if include:
        Ygrid = Ygrid + pedestal

    return _flatten(Xgrid, Ygrid)
