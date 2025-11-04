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
# Kinematic (2-link) model — per your confirmed convention
#   θ = main angle from horizontal (0°..~84°, CCW positive)
#   φ = folding angle relative to main (0° folded flat colinear, CCW positive to ~102°)
#   x = L1*cosθ + L2*cos(θ - φ) + x0
#   y = L1*sinθ + L2*sin(θ - φ) + y0
# ----------------------------------------------------------------------
def _solve_2link_from_points(df_points: pd.DataFrame):
    """Solve L1, L2, x0, y0 by linear least squares from original CSV points (no pedestal)."""
    th = np.deg2rad(df_points["main_deg"].to_numpy())
    ph = np.deg2rad(df_points["folding_deg"].to_numpy())
    x  = df_points["Outreach [m]"].to_numpy()
    y  = df_points["Height [m]"].to_numpy()

    # Basis vectors
    c1 = np.cos(th)
    s1 = np.sin(th)
    c2 = np.cos(th - ph)
    s2 = np.sin(th - ph)

    # Build linear system A @ [L1, L2, x0, y0] = b
    Ax = np.column_stack([c1,  c2,  np.ones_like(x), np.zeros_like(x)])
    Ay = np.column_stack([s1,  s2,  np.zeros_like(y), np.ones_like(y)])
    A  = np.vstack([Ax, Ay])
    b  = np.concatenate([x, y])

    params, *_ = np.linalg.lstsq(A, b, rcond=None)
    L1, L2, x0, y0 = params
    return float(L1), float(L2), float(x0), float(y0)


def _evaluate_2link(main_deg: np.ndarray, fold_deg: np.ndarray, L1: float, L2: float, x0: float, y0: float):
    """Evaluate the 2-link model on a main×folding angle grid (degrees). Returns (Xgrid, Ygrid) DataFrames."""
    TH, PH = np.meshgrid(main_deg, fold_deg)        # columns=main, rows=folding
    th = np.deg2rad(TH)
    ph = np.deg2rad(PH)
    x = L1 * np.cos(th) + L2 * np.cos(th - ph) + x0
    y = L1 * np.sin(th) + L2 * np.sin(th - ph) + y0
    return pd.DataFrame(x, index=fold_deg, columns=main_deg), pd.DataFrame(y, index=fold_deg, columns=main_deg)


def _anchor_with_original(Xgrid: pd.DataFrame, Ygrid: pd.DataFrame,
                          Xorig: pd.DataFrame, Yorig: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Overwrite dense grids at original angle pairs with exact CSV values (guarantees identical originals)."""
    Xgrid = Xgrid.copy(); Ygrid = Ygrid.copy()
    Xgrid.index = Xgrid.index.astype(float); Xgrid.columns = Xgrid.columns.astype(float)
    Ygrid.index = Ygrid.index.astype(float); Ygrid.columns = Ygrid.columns.astype(float)

    for f in Xorig.index.astype(float):
        if f in Xgrid.index:
            for m in Xorig.columns.astype(float):
                if m in Xgrid.columns:
                    Xgrid.at[f, m] = Xorig.at[f, m]
                    Ygrid.at[f, m] = Yorig.at[f, m]
    return Xgrid, Ygrid


# ----------------------------------------------------------------------
# Main entry: get_crane_points
# ----------------------------------------------------------------------
def get_crane_points(config: dict | None = None, data_dir: str = "data") -> pd.DataFrame:
    outreach_grid, height_grid = load_crane_grids(data_dir)

    include = bool(config.get("include_pedestal", False)) if config else False
    pedestal = float(config.get("pedestal_height", 6.0)) if config else 6.0
    main_factor = int(config.get("main_factor", 1)) if config else 1
    folding_factor = int(config.get("folding_factor", 1)) if config else 1
    mode = (config.get("interp_mode") or "linear").lower() if config else "linear"

    orig_main = outreach_grid.columns.to_numpy(dtype=float)
    orig_fold = outreach_grid.index.to_numpy(dtype=float)
    new_main = _subdivide_angles(orig_main, main_factor)
    new_fold = _subdivide_angles(orig_fold, folding_factor)

    # --- Kinematic mode (exact 2-link geometry) with hard anchoring ------
    if mode == "kinematic":
        # Fit on ORIGINAL grids (no pedestal)
        df_orig = _flatten(outreach_grid, height_grid)
        L1, L2, x0, y0 = _solve_2link_from_points(df_orig)

        # Evaluate on dense grid
        Xgrid, Ygrid = _evaluate_2link(new_main, new_fold, L1, L2, x0, y0)

        # Anchor exact original values back in
        Xgrid, Ygrid = _anchor_with_original(Xgrid, Ygrid, outreach_grid, height_grid)

        # Apply pedestal AFTER anchoring
        if include:
            Ygrid = Ygrid + pedestal

        df = _flatten(Xgrid, Ygrid)
        # Optional diagnostics (you can read attrs in a page to show fit errors if desired)
        # Compute RMS vs originals for sanity:
        x_fit_o, y_fit_o = _evaluate_2link(orig_main, orig_fold, L1, L2, x0, y0)
        rms = float(np.sqrt(np.nanmean((x_fit_o - outreach_grid).to_numpy()**2 +
                                       (y_fit_o - height_grid).to_numpy()**2)))
        df.attrs["fit_rms"] = rms
        df.attrs["fit_params"] = {"L1": L1, "L2": L2, "x0": x0, "y0": y0}
        return df

    # --- Spline / Linear modes ------------------------------------------
    if mode == "spline":
        Xgrid = _interp_grid_spline(outreach_grid, new_main, new_fold)
        Ygrid = _interp_grid_spline(height_grid, new_main, new_fold)
    else:
        Xgrid = _interp_grid_linear(outreach_grid, new_main, new_fold)
        Ygrid = _interp_grid_linear(height_grid, new_main, new_fold)

    if include:
        Ygrid = Ygrid + pedestal

    return _flatten(Xgrid, Ygrid)
