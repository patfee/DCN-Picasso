import os
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline


# ----------------------------------------------------------------------
# Robust CSV → angle grid reader
# ----------------------------------------------------------------------

def _detect_sep(sample: str) -> str:
    """
    Heuristic: pick ';' if it appears more than ',', otherwise ','.
    Works for EU 'CSV' exports where ';' is the delimiter and ',' may be decimals.
    """
    return ";" if sample.count(";") > sample.count(",") else ","


def _read_angle_grid(filename: str, data_dir: str = "data"):
    """
    Reads an angle-grid CSV:
      - first row (from col 2 onward) = main jib angles (deg)
      - first col (from row 2 downward) = folding jib angles (deg)
      - body = numeric values
    Robust to either ';' or ',' as delimiter and to decimal commas.

    Returns:
      main_angles: np.ndarray[float]
      fold_angles: np.ndarray[float]
      body_df:     pd.DataFrame (index = fold_angles, columns = main_angles)
    """
    path = os.path.join(data_dir, filename)
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(4096)
    sep = _detect_sep(head)

    # Read with detected separator
    df_raw = pd.read_csv(path, sep=sep, engine="python")

    if df_raw.shape[0] < 2 or df_raw.shape[1] < 2:
        raise ValueError(f"{filename}: not a 2D angle grid (shape={df_raw.shape}).")

    # Parse header (main angles) — convert decimal commas to dots
    main_headers = [str(c).strip().replace(",", ".") for c in df_raw.columns[1:]]
    try:
        main_angles = np.array([float(x) for x in main_headers], dtype=float)
    except Exception as e:
        raise ValueError(f"{filename}: failed to parse main-jib angles from header.") from e

    # Parse first column (fold angles) — rows 1:end
    fold_col = df_raw.iloc[1:, 0].astype(str).str.strip().str.replace(",", ".", regex=False)
    try:
        fold_angles = np.array(fold_col.astype(float).to_list(), dtype=float)
    except Exception as e:
        raise ValueError(f"{filename}: failed to parse folding-jib angles from first column.") from e

    # Convert body to numeric (coerce irregulars to NaN), then fill gently
    body_str = df_raw.iloc[1:, 1:].astype(str).replace({",": "."}, regex=True)
    body = body_str.apply(lambda s: pd.to_numeric(s, errors="coerce"))

    body.index = fold_angles
    body.columns = main_angles

    return main_angles, fold_angles, body


# ----------------------------------------------------------------------
# Interpolation helpers
# ----------------------------------------------------------------------

def _interp_fill(values: pd.DataFrame) -> pd.DataFrame:
    """
    Light-touch infill for sporadic NaNs:
      - interpolate horizontally & vertically
      - ffill/bfill as final guard (no deprecated fillna(method=...))
    """
    tmp = values.copy()
    tmp = tmp.interpolate(axis=1, limit_direction="both")
    tmp = tmp.interpolate(axis=0, limit_direction="both")
    tmp = tmp.ffill(axis=1).bfill(axis=1)
    tmp = tmp.ffill(axis=0).bfill(axis=0)
    return tmp


def _interp_indices(n_old_rows: int, n_old_cols: int, n_new_rows: int, n_new_cols: int):
    """Index-space coordinates for interpolation (agnostic to actual angle spacing)."""
    fold_idx_old = np.arange(n_old_rows, dtype=float)
    main_idx_old = np.arange(n_old_cols, dtype=float)
    fold_idx_new = np.linspace(0, n_old_rows - 1, n_new_rows)
    main_idx_new = np.linspace(0, n_old_cols - 1, n_new_cols)
    return fold_idx_old, main_idx_old, fold_idx_new, main_idx_new


def _angles_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Safe equality for float angle arrays."""
    if len(a) != len(b):
        return False
    return np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=0, atol=1e-12)


def interpolate_value_grid(values: pd.DataFrame,
                           new_main: np.ndarray,
                           new_fold: np.ndarray,
                           mode: str = "linear") -> np.ndarray:
    """
    Interpolate a 2D angle grid (values indexed by fold angles, columns = main angles)
    to new angle arrays (new_main, new_fold).

    IMPORTANT:
    - If (new_main, new_fold) exactly match the original axes, returns the original
      numeric array WITHOUT interpolating (bitwise-safe vs your 1×/1× expectation).
    """
    # Short-circuit: exact grid → return original values
    orig_main = np.asarray(values.columns, dtype=float)
    orig_fold = np.asarray(values.index, dtype=float)
    if _angles_equal(orig_main, new_main) and _angles_equal(orig_fold, new_fold):
        return values.to_numpy(dtype=float)

    vals = _interp_fill(values)  # clean a bit before interpolating
    arr = vals.to_numpy(dtype=float)

    # Old grid index space
    n_rows, n_cols = arr.shape
    fold_idx_old, main_idx_old, fold_idx_new, main_idx_new = _interp_indices(
        n_rows, n_cols, len(new_fold), len(new_main)
    )

    mode = (mode or "linear").lower()
    if mode == "spline":
        # bicubic spline in index space (smooth)
        spline = RectBivariateSpline(fold_idx_old, main_idx_old, arr, kx=3, ky=3)
        result = spline(fold_idx_new, main_idx_new)
    else:
        # bilinear in index space (stable)
        interp = RegularGridInterpolator(
            (fold_idx_old, main_idx_old), arr,
            method="linear", bounds_error=False, fill_value=None
        )
        F, M = np.meshgrid(fold_idx_new, main_idx_new, indexing="ij")
        pts = np.column_stack([F.ravel(), M.ravel()])
        result = interp(pts).reshape(len(new_fold), len(new_main))

        # Edge clean-up with nearest if any NaNs remain
        if np.isnan(result).any():
            interp_nn = RegularGridInterpolator(
                (fold_idx_old, main_idx_old), arr,
                method="nearest", bounds_error=False, fill_value=None
            )
            result_nn = interp_nn(pts).reshape(len(new_fold), len(new_main))
            result = np.where(np.isnan(result), result_nn, result)

    return result


# ----------------------------------------------------------------------
# Crane outreach/height loaders
# ----------------------------------------------------------------------

def load_crane_data(data_dir: str = "data",
                    outreach_file: str = "outreach.csv",
                    height_file: str = "height.csv"):
    """
    Loads the crane outreach and height matrices from CSVs.
    Returns:
      main_angles (deg), fold_angles (deg), outreach_df, height_df
    """
    main_angles, fold_angles, outreach_data = _read_angle_grid(outreach_file, data_dir)
    main2, fold2, height_data = _read_angle_grid(height_file, data_dir)

    if not np.allclose(main_angles, main2) or not np.allclose(fold_angles, fold2):
        raise ValueError(
            "Angle mismatch between outreach.csv and height.csv "
            f"(main: {main_angles.shape} vs {main2.shape}; fold: {fold_angles.shape} vs {fold2.shape})"
        )

    return main_angles, fold_angles, outreach_data, height_data


# ----------------------------------------------------------------------
# Grid / dataset generators
# ----------------------------------------------------------------------

def _subdivide_angles(orig_angles: np.ndarray, factor: int) -> np.ndarray:
    """
    Return a new array subdividing each original interval into 'factor' parts,
    preserving all original angle values. factor=1 returns a copy of the original.
    """
    orig = np.asarray(orig_angles, dtype=float)
    if factor <= 1 or len(orig) < 2:
        return orig.copy()

    pieces = []
    for i in range(len(orig) - 1):
        a, b = orig[i], orig[i + 1]
        # include left endpoint, exclude right to avoid duplicates across segments
        pieces.append(np.linspace(a, b, factor, endpoint=False))
    pieces.append(np.array([orig[-1]]))
    return np.concatenate(pieces)


def get_position_grids(config: dict | None = None, data_dir: str = "data"):
    """
    Creates the (possibly interpolated) Outreach & Height matrices based on configuration,
    and returns:
      Xgrid (outreach), Ygrid (height), new_main_angles, new_fold_angles

    Config keys:
      - interp_mode: "linear" | "spline"
      - main_factor: int (1,2,4,8,16)
      - folding_factor (or fold_factor): int (1,2,4,8,16,32)
      - include_pedestal: bool
      - pedestal_height: float (default 6.0 m)
    """
    config = config or {}
    mode = (config.get("interp_mode") or "linear").lower()
    include_pedestal = bool(config.get("include_pedestal", False))
    pedestal_height = float(config.get("pedestal_height", 6.0))

    main_factor = int(config.get("main_factor", 1))
    folding_factor = int(config.get("folding_factor", config.get("fold_factor", 1)))

    # Load base data
    main_angles, fold_angles, outreach_df, height_df = load_crane_data(data_dir)

    # New angle arrays using exact subdivision (keeps original points)
    new_main = _subdivide_angles(main_angles, main_factor)
    new_fold = _subdivide_angles(fold_angles, folding_factor)

    # Interpolate to new grid (OR pass-through when factors==1 thanks to short-circuit)
    outreach_grid = interpolate_value_grid(outreach_df, new_main, new_fold, mode)
    height_grid   = interpolate_value_grid(height_df,   new_main, new_fold, mode)

    if include_pedestal:
        height_grid = height_grid + pedestal_height

    return outreach_grid, height_grid, new_main, new_fold


# ----------------------------------------------------------------------
# Flatteners
# ----------------------------------------------------------------------

def flatten_with_values(X: np.ndarray, Y: np.ndarray, values, value_name: str = "Value") -> pd.DataFrame:
    """
    Flattens 2D outreach/height/value matrices into a tidy DataFrame with XY and one value column.
    'values' can be a 2D np.ndarray or a pd.DataFrame matching X/Y shape.
    """
    if isinstance(values, pd.DataFrame):
        V = values.to_numpy(dtype=float)
    else:
        V = np.asarray(values, dtype=float)

    df = pd.DataFrame({
        "Outreach [m]": X.ravel(),
        "Height [m]":   Y.ravel(),
        value_name:     V.ravel(),
    })
    return df


def _flatten_with_angles(outreach_grid: np.ndarray,
                         height_grid: np.ndarray,
                         main_angles: np.ndarray,
                         fold_angles: np.ndarray) -> pd.DataFrame:
    """
    Flatten grids and keep the angle pair for each point.
    Returns columns: Outreach [m], Height [m], main_deg, folding_deg
    """
    F, M = np.meshgrid(fold_angles, main_angles, indexing="ij")  # F rows, M cols
    df = pd.DataFrame({
        "Outreach [m]": outreach_grid.ravel(),
        "Height [m]":   height_grid.ravel(),
        "main_deg":     M.ravel(),
        "folding_deg":  F.ravel(),
    })
    return df


# ----------------------------------------------------------------------
# Backward-compatible API used by pages/subpages
# ----------------------------------------------------------------------

def get_crane_points(config: dict | None = None, data_dir: str = "data") -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      ['Outreach [m]', 'Height [m]', 'main_deg', 'folding_deg']
    honoring interpolation/pedestal settings in config.
    """
    Xgrid, Ygrid, new_main, new_fold = get_position_grids(config=config, data_dir=data_dir)
    df = _flatten_with_angles(Xgrid, Ygrid, new_main, new_fold)
    return df


def load_value_grid(filename: str, data_dir: str = "data") -> pd.DataFrame:
    """
    Loads and cleans a capacity (or similar) grid file.
    Returns a DataFrame indexed by folding angles, columns = main angles.
    """
    _, _, df = _read_angle_grid(filename, data_dir)
    df = _interp_fill(df)
    return df


# ----------------------------------------------------------------------
# (Optional) quick validator you can call during debugging
# ----------------------------------------------------------------------

def validate_grid(df: pd.DataFrame, name="grid"):
    """
    Prints basic diagnostics for a numeric grid DataFrame.
    """
    if df.empty:
        print(f"[WARN] {name}: empty DataFrame")
        return
    n_nan = int(df.isna().sum().sum())
    print(f"[OK] {name}: {df.shape[0]}×{df.shape[1]} grid, NaNs={n_nan}")
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        print(f"[WARN] {name}: non-finite values exist.")
