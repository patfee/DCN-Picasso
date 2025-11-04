import os
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline

# ----------------------------------------------------------------------
# Data reading utilities
# ----------------------------------------------------------------------

def _read_angle_grid(filename: str, data_dir: str = "data"):
    """
    Reads an angle-grid CSV (first row = main jib angles, first col = folding jib angles).
    Returns: (main_angles: np.ndarray, fold_angles: np.ndarray, body: pd.DataFrame)
    """
    path = os.path.join(data_dir, filename)
    # sep=None = automatic delimiter detection (comma/semicolon/tab)
    df_raw = pd.read_csv(path, sep=None, engine="python")

    # First row header cells (excluding first col header) are the main-jib angles
    main_angles = df_raw.columns[1:]
    main_angles = np.array([float(str(x).replace(",", ".")) for x in main_angles], dtype=float)

    # First column (excluding header row) are the folding-jib angles
    fold_angles = df_raw.iloc[1:, 0].astype(str)
    fold_angles = np.array([float(x.replace(",", ".")) for x in fold_angles], dtype=float)

    # Convert the numeric grid body, coercing non-numeric to NaN
    body_str = df_raw.iloc[1:, 1:].astype(str).replace({",": "."}, regex=True)
    # column-wise numeric coercion (no deprecated applymap)
    body = body_str.apply(lambda s: pd.to_numeric(s, errors="coerce"))

    # Ensure clean index/columns as floats
    body.index = fold_angles
    body.columns = main_angles

    return main_angles, fold_angles, body


# ----------------------------------------------------------------------
# Interpolation helpers
# ----------------------------------------------------------------------

def _interp_fill(values: pd.DataFrame) -> pd.DataFrame:
    """Interpolates missing values, filling NaNs with nearest neighbours."""
    tmp = values.copy()
    tmp = tmp.interpolate(axis=1, limit_direction="both")
    tmp = tmp.interpolate(axis=0, limit_direction="both")
    # use ffill/bfill (not deprecated fillna(method=...))
    tmp = tmp.ffill(axis=1).bfill(axis=1)
    tmp = tmp.ffill(axis=0).bfill(axis=0)
    return tmp


def _interp_indices(n_old_rows: int, n_old_cols: int, n_new_rows: int, n_new_cols: int):
    """
    Build index arrays that map new grid indices to the old [0..N-1] index space
    so we can interpolate irrespective of actual angle spacing.
    """
    fold_idx_old = np.arange(n_old_rows, dtype=float)
    main_idx_old = np.arange(n_old_cols, dtype=float)
    fold_idx_new = np.linspace(0, n_old_rows - 1, n_new_rows)
    main_idx_new = np.linspace(0, n_old_cols - 1, n_new_cols)
    return fold_idx_old, main_idx_old, fold_idx_new, main_idx_new


def interpolate_value_grid(values: pd.DataFrame,
                           new_main: np.ndarray,
                           new_fold: np.ndarray,
                           mode: str = "linear") -> np.ndarray:
    """
    Interpolates a 2D angle grid (values indexed by fold angles, columns = main angles)
    to new main/fold angle arrays. We resample by index-space to avoid assuming linear
    angle spacing in the source CSV.
    """
    vals = _interp_fill(values)
    arr = vals.to_numpy(dtype=float)

    # Old grid index space
    n_rows, n_cols = arr.shape
    fold_idx_old, main_idx_old, fold_idx_new, main_idx_new = _interp_indices(
        n_rows, n_cols, len(new_fold), len(new_main)
    )

    if (mode or "linear").lower() == "spline":
        # bicubic spline in index space
        spline = RectBivariateSpline(fold_idx_old, main_idx_old, arr, kx=3, ky=3)
        result = spline(fold_idx_new, main_idx_new)
    else:
        # bilinear in index space
        interp = RegularGridInterpolator(
            (fold_idx_old, main_idx_old), arr,
            method="linear", bounds_error=False, fill_value=None
        )
        F, M = np.meshgrid(fold_idx_new, main_idx_new, indexing="ij")
        pts = np.column_stack([F.ravel(), M.ravel()])
        result = interp(pts).reshape(len(new_fold), len(new_main))

        # Optional: light fill for any remaining NaNs at the edges
        if np.isnan(result).any():
            # nearest neighbor pass
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
    Returns: (main_angles, fold_angles, outreach_df, height_df)
    """
    main_angles, fold_angles, outreach_data = _read_angle_grid(outreach_file, data_dir)
    main2, fold2, height_data = _read_angle_grid(height_file, data_dir)

    if not np.allclose(main_angles, main2) or not np.allclose(fold_angles, fold2):
        raise ValueError("Main/Folding angles mismatch between outreach.csv and height.csv.")

    return main_angles, fold_angles, outreach_data, height_data


# ----------------------------------------------------------------------
# Grid / dataset generators
# ----------------------------------------------------------------------

def _subdivide_angles(orig_angles: np.ndarray, factor: int) -> np.ndarray:
    """
    Return a new array subdividing each original interval into 'factor' parts,
    preserving all original angle values.
    factor=1 returns the original angles.
    """
    orig = np.asarray(orig_angles, dtype=float)
    if factor <= 1 or len(orig) < 2:
        return orig.copy()

    pieces = []
    for i in range(len(orig) - 1):
        a, b = orig[i], orig[i + 1]
        # include left endpoint, exclude right to avoid duplicates
        pieces.append(np.linspace(a, b, factor, endpoint=False))
    pieces.append(np.array([orig[-1]]))
    return np.concatenate(pieces)


def get_position_grids(config: dict | None = None,
                       data_dir: str = "data"):
    """
    Creates the interpolated Outreach & Height matrices based on configuration,
    and returns (outreach_grid, height_grid, new_main_angles, new_fold_angles).

    - Respects interpolation mode (linear/spline)
    - Respects main_factor and folding_factor (or fold_factor)
    - Adds pedestal height to Height if include_pedestal is True
    """
    config = config or {}
    mode = (config.get("interp_mode") or "linear").lower()
    include_pedestal = bool(config.get("include_pedestal", False))
    pedestal_height = float(config.get("pedestal_height", 6.0))

    # accept both keys for backwards compatibility
    main_factor = int(config.get("main_factor", 1))
    folding_factor = int(config.get("folding_factor", config.get("fold_factor", 1)))

    # Load base data
    main_angles, fold_angles, outreach_df, height_df = load_crane_data(data_dir)

    # New angle arrays using exact subdivision (keeps original points)
    new_main = _subdivide_angles(main_angles, main_factor)
    new_fold = _subdivide_angles(fold_angles, folding_factor)

    # Interpolate to new grid
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
    values can be a 2D np.ndarray or a pd.DataFrame matching X/Y shape.
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
# Backward-compat API used by Page 1: get_crane_points
# ----------------------------------------------------------------------

def get_crane_points(config: dict | None = None, data_dir: str = "data") -> pd.DataFrame:
    """
    Backwards-compatible function expected by subpages/page1_tab_a.py.
    Returns a DataFrame with columns:
      ['Outreach [m]', 'Height [m]', 'main_deg', 'folding_deg']
    honoring interpolation/pedestal settings in config.
    """
    Xgrid, Ygrid, new_main, new_fold = get_position_grids(config=config, data_dir=data_dir)
    df = _flatten_with_angles(Xgrid, Ygrid, new_main, new_fold)
    return df


# ----------------------------------------------------------------------
# Harbour capacity data loader (angle-grid values)
# ----------------------------------------------------------------------

def load_value_grid(filename: str, data_dir: str = "data") -> pd.DataFrame:
    """
    Loads and cleans a capacity (or similar) grid file.
    Returns a DataFrame indexed by folding angles, columns = main angles.
    """
    _, _, df = _read_angle_grid(filename, data_dir)
    df = _interp_fill(df)
    return df
