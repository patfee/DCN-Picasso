import os
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline

# ----------------------------------------------------------------------
# Data reading utilities
# ----------------------------------------------------------------------

def _read_angle_grid(filename: str, data_dir: str = "data") -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Reads an angle-grid CSV (first row = main jib angles, first col = folding jib angles)."""
    path = os.path.join(data_dir, filename)
    df_raw = pd.read_csv(path, sep=None, engine="python")

    # First row (excluding first cell) = main jib angles
    main_angles = df_raw.columns[1:]
    main_angles = np.array([float(str(x).replace(",", ".")) for x in main_angles])

    # First column (excluding header) = folding jib angles
    fold_angles = df_raw.iloc[1:, 0].astype(str)
    fold_angles = np.array([float(x.replace(",", ".")) for x in fold_angles])

    # Convert numeric grid body, coercing non-numeric to NaN
    body_str = df_raw.iloc[1:, 1:].astype(str).replace({",": "."}, regex=True)
    body = body_str.apply(lambda s: pd.to_numeric(s, errors="coerce"))

    return main_angles, fold_angles, body


# ----------------------------------------------------------------------
# Interpolation helper (safe fill + cubic/linear modes)
# ----------------------------------------------------------------------

def _interp_fill(values: pd.DataFrame) -> pd.DataFrame:
    """Interpolates missing values, filling NaNs with nearest neighbours."""
    tmp = values.copy()
    tmp = tmp.interpolate(axis=1, limit_direction="both")
    tmp = tmp.interpolate(axis=0, limit_direction="both")
    tmp = tmp.ffill(axis=1).bfill(axis=1)
    tmp = tmp.ffill(axis=0).bfill(axis=0)
    return tmp


def interpolate_value_grid(values: pd.DataFrame,
                           new_main: np.ndarray,
                           new_fold: np.ndarray,
                           mode: str = "linear") -> np.ndarray:
    """Interpolates a 2D angle grid to new main/fold angle arrays."""
    vals = _interp_fill(values)
    main = np.arange(vals.shape[1])
    fold = np.arange(vals.shape[0])
    arr = vals.to_numpy(dtype=float)

    if mode.lower() == "spline":
        spline = RectBivariateSpline(fold, main, arr, kx=3, ky=3)
        fold_idx = np.linspace(0, len(fold) - 1, len(new_fold))
        main_idx = np.linspace(0, len(main) - 1, len(new_main))
        result = spline(fold_idx, main_idx)
    else:
        interp = RegularGridInterpolator((fold, main), arr, method="linear", bounds_error=False, fill_value=None)
        fold_idx = np.linspace(0, len(fold) - 1, len(new_fold))
        main_idx = np.linspace(0, len(main) - 1, len(new_main))
        F, M = np.meshgrid(fold_idx, main_idx, indexing="ij")
        pts = np.column_stack([F.ravel(), M.ravel()])
        result = interp(pts).reshape(len(new_fold), len(new_main))
    return result


# ----------------------------------------------------------------------
# Position grid (Outreach/Height) loader
# ----------------------------------------------------------------------

def load_crane_data(data_dir: str = "data",
                    outreach_file: str = "outreach.csv",
                    height_file: str = "height.csv") -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Loads the crane outreach and height matrices from CSVs."""
    main_angles, fold_angles, outreach_data = _read_angle_grid(outreach_file, data_dir)
    main2, fold2, height_data = _read_angle_grid(height_file, data_dir)
    assert np.allclose(main_angles, main2), "Main jib angles mismatch between outreach/height."
    assert np.allclose(fold_angles, fold2), "Folding jib angles mismatch between outreach/height."
    return main_angles, fold_angles, outreach_data, height_data


# ----------------------------------------------------------------------
# Grid / dataset generators
# ----------------------------------------------------------------------

def get_position_grids(config: dict | None = None,
                       data_dir: str = "data") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Creates the interpolated Outreach & Height matrices based on configuration."""
    if config is None:
        config = {}
    main_factor = int(config.get("main_factor", 1))
    fold_factor = int(config.get("fold_factor", 1))
    mode = (config.get("interp_mode") or "linear").lower()
    include_pedestal = bool(config.get("include_pedestal", False))
    pedestal_height = float(config.get("pedestal_height", 6.0))

    # Load base data
    main_angles, fold_angles, outreach_data, height_data = load_crane_data(data_dir)

    # New finer grids
    new_main = np.linspace(main_angles.min(), main_angles.max(),
                           len(main_angles) * main_factor)
    new_fold = np.linspace(fold_angles.min(), fold_angles.max(),
                           len(fold_angles) * fold_factor)

    # Interpolate to new grid
    outreach_grid = interpolate_value_grid(outreach_data, new_main, new_fold, mode)
    height_grid = interpolate_value_grid(height_data, new_main, new_fold, mode)

    if include_pedestal:
        height_grid = height_grid + pedestal_height

    return outreach_grid, height_grid, new_main, new_fold


# ----------------------------------------------------------------------
# Flatten to tabular format
# ----------------------------------------------------------------------

def flatten_with_values(X: np.ndarray,
                        Y: np.ndarray,
                        values: np.ndarray,
                        value_name: str = "Value") -> pd.DataFrame:
    """Flattens 2D outreach/height/value matrices into a tidy DataFrame."""
    df = pd.DataFrame({
        "Outreach [m]": X.ravel(),
        "Height [m]": Y.ravel(),
        value_name: values.ravel(),
    })
    return df


# ----------------------------------------------------------------------
# Harbour capacity data loader
# ----------------------------------------------------------------------

def load_value_grid(filename: str, data_dir: str = "data") -> pd.DataFrame:
    """Loads and cleans a capacity or similar grid file."""
    _, _, df = _read_angle_grid(filename, data_dir)
    df = _interp_fill(df)
    return df
