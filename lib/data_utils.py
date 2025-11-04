import os
import csv
import numpy as np
import pandas as pd


def _read_matrix(path: str) -> pd.DataFrame:
    """
    Read a CSV that may use ; , or tab delimiters and may contain decimal commas.
    Returns a numeric DataFrame (no header), with empty rows/cols removed.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    # Auto-detect delimiter
    with open(path, "r", newline="") as f:
        sample = f.read(2048)
        try:
            sniff = csv.Sniffer().sniff(sample)
            delim = sniff.delimiter
        except Exception:
            # Fall back: prefer ; if present, else comma, else tab
            if ";" in sample:
                delim = ";"
            elif "," in sample:
                delim = ","
            else:
                delim = "\t"

    df = pd.read_csv(path, sep=delim, header=None, engine="python")

    # Normalize decimal commas -> dots, coerce to numeric
    df = df.applymap(lambda v: str(v).replace(",", ".") if isinstance(v, str) else v)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Remove entirely empty rows/cols
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all").reset_index(drop=True)

    # If the file is a single row with many numbers (e.g., "0;1;2;..."), keep as 1 x N
    # If it's a single column with many numbers, keep as N x 1 — already handled.

    return df


def load_crane_points(data_dir: str = "data") -> pd.DataFrame:
    """
    Load outreach.csv (X) and height.csv (Y) as matrices, broadcast if needed,
    and return a tidy DataFrame with ALL valid (x, y) pairs from the grid.
    """
    outreach_path = os.path.join(data_dir, "outreach.csv")
    height_path   = os.path.join(data_dir, "height.csv")

    X = _read_matrix(outreach_path)
    Y = _read_matrix(height_path)

    # Broadcast logic:
    # 1) Same shape → pair elementwise
    # 2) One is vector (1xN or Nx1) and the other is matrix → broadcast along matching axis
    def _as_numpy(df): return df.to_numpy(dtype=float, copy=False)

    Xn, Yn = _as_numpy(X), _as_numpy(Y)

    def _broadcast(a: np.ndarray, target_shape):
        """Broadcast 1D row/col vectors to target_shape when possible."""
        if a.shape == target_shape:
            return a
        r, c = target_shape
        if a.ndim == 2:
            # Row vector 1 x N
            if a.shape[0] == 1 and a.shape[1] == c:
                return np.tile(a, (r, 1))
            # Column vector N x 1
            if a.shape[1] == 1 and a.shape[0] == r:
                return np.tile(a, (1, c))
        # Also allow flat vector length match to rows or cols
        if a.size == c:  # treat as row vector
            return np.tile(a.reshape(1, c), (r, 1))
        if a.size == r:  # treat as column vector
            return np.tile(a.reshape(r, 1), (1, c))
        raise ValueError(f"Cannot broadcast shape {a.shape} to {target_shape}")

    if Xn.shape != Yn.shape:
        # Try to broadcast the smaller structure to the bigger one's shape
        # Decide target by picking the larger #elements
        target = Xn if Xn.size >= Yn.size else Yn
        target_shape = target.shape
        try:
            Xn = _broadcast(Xn, target_shape)
        except Exception:
            pass
        try:
            Yn = _broadcast(Yn, target_shape)
        except Exception:
            pass

        if Xn.shape != Yn.shape:
            raise ValueError(f"Outreach and Height shapes incompatible: {Xn.shape} vs {Yn.shape}")

    # Mask invalid pairs
    mask = ~np.isnan(Xn) & ~np.isnan(Yn)

    xs = Xn[mask].ravel()
    ys = Yn[mask].ravel()

    return pd.DataFrame({"Outreach [m]": xs, "Height [m]": ys})
