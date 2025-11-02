import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import ConvexHull

# Optional concave geometry support (alphashape + shapely)
try:
    import alphashape
    _HAS_ALPHASHAPE = True
except Exception:
    alphashape = None
    _HAS_ALPHASHAPE = False

# Performance limits
MAX_POINTS_FOR_DISPLAY = 15000
MAX_POINTS_FOR_ENVELOPE = 6000
MAX_POINTS_FOR_KNN = 1200
KNN_K = 8


# ---------------------------------------------------------------------
# Interpolation and resampling utilities
# ---------------------------------------------------------------------
def build_interpolators(height_df, outreach_df, load_df=None):
    """
    Return (fold_angles, main_angles, height_itp, outreach_itp, load_itp)
    as RegularGridInterpolators.
    """
    fold_angles = height_df.index.values.astype(float)
    main_angles = height_df.columns.values.astype(float)
    outreach_df = outreach_df.reindex(index=fold_angles, columns=main_angles)

    height_itp = RegularGridInterpolator(
        (fold_angles, main_angles),
        height_df.values,
        bounds_error=False,
        fill_value=None,
    )
    outre_itp = RegularGridInterpolator(
        (fold_angles, main_angles),
        outreach_df.values,
        bounds_error=False,
        fill_value=None,
    )

    load_itp = None
    if load_df is not None and not load_df.empty:
        load_df = load_df.reindex(index=fold_angles, columns=main_angles)
        load_itp = RegularGridInterpolator(
            (fold_angles, main_angles),
            load_df.values,
            bounds_error=False,
            fill_value=None,
        )

    return fold_angles, main_angles, height_itp, outre_itp, load_itp


def resample_grid_by_factors(fold_angles, main_angles, fold_factor, main_factor):
    """
    Subdivide each angular interval by a given factor.
    Returns fnew, mnew, F, M, pts (Nx2 array).
    """
    def expand(arr, factor):
        arr = np.unique(np.asarray(arr, float))
        arr.sort()
        if len(arr) < 2 or int(factor) <= 1:
            return arr
        out = [arr[0]]
        for i in range(len(arr) - 1):
            a, b = arr[i], arr[i + 1]
            seg = np.linspace(a, b, int(factor) + 1, endpoint=True)[1:]
            out.extend(seg.tolist())
        return np.array(out, float)

    fnew = expand(fold_angles, fold_factor)
    mnew = expand(main_angles, main_factor)
    F, M = np.meshgrid(fnew, mnew, indexing="ij")
    pts = np.column_stack([F.ravel(), M.ravel()])  # [Folding, Main]
    return fnew, mnew, F, M, pts


def _sample_points(arr, max_n):
    """
    Uniformly sample up to max_n points for plotting or envelope building.
    """
    n = len(arr)
    if n <= max_n:
        return arr
    idx = np.linspace(0, n - 1, num=max_n, dtype=int)
    return arr[idx]


# ---------------------------------------------------------------------
# Envelope / boundary computation
# ---------------------------------------------------------------------
def compute_boundary_curve(xy_points, prefer_concave=True, alpha_scale=1.0):
    """
    Compute a 2D boundary curve around a set of (x,y) points.

    Parameters
    ----------
    xy_points : ndarray (N,2)
        Input coordinates (Outreach, Height)
    prefer_concave : bool
        If True and alphashape is available, compute a concave hull (alpha shape)
    alpha_scale : float
        Multiplier for the automatically estimated alpha parameter (default 1.0)

    Returns
    -------
    ndarray (M,2) or None
        Boundary polygon vertices, closed loop (x,y)
    """
    pts = np.asarray(xy_points, float)
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts) < 3:
        return None

    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return None

    # --- Concave attempt ---
    if prefer_concave and _HAS_ALPHASHAPE:
        try:
            # Light subsample for performance
            n = min(len(pts), MAX_POINTS_FOR_ENVELOPE)
            idx = np.linspace(0, len(pts) - 1, n, dtype=int)
            pts_s = pts[idx]

            # Automatically optimise alpha for shape tightness
            alpha_opt = alphashape.optimizealpha(pts_s)
            alpha = max(1e-7, float(alpha_scale or 1.0) * float(alpha_opt))

            poly = alphashape.alphashape(pts_s, alpha)

            # Handle MultiPolygon by picking the largest
            if hasattr(poly, "geoms"):
                poly = max(poly.geoms, key=lambda g: g.area)

            if hasattr(poly, "exterior"):
                x, y = poly.exterior.coords.xy
                out = np.column_stack([x, y])
                if len(out) >= 3:
                    return out
        except Exception:
            pass  # fall back to convex hull

    # --- Convex hull fallback ---
    try:
        hull = ConvexHull(pts)
        cycle = np.append(hull.vertices, hull.vertices[0])
        return pts[cycle]
    except Exception:
        return None
