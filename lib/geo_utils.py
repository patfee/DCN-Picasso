import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import ConvexHull

# --- keep the rest of your imports & constants as-is ---
# MAX_POINTS_FOR_DISPLAY, MAX_POINTS_FOR_ENVELOPE, MAX_POINTS_FOR_KNN, KNN_K, etc.

# Add this small safe-import block at the top (below other imports):
try:
    import alphashape  # optional
    _HAS_ALPHASHAPE = True
except Exception:
    alphashape = None
    _HAS_ALPHASHAPE = False

# caps to keep plots responsive
MAX_POINTS_FOR_DISPLAY = 15000
MAX_POINTS_FOR_ENVELOPE = 6000
MAX_POINTS_FOR_KNN = 1200
KNN_K = 8

def build_interpolators(height_df, outreach_df, load_df=None):
    """
    Returns:
      fold_angles, main_angles, height_itp, outre_itp, load_itp (load_itp may be None)
    Aligns all matrices to the Height grid (index=folding, columns=main).
    """
    fold_angles = height_df.index.values.astype(float)
    main_angles = height_df.columns.values.astype(float)
    outreach_df = outreach_df.reindex(index=fold_angles, columns=main_angles)

    height_itp = RegularGridInterpolator(
        (fold_angles, main_angles), height_df.values, bounds_error=False, fill_value=None
    )
    outre_itp = RegularGridInterpolator(
        (fold_angles, main_angles), outreach_df.values, bounds_error=False, fill_value=None
    )

    load_itp = None
    if load_df is not None:
        load_df = load_df.reindex(index=fold_angles, columns=main_angles)
        load_itp = RegularGridInterpolator(
            (fold_angles, main_angles), load_df.values, bounds_error=False, fill_value=None
        )
    return fold_angles, main_angles, height_itp, outre_itp, load_itp

def resample_grid_by_factors(fold_angles, main_angles, fold_factor, main_factor):
    """
    Per-interval subdivision:
    factor=1 -> original grid
    factor=2 -> one extra point between originals, etc.
    Returns: fnew, mnew, F, M, pts[[fold, main], ...]
    """
    def expand(arr, f):
        arr = np.unique(np.asarray(arr, float))
        arr.sort()
        if len(arr) < 2 or int(f) <= 1:
            return arr
        out = [arr[0]]
        for i in range(len(arr) - 1):
            a, b = arr[i], arr[i + 1]
            seg = np.linspace(a, b, int(f) + 1, endpoint=True)[1:]
            out.extend(seg.tolist())
        return np.array(out, float)

    fnew = expand(fold_angles, fold_factor)
    mnew = expand(main_angles, main_factor)
    F, M = np.meshgrid(fnew, mnew, indexing="ij")
    pts = np.column_stack([F.ravel(), M.ravel()])  # [Folding, Main]
    return fnew, mnew, F, M, pts

def _sample_points(arr, max_n):
    n = len(arr)
    if n <= max_n:
        return arr
    idx = np.linspace(0, n - 1, num=max_n, dtype=int)
    return arr[idx]

def _estimate_alpha(pts, k=KNN_K):
    pts = _sample_points(pts, MAX_POINTS_FOR_KNN)
    if len(pts) < k + 2:
        return 1.0
    A = pts[:, None, :] - pts[None, :, :]
    D = np.sqrt((A**2).sum(axis=2))
    D.sort(axis=1)
    dk = np.median(D[:, k])
    if not np.isfinite(dk) or dk <= 1e-9:
        return 1.0
    return 1.0 / (1.8 * dk)

def compute_boundary_curve(xy_points, prefer_concave=True):
    """
    Return Nx2 array (x,y) for outer boundary.
      - If prefer_concave and alphashape is available, try concave alpha-shape.
      - Otherwise (or on failure) fall back to SciPy ConvexHull.
    Robust to NaNs and duplicate points. Returns None if not enough points.
    """
    pts = np.asarray(xy_points, float)
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts) < 3:
        return None

    # De-duplicate to help hull robustness
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return None

    # Concave attempt (optional)
    if prefer_concave and _HAS_ALPHASHAPE:
        try:
            # Light subsample for performance
            n = min(len(pts), MAX_POINTS_FOR_ENVELOPE)
            idx = np.linspace(0, len(pts) - 1, n, dtype=int)
            pts_s = pts[idx]

            # Estimate alpha ~ inverse of typical neighbor distance
            A = pts_s[:, None, :] - pts_s[None, :, :]
            D = np.sqrt((A ** 2).sum(axis=2))
            D.sort(axis=1)
            k = min(8, len(pts_s) - 2)
            dk = np.median(D[:, k]) if k >= 1 else np.median(D)
            alpha = 1.0 / (1.8 * dk) if np.isfinite(dk) and dk > 0 else 1.0

            poly = alphashape.alphashape(pts_s, alpha)
            # Extract exterior ring if polygon
            try:
                # Shapely-like geometry interface expected
                if hasattr(poly, "geoms"):  # MultiPolygon, pick largest
                    poly = max(poly.geoms, key=lambda g: g.area)
                if hasattr(poly, "exterior"):
                    x, y = poly.exterior.coords.xy
                    out = np.column_stack([x, y])
                    if len(out) >= 3:
                        return out
            except Exception:
                pass
        except Exception:
            pass  # fall back to convex hull

    # Convex hull fallback (pure SciPy)
    try:
        hull = ConvexHull(pts)
        cycle = np.append(hull.vertices, hull.vertices[0])
        return pts[cycle]
    except Exception:
        return None
