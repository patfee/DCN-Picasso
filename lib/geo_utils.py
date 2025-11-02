import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import ConvexHull

# optional concave geometry support
try:
    import alphashape
    _HAS_ALPHASHAPE = True
except Exception:
    alphashape = None
    _HAS_ALPHASHAPE = False

# Limits for performance
MAX_POINTS_FOR_DISPLAY = 15000
MAX_POINTS_FOR_ENVELOPE = 6000
MAX_POINTS_FOR_KNN = 1200
KNN_K = 8


def build_interpolators(height_df, outreach_df, load_df=None):
    """Return fold_angles, main_angles, interpolators for height/outreach/load."""
    fold_angles = height_df.index.values.astype(float)
    main_angles = height_df.columns.values.astype(float)
    outreach_df = outreach_df.reindex(index=fold_angles, columns=main_angles)

    height_itp = RegularGridInterpolator(
        (fold_angles, main_angles), height_df.values,
        bounds_error=False, fill_value=None
    )
    outre_itp = RegularGridInterpolator(
        (fold_angles, main_angles), outreach_df.values,
        bounds_error=False, fill_value=None
    )

    load_itp = None
    if load_df is not None and not load_df.empty:
        load_df = load_df.reindex(index=fold_angles, columns=main_angles)
        load_itp = RegularGridInterpolator(
            (fold_angles, main_angles), load_df.values,
            bounds_error=False, fill_value=None
        )
    return fold_angles, main_angles, height_itp, outre_itp, load_itp


def resample_grid_by_factors(fold_angles, main_angles, fold_factor, main_factor):
    """Subdivide each angle interval by a factor."""
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


def compute_boundary_curve(xy_points, prefer_concave=True, alpha_scale=1.0):
    """
    Return Nx2 array (x,y) for outer boundary.
      - Concave: alphashape if available, scaled by alpha_scale
      - Fallback: ConvexHull (SciPy)
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
            n = min(len(pts), MAX_POINTS_FOR_ENVELOPE)
            idx = np.linspace(0, len(pts) - 1, n, dtype=int)
            pts_s = pts[idx]

            # Estimate alpha (inverse of neighbor distance)
            A = pts_s[:, None, :] - pts_s[None, :, :]
            D = np.sqrt((A ** 2).sum(axis=2))
            D.sort(axis=1)
            k = min(8, len(pts_s) - 2)
            dk = np.median(D[:, k]) if k >= 1 else np.median(D)
            base_alpha = 1.0 / (1.8 * dk) if np.isfinite(dk) and dk > 0 else 1.0
            alpha = max(1e-6, float(alpha_scale) * base_alpha)

            poly = alphashape.alphashape(pts_s, alpha)
            if hasattr(poly, "geoms"):  # MultiPolygon -> largest
                poly = max(poly.geoms, key=lambda g: g.area)
            if hasattr(poly, "exterior"):
                x, y = poly.exterior.coords.xy
                out = np.column_stack([x, y])
                if len(out) >= 3:
                    return out
        except Exception:
            pass  # fall back

    # --- Convex hull fallback ---
    try:
        hull = ConvexHull(pts)
        cycle = np.append(hull.vertices, hull.vertices[0])
        return pts[cycle]
    except Exception:
        return None
