import numpy as np
from scipy.interpolate import RegularGridInterpolator
import shapely.geometry as sgeom
import alphashape

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
    Return Nx2 array (x,y) for outer boundary. Uses alphashape (concave) if available,
    otherwise convex hull. Robust to duplicates and NaNs.
    """
    pts = np.asarray(xy_points, float)
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts) < 3:
        return None
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return None

    poly = None
    if prefer_concave:
        try:
            pts_cap = _sample_points(pts, MAX_POINTS_FOR_ENVELOPE)
            alpha = _estimate_alpha(pts_cap)
            poly = alphashape.alphashape(pts_cap, alpha)
        except Exception:
            poly = None

    if poly is None:
        poly = sgeom.MultiPoint(pts).convex_hull

    if isinstance(poly, sgeom.MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    if isinstance(poly, sgeom.Polygon):
        x, y = poly.exterior.coords.xy
        return np.column_stack([x, y])
    if isinstance(poly, sgeom.LineString):
        x, y = poly.coords.xy
        return np.column_stack([x, y])
    return None
