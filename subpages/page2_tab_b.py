from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from matplotlib.path import Path

from lib.data_utils import (
    get_position_grids,       # XY grids aligned with Page 1 (mode, factors, pedestal)
    load_value_grid,          # loads angle-grid CSV (Harbour_Cdyn115.csv)
    interpolate_value_grid,   # interpolates that value grid to current angle grid
    flatten_with_values,      # flattens XY+values to a tidy table
)

VALUE_FILE  = "Harbour_Cdyn115.csv"
VALUE_LABEL = "Capacity [t]"

# Band edges (t) – as in the manual
ISO_LEVELS = [0, 35, 70, 105, 140]

# Colors broadly matching the manual
COLORSCALE = [
    [0.00, "#003b46"],  # deep teal
    [0.20, "#00b3c6"],  # cyan
    [0.40, "#9ecf2a"],  # green-yellow
    [0.60, "#ffcc33"],  # yellow
    [0.80, "#ff8840"],  # orange
    [1.00, "#cc2f2f"],  # red
]

BAND_COLORS = ["#00b3c6", "#9ecf2a", "#ffcc33", "#ff8840"]  # legend swatches


def _legend_card() -> dbc.Card:
    rows = []
    for i in range(len(ISO_LEVELS) - 1):
        lo, hi = ISO_LEVELS[i], ISO_LEVELS[i + 1]
        color = BAND_COLORS[i % len(BAND_COLORS)]
        rows.append(
            html.Div(
                className="d-flex align-items-center mb-1",
                children=[
                    html.Div(
                        style={
                            "width": "16px",
                            "height": "16px",
                            "backgroundColor": color,
                            "border": "1px solid rgba(0,0,0,0.2)",
                            "marginRight": "8px",
                            "borderRadius": "3px",
                        }
                    ),
                    html.Span(f"{lo:g} – {hi:g} t"),
                ],
            )
        )
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div("Iso-capacity bands (Cdyn 1.15)", className="fw-semibold mb-2"),
                *rows,
                html.Div(
                    "Bands follow your Page 1 interpolation & subdivisions.",
                    className="text-muted small mt-2",
                ),
            ]
        ),
        className="mt-3",
        style={"maxWidth": "260px"},
    )


# ---------- Concave hull (alpha-shape) utilities (no Shapely) ----------

def _triangle_circumradius(a, b, c):
    """Circumradius of triangle ABC (edges as vectors)."""
    A = np.linalg.norm(b - c)
    B = np.linalg.norm(c - a)
    C = np.linalg.norm(a - b)
    s = (A + B + C) / 2.0
    area = max(s * (s - A) * (s - B) * (s - C), 0.0) ** 0.5
    if area == 0:
        return np.inf
    return (A * B * C) / (4.0 * area)

def _alpha_shape(points, alpha):
    """
    Alpha-shape polygon (as ordered vertices) from 2D points.
    Keeps triangles with circumradius < 1/alpha (smaller alpha => tighter hull).
    Returns Nx2 array of polygon vertices ordered as a boundary.
    """
    if len(points) < 4:
        # trivial: return convex hull order
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        return points[hull.vertices]

    tri = Delaunay(points)
    triangles = tri.simplices
    keep = []
    inv_alpha = 1.0 / max(alpha, 1e-9)

    for tri_ix in triangles:
        pa, pb, pc = points[tri_ix]
        R = _triangle_circumradius(pa, pb, pc)
        if R < inv_alpha:
            keep.append(tri_ix)

    if not keep:
        # fallback to convex hull if alpha too small
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        return points[hull.vertices]

    # Build a set of boundary edges (edges that appear only once)
    edges = {}
    def add_edge(i, j):
        if i > j:
            i, j = j, i
        edges[(i, j)] = edges.get((i, j), 0) + 1

    for tri_ix in keep:
        i, j, k = tri_ix
        add_edge(i, j); add_edge(j, k); add_edge(k, i)

    boundary = [e for e, cnt in edges.items() if cnt == 1]
    if not boundary:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        return points[hull.vertices]

    # Order boundary edges into a polygon path
    # Build adjacency
    adj = {}
    for i, j in boundary:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)

    # Start from a node with degree 1 if available (should be 2 for a closed loop)
    start = boundary[0][0]
    poly = [start]
    prev = None
    curr = start
    while True:
        nxts = [n for n in adj[curr] if n != prev]
        if not nxts:
            break
        nxt = nxts[0]
        poly.append(nxt)
        prev, curr = curr, nxt
        if curr == start:
            break

    poly = np.array(poly, dtype=int)
    poly = np.unique(poly, return_index=True)[1].argsort()  # ensure unique order
    # Rebuild with original indices
    ordered_idx = [int(list(set([b for e in boundary for b in e]))[0])]
    # More robust ordering
    used = set()
    ordered = []
    curr = start
    prev = None
    while True:
        ordered.append(curr)
        used.add(curr)
        nxts = [n for n in adj[curr] if n != prev]
        nxts = [n for n in nxts if n not in used] or [nxts[0] if nxts else start]
        nxt = nxts[0]
        prev, curr = curr, nxt
        if curr == start:
            break

    return points[np.array(ordered)]


# ---------- Plot builder ----------

def _build_contour_figure(df: pd.DataFrame, include_pedestal: bool, show_samples: bool) -> go.Figure:
    """Rasterize scattered points to a grid, mask to concave hull, and draw iso-bands."""
    if df.empty:
        return go.Figure()

    x = df["Outreach [m]"].to_numpy()
    y = df["Height [m]"].to_numpy()
    z = df[VALUE_LABEL].to_numpy()

    pts = np.column_stack([x, y])

    # Concave hull (alpha in meters ~ controls tightness). 0.25–0.6 usually good.
    alpha = 0.35
    hull_xy = _alpha_shape(pts, alpha=alpha)

    # Build raster canvas
    x_pad = 0.03 * (x.max() - x.min() + 1e-9)
    y_pad = 0.03 * (y.max() - y.min() + 1e-9)
    xi = np.linspace(x.min() - x_pad, x.max() + x_pad, 400)
    yi = np.linspace(y.min() - y_pad, y.max() + y_pad, 400)
    XI, YI = np.meshgrid(xi, yi)

    # Linear interpolation (no extrapolation)
    Z = griddata(points=pts, values=z, xi=(XI, YI), method="linear")

    # Mask outside the concave hull
    hull_path = Path(hull_xy)
    mask_flat = hull_path.contains_points(np.column_stack([XI.ravel(), YI.ravel()]))
    mask = mask_flat.reshape(XI.shape)
    Z_masked = np.where(mask, Z, np.nan)

    band_step = ISO_LEVELS[1] - ISO_LEVELS[0] if len(ISO_LEVELS) > 1 else 35
    fig = go.Figure()

    # Filled bands
    fig.add_trace(
        go.Contour(
            x=xi,
            y=yi,
            z=Z_masked,
            contours=dict(
                coloring="fill",
                showlines=True,
                start=min(ISO_LEVELS),
                end=max(ISO_LEVELS),
                size=band_step,
            ),
            colorscale=COLORSCALE,
            colorbar=dict(title=VALUE_LABEL, tickvals=ISO_LEVELS),
            line=dict(width=1, color="rgba(0,0,0,0.7)"),
            hovertemplate=(
                "Outreach: %{x:.2f} m<br>"
                "Height: %{y:.2f} m<br>"
                f"{VALUE_LABEL}: %{{z:.1f}}<extra></extra>"
            ),
            showscale=True,
        )
    )

    # Optional sample points
    if show_samples:
        fig.add_trace(
            go.Scattergl(
                x=x, y=y, mode="markers",
                marker=dict(size=3, opacity=0.28),
                name="Samples", hoverinfo="skip",
            )
        )

    # Emphasize 70 t & 105 t isolines
    for level in (70, 105):
        fig.add_trace(
            go.Contour(
                x=xi, y=yi, z=Z_masked,
                contours=dict(coloring="none", showlines=True, start=level, end=level, size=1e-6),
                line=dict(color="#ffd000", width=2.5),
                showscale=False, hoverinfo="skip",
                name=f"{level} t line",
            )
        )

    # Black outer envelope (concave hull)
    fig.add_trace(
        go.Scatter(
            x=np.r_[hull_xy[:, 0], hull_xy[0, 0]],
            y=np.r_[hull_xy[:, 1], hull_xy[0, 1]],
            mode="lines",
            line=dict(color="#1b1b1b", width=2.0),
            name="Envelope",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title="Harbour lift — Cdyn = 1.15 (iso-capacity hulls)",
        xaxis_title="Outreach [m]",
        yaxis_title=("Jib head above deck level [m]" if include_pedestal
                     else "Jib head above pedestal flange [m]"),
        template="plotly_white",
        height=760, margin=dict(l=40, r=20, t=60, b=40),
    )

    return fig


# ---------- Dash layout & callbacks ----------

layout = html.Div(
    [
        html.H5("Page 2 – Sub B: Harbour lift (iso-capacity hulls)"),
        html.Div("Aligned with Page 1 settings (linear/spline, subdivisions, pedestal).",
                 className="mb-2 small text-muted"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Checklist(
                        id="tabb-show-samples",
                        options=[{"label": "Show sample points overlay", "value": "on"}],
                        value=["on"], switch=True,
                    ),
                    md="auto",
                ),
            ],
            className="mb-2",
        ),
        dcc.Graph(id="harbour-cdyn115-contours"),
        _legend_card(),
    ]
)


@callback(
    Output("harbour-cdyn115-contours", "figure"),
    Input("app-config", "data"),
    Input("tabb-show-samples", "value"),
)
def update_iso_hulls(config, show_samples_value):
    include = bool(config.get("include_pedestal", False)) if config else False
    mode    = (config.get("interp_mode") or "linear").lower() if config else "linear"
    if mode not in {"linear", "spline"}:
        mode = "linear"

    # 1) XY grid to match Page 1
    Xgrid, Ygrid, new_main, new_fold = get_position_grids(config=config, data_dir="data")

    # 2) Interpolate Harbour capacity to the same angle grid, then flatten to XY+value
    V_orig = load_value_grid(VALUE_FILE, data_dir="data")
    Vgrid  = interpolate_value_grid(V_orig, new_main, new_fold, mode=mode)
    df = flatten_with_values(Xgrid, Ygrid, Vgrid, value_name=VALUE_LABEL)

    show_samples = "on" in (show_samples_value or [])
    fig = _build_contour_figure(df, include_pedestal=include, show_samples=show_samples)
    return fig
