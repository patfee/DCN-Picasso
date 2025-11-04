from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import Delaunay, ConvexHull

from lib.data_utils import (
    get_position_grids,
    load_value_grid,
    interpolate_value_grid,
    flatten_with_values,
)

VALUE_FILE  = "Harbour_Cdyn115.csv"
VALUE_LABEL = "Capacity [t]"

ISO_LEVELS = [0, 35, 70, 105, 140]
COLORSCALE = [
    [0.00, "#003b46"],  # deep teal
    [0.20, "#00b3c6"],  # cyan
    [0.40, "#9ecf2a"],  # green-yellow
    [0.60, "#ffcc33"],  # yellow
    [0.80, "#ff8840"],  # orange
    [1.00, "#cc2f2f"],  # red
]
BAND_COLORS = ["#00b3c6", "#9ecf2a", "#ffcc33", "#ff8840"]


def _legend_card() -> dbc.Card:
    rows = []
    for i in range(len(ISO_LEVELS) - 1):
        lo, hi = ISO_LEVELS[i], ISO_LEVELS[i + 1]
        color = BAND_COLORS[i % len(BAND_COLORS)]
        rows.append(
            html.Div(
                className="d-flex align-items-center mb-1",
                children=[
                    html.Div(style={
                        "width": "16px", "height": "16px",
                        "backgroundColor": color,
                        "border": "1px solid rgba(0,0,0,0.2)",
                        "marginRight": "8px", "borderRadius": "3px"
                    }),
                    html.Span(f"{lo:g} – {hi:g} t"),
                ],
            )
        )
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div("Iso-capacity bands (Cdyn 1.15)", className="fw-semibold mb-2"),
                *rows,
                html.Div("Bands follow your Page 1 interpolation & subdivisions.",
                         className="text-muted small mt-2"),
            ]
        ),
        className="mt-3",
        style={"maxWidth": "260px"},
    )


# ----------------- Concave hull (alpha-shape) without extra deps -----------------

def _triangle_circumradius(a, b, c):
    A = np.linalg.norm(b - c)
    B = np.linalg.norm(c - a)
    C = np.linalg.norm(a - b)
    s = (A + B + C) / 2.0
    area = max(s * (s - A) * (s - B) * (s - C), 0.0) ** 0.5
    if area == 0:
        return np.inf
    return (A * B * C) / (4.0 * area)

def _alpha_shape(points: np.ndarray, alpha: float) -> np.ndarray:
    """Return ordered polygon vertices of an alpha-shape (concave hull)."""
    pts = np.asarray(points, float)
    if len(pts) < 4:
        hull = ConvexHull(pts)
        return pts[hull.vertices]

    tri = Delaunay(pts)
    keep = []
    inv_alpha = 1.0 / max(alpha, 1e-9)
    for simp in tri.simplices:
        pa, pb, pc = pts[simp]
        if _triangle_circumradius(pa, pb, pc) < inv_alpha:
            keep.append(simp)
    if not keep:
        hull = ConvexHull(pts)
        return pts[hull.vertices]

    # edges that appear once = boundary
    edges = {}
    def add_edge(i, j):
        if i > j:
            i, j = j, i
        edges[(i, j)] = edges.get((i, j), 0) + 1
    for simp in keep:
        i, j, k = simp
        add_edge(i, j); add_edge(j, k); add_edge(k, i)
    boundary = [e for e, c in edges.items() if c == 1]
    if not boundary:
        hull = ConvexHull(pts)
        return pts[hull.vertices]

    # order boundary into a closed loop
    adj = {}
    for i, j in boundary:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)

    start = boundary[0][0]
    poly_idx = [start]
    prev = None
    curr = start
    while True:
        nbrs = [n for n in adj[curr] if n != prev]
        if not nbrs:
            break
        nxt = nbrs[0]
        poly_idx.append(nxt)
        prev, curr = curr, nxt
        if curr == start:
            break
    return pts[np.array(poly_idx)]


# --------------- Fast vectorized point-in-polygon (ray casting) ------------------

def _points_in_polygon(xx: np.ndarray, yy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """
    Return boolean mask of shape xx.shape for points (xx, yy) inside polygon 'poly'.
    Vectorized ray-casting without third-party deps.
    """
    x = xx.ravel()
    y = yy.ravel()
    n = len(poly)
    px = poly[:, 0]
    py = poly[:, 1]

    inside = np.zeros(x.shape, dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        # edge crosses horizontal ray?
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        inside ^= cond
        j = i
    return inside.reshape(xx.shape)


# ------------------------------- Plot builder -----------------------------------

def _build_contour_figure(df: pd.DataFrame, include_pedestal: bool, show_samples: bool) -> go.Figure:
    if df.empty:
        return go.Figure()

    x = df["Outreach [m]"].to_numpy()
    y = df["Height [m]"].to_numpy()
    z = df[VALUE_LABEL].to_numpy()
    pts = np.column_stack([x, y])

    # Concave hull to clip colors tightly to the valid region
    alpha = 0.35  # lower = tighter; tune 0.30–0.45 to best match your manual
    hull_xy = _alpha_shape(pts, alpha=alpha)

    # Raster canvas
    x_pad = 0.03 * (x.max() - x.min() + 1e-9)
    y_pad = 0.03 * (y.max() - y.min() + 1e-9)
    xi = np.linspace(x.min() - x_pad, x.max() + x_pad, 400)
    yi = np.linspace(y.min() - y_pad, y.max() + y_pad, 400)
    XI, YI = np.meshgrid(xi, yi)

    # Interpolate (linear). Outside sample hull remains NaN.
    Z = griddata(points=pts, values=z, xi=(XI, YI), method="linear")

    # Mask outside concave hull (no matplotlib needed)
    mask = _points_in_polygon(XI, YI, hull_xy)
    Z_masked = np.where(mask, Z, np.nan)

    band_step = ISO_LEVELS[1] - ISO_LEVELS[0] if len(ISO_LEVELS) > 1 else 35
    fig = go.Figure()

    fig.add_trace(go.Contour(
        x=xi, y=yi, z=Z_masked,
        contours=dict(coloring="fill", showlines=True,
                      start=min(ISO_LEVELS), end=max(ISO_LEVELS), size=band_step),
        colorscale=COLORSCALE,
        colorbar=dict(title=VALUE_LABEL, tickvals=ISO_LEVELS),
        line=dict(width=1, color="rgba(0,0,0,0.7)"),
        hovertemplate=("Outreach: %{x:.2f} m<br>"
                       "Height: %{y:.2f} m<br>"
                       f"{VALUE_LABEL}: %{{z:.1f}}<extra></extra>"),
        showscale=True,
    ))

    if show_samples:
        fig.add_trace(go.Scattergl(
            x=x, y=y, mode="markers",
            marker=dict(size=3, opacity=0.28),
            name="Samples", hoverinfo="skip",
        ))

    for level in (70, 105):  # emphasize key isolines
        fig.add_trace(go.Contour(
            x=xi, y=yi, z=Z_masked,
            contours=dict(coloring="none", showlines=True, start=level, end=level, size=1e-6),
            line=dict(color="#ffd000", width=2.5),
            showscale=False, hoverinfo="skip",
            name=f"{level} t line",
        ))

    # Black outer envelope
    fig.add_trace(go.Scatter(
        x=np.r_[hull_xy[:, 0], hull_xy[0, 0]],
        y=np.r_[hull_xy[:, 1], hull_xy[0, 1]],
        mode="lines",
        line=dict(color="#1b1b1b", width=2.0),
        name="Envelope", hoverinfo="skip",
    ))

    fig.update_layout(
        title="Harbour lift — Cdyn = 1.15 (iso-capacity hulls)",
        xaxis_title="Outreach [m]",
        yaxis_title=("Jib head above deck level [m]" if include_pedestal
                     else "Jib head above pedestal flange [m]"),
        template="plotly_white",
        height=760, margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ---------------------------- Dash layout + callback ----------------------------

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

    # Align to Page 1 grids
    Xgrid, Ygrid, new_main, new_fold = get_position_grids(config=config, data_dir="data")

    # Interpolate Harbour capacity to the same angle grid
    V_orig = load_value_grid(VALUE_FILE, data_dir="data")
    Vgrid  = interpolate_value_grid(V_orig, new_main, new_fold, mode=mode)
    df     = flatten_with_values(Xgrid, Ygrid, Vgrid, value_name=VALUE_LABEL)

    show_samples = "on" in (show_samples_value or [])
    return _build_contour_figure(df, include_pedestal=include, show_samples=show_samples)
