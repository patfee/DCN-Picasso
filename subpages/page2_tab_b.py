from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from functools import lru_cache
from scipy.spatial import Delaunay, ConvexHull
from scipy.interpolate import LinearNDInterpolator

from lib.data_utils import (
    get_position_grids,     # XY grids aligned with Page 1 (mode, factors, pedestal)
    load_value_grid,        # standard loader (we'll fall back to a local robust loader if needed)
    interpolate_value_grid,
    flatten_with_values,
)

# ---------------------------- Config ---------------------------------

DATA_DIR    = "data"
VALUE_FILE  = "Harbour_Cdyn115.csv"
VALUE_LABEL = "Capacity [t]"

# Iso-band edges (t)
ISO_LEVELS  = [0, 35, 70, 105, 140]
COLORSCALE  = [
    [0.00, "#003b46"],  # deep teal
    [0.20, "#00b3c6"],  # cyan
    [0.40, "#9ecf2a"],  # green-yellow
    [0.60, "#ffcc33"],  # yellow
    [0.80, "#ff8840"],  # orange
    [1.00, "#cc2f2f"],  # red
]

# Raster grid resolution (speed/quality knob). 256–384 are good values.
RASTER_N = 320


# ------------------------- Small UI helpers ---------------------------

def _legend_card() -> dbc.Card:
    swatches = ["#00b3c6", "#9ecf2a", "#ffcc33", "#ff8840"]
    rows = []
    for i in range(len(ISO_LEVELS) - 1):
        lo, hi = ISO_LEVELS[i], ISO_LEVELS[i + 1]
        rows.append(
            html.Div(
                className="d-flex align-items-center mb-1",
                children=[
                    html.Div(style={
                        "width": "16px", "height": "16px",
                        "backgroundColor": swatches[i % len(swatches)],
                        "border": "1px solid rgba(0,0,0,0.2)",
                        "marginRight": "8px", "borderRadius": "3px",
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
                html.Div("Bands follow Page 1 interpolation & subdivisions.",
                         className="text-muted small mt-2"),
            ]
        ),
        className="mt-3",
        style={"maxWidth": "260px"},
    )


# ---------------------- Robust value-grid loader ----------------------

def _try_read_angle_grid(path: str, sep: str) -> pd.DataFrame | None:
    try:
        df_raw = pd.read_csv(path, sep=sep, engine="python")
    except Exception:
        return None

    # Expect first row = main angles; first col = folding angles
    if df_raw.shape[0] < 2 or df_raw.shape[1] < 2:
        return None

    # parse headers (main) and index (folding)
    try:
        main_angles = [float(str(c).strip().replace(",", ".")) for c in df_raw.columns[1:]]
        fold_angles = [float(str(v).strip().replace(",", ".")) for v in df_raw.iloc[1:, 0].tolist()]
    except Exception:
        return None

    body = df_raw.iloc[1:, 1:].astype(str)
    # normalize decimals to dot
    body = body.replace({",": "."}, regex=True)

    try:
        body = body.apply(lambda s: pd.to_numeric(s, errors="coerce"))
    except Exception:
        return None

    body.index = fold_angles
    body.columns = main_angles
    return body


def _load_value_grid_robust(filename: str, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Try the standard loader first; if the result is empty or mostly NaN,
    attempt robust parsing with explicit separators and decimal normalization.
    """
    # 1) try standard loader
    try:
        df = load_value_grid(filename, data_dir=data_dir)
        if df.size and (df.notna().sum().sum() > 0):
            return df
    except Exception:
        pass

    # 2) robust attempts
    path = f"{data_dir}/{filename}"

    # a) explicit semicolon
    df = _try_read_angle_grid(path, sep=";")
    if df is not None and df.notna().sum().sum() > 0:
        return df.ffill(axis=1).bfill(axis=1).ffill(axis=0).bfill(axis=0)

    # b) explicit comma
    df = _try_read_angle_grid(path, sep=",")
    if df is not None and df.notna().sum().sum() > 0:
        return df.ffill(axis=1).bfill(axis=1).ffill(axis=0).bfill(axis=0)

    # c) last resort: read raw, normalize separators in memory, then parse as comma
    try:
        raw = open(path, encoding="utf-8").read()
        # If it looks like European CSV with ; and decimal ., keep ; as separator
        if raw.count(";") > raw.count(","):
            norm = raw
            # do not touch decimals here (assume already dot)
            tmp = norm
            from io import StringIO
            df_raw = pd.read_csv(StringIO(tmp), sep=";", engine="python")
        else:
            norm = raw.replace(";", ",")
            tmp = norm
            from io import StringIO
            df_raw = pd.read_csv(StringIO(tmp), sep=",", engine="python")

        if df_raw.shape[0] >= 2 and df_raw.shape[1] >= 2:
            main_angles = [float(str(c).strip().replace(",", ".")) for c in df_raw.columns[1:]]
            fold_angles = [float(str(v).strip().replace(",", ".")) for v in df_raw.iloc[1:, 0].tolist()]
            body = df_raw.iloc[1:, 1:].astype(str).replace({",": "."}, regex=True)
            body = body.apply(lambda s: pd.to_numeric(s, errors="coerce"))
            body.index = fold_angles
            body.columns = main_angles
            return body.ffill(axis=1).bfill(axis=1).ffill(axis=0).bfill(axis=0)
    except Exception:
        pass

    # If all failed, return empty DF to trigger a user-friendly message
    return pd.DataFrame()


# ------------------------------ Caching -------------------------------

def _cfg_key(config: dict | None):
    config = config or {}
    return (
        (config.get("interp_mode") or "linear").lower(),
        int(config.get("main_factor", 1)),
        int(config.get("folding_factor", config.get("fold_factor", 1))),
        bool(config.get("include_pedestal", False)),
        float(config.get("pedestal_height", 6.0)),
        RASTER_N,
    )

@lru_cache(maxsize=16)
def _compute_arrays(cfg_key):
    """
    Heavy part (cached):
      - Align XY to Page 1
      - Interp Harbour capacity to that angle grid (robust loader)
      - Build Delaunay once
      - Evaluate LinearNDInterpolator on a rect grid
      - Mask points outside triangulation (no color outside hull)
    Returns: xi, yi, Z_masked, hull_xy, include_flag, (sample_x, sample_y)
    """
    mode, main_f, fold_f, include, ped_h, raster_n = cfg_key
    config = dict(
        interp_mode=mode,
        main_factor=main_f,
        folding_factor=fold_f,
        include_pedestal=include,
        pedestal_height=ped_h,
    )

    # 1) XY grids (from Page 1 settings)
    Xgrid, Ygrid, new_main, new_fold = get_position_grids(config=config, data_dir=DATA_DIR)

    # 2) Interpolate Harbour capacity to the same angle grid
    V_orig = _load_value_grid_robust(VALUE_FILE, data_dir=DATA_DIR)  # robust
    if V_orig.empty or (not np.isfinite(V_orig.to_numpy()).any()):
        return None  # signal error to the callback

    Vgrid  = interpolate_value_grid(V_orig, new_main, new_fold, mode=mode)

    # 3) Flatten to scattered XY + Z (capacity)
    df = flatten_with_values(Xgrid, Ygrid, Vgrid, value_name=VALUE_LABEL)

    # Keep finite points only
    m = np.isfinite(df[VALUE_LABEL].to_numpy())
    x = df.loc[m, "Outreach [m]"].to_numpy()
    y = df.loc[m, "Height [m]"].to_numpy()
    z = df.loc[m, VALUE_LABEL].to_numpy()

    if x.size < 3:
        return None  # not enough points to triangulate

    pts = np.column_stack([x, y])

    # 4) Delaunay triangulation once (fast hull + barycentric eval)
    try:
        tri = Delaunay(pts)
    except Exception:
        return None

    # 5) Rectilinear canvas
    x_pad = 0.03 * (x.max() - x.min() + 1e-9)
    y_pad = 0.03 * (y.max() - y.min() + 1e-9)
    xi = np.linspace(x.min() - x_pad, x.max() + x_pad, raster_n)
    yi = np.linspace(y.min() - y_pad, y.max() + y_pad, raster_n)
    XI, YI = np.meshgrid(xi, yi)
    XY = np.column_stack([XI.ravel(), YI.ravel()])

    # 6) Fast scattered→grid with LinearNDInterpolator
    lin = LinearNDInterpolator(tri, z, fill_value=np.nan)
    Z = lin(XY).reshape(XI.shape)

    # 7) Mask strictly outside triangulation (no extrapolation)
    inside = tri.find_simplex(XY) >= 0
    Z_masked = Z.copy()
    Z_masked[~inside.reshape(Z.shape)] = np.nan

    # 8) Outer envelope (convex hull for speed and stability)
    try:
        hull = ConvexHull(pts)
        hull_xy = pts[hull.vertices]
    except Exception:
        hull_xy = np.column_stack([x, y])  # fallback (will draw nothing special)

    # Optional downsample for overlay
    if len(x) > 3000:
        idx = np.linspace(0, len(x) - 1, 3000).astype(int)
        sx, sy = x[idx], y[idx]
    else:
        sx, sy = x, y

    return xi, yi, Z_masked, hull_xy, include, sx, sy


# -------------------------- Figure building --------------------------

def _empty_figure(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title="Harbour lift — Cdyn = 1.15 (iso-capacity hulls)",
        template="plotly_white",
        xaxis_title="Outreach [m]",
        yaxis_title="Height [m]",
        height=760, margin=dict(l=40, r=20, t=60, b=40),
        annotations=[dict(text=msg, x=0.5, y=0.5, xref='paper', yref='paper',
                          showarrow=False, font=dict(size=14, color="crimson"))],
    )
    return fig


def _build_figure(xi, yi, Z_masked, hull_xy, include, show_samples, sx, sy):
    band_step = ISO_LEVELS[1] - ISO_LEVELS[0] if len(ISO_LEVELS) > 1 else 35
    fig = go.Figure()

    # Filled contour bands (no values outside hull)
    fig.add_trace(go.Contour(
        x=xi, y=yi, z=Z_masked,
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
        hovertemplate=("Outreach: %{x:.2f} m<br>"
                       "Height: %{y:.2f} m<br>"
                       f"{VALUE_LABEL}: %{{z:.1f}}<extra></extra>"),
        showscale=True,
        zauto=False, zmin=min(ISO_LEVELS), zmax=max(ISO_LEVELS),
    ))

    # Emphasize 70 t and 105 t isolines (manual-style)
    for level in (70, 105):
        fig.add_trace(go.Contour(
            x=xi, y=yi, z=Z_masked,
            contours=dict(coloring="none", showlines=True, start=level, end=level, size=1e-6),
            line=dict(color="#ffd000", width=2.5),
            showscale=False, hoverinfo="skip",
            name=f"{level} t line",
            zauto=False, zmin=min(ISO_LEVELS), zmax=max(ISO_LEVELS),
        ))

    # Black outer envelope
    if hull_xy is not None and hull_xy.size:
        fig.add_trace(go.Scatter(
            x=np.r_[hull_xy[:, 0], hull_xy[0, 0]],
            y=np.r_[hull_xy[:, 1], hull_xy[0, 1]],
            mode="lines",
            line=dict(color="#1b1b1b", width=2.0),
            name="Envelope", hoverinfo="skip",
        ))

    # Optional sample overlay
    if show_samples and sx.size:
        fig.add_trace(go.Scattergl(
            x=sx, y=sy, mode="markers",
            marker=dict(size=3, opacity=0.28),
            name="Samples", hoverinfo="skip",
        ))

    fig.update_layout(
        title="Harbour lift — Cdyn = 1.15 (iso-capacity hulls)",
        xaxis_title="Outreach [m]",
        yaxis_title=("Jib head above deck level [m]" if include
                     else "Jib head above pedestal flange [m]"),
        template="plotly_white",
        height=760, margin=dict(l=40, r=20, t=60, b=40),
        uirevision="tab2b",   # persist zoom/pan on updates
    )
    return fig


# ------------------------------ Layout --------------------------------

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
        dcc.Loading(
            id="tabb-loading",
            type="circle",
            className="loading-progress",
            children=[dcc.Graph(id="harbour-cdyn115-contours")],
        ),
        _legend_card(),
    ]
)


# ------------------------------ Callback ------------------------------

@callback(
    Output("harbour-cdyn115-contours", "figure"),
    Input("app-config", "data"),
    Input("tabb-show-samples", "value"),
)
def update_iso_hulls(config, show_samples_value):
    cfg_key = (
        (config.get("interp_mode") or "linear").lower() if config else "linear",
        int(config.get("main_factor", 1)) if config else 1,
        int(config.get("folding_factor", config.get("fold_factor", 1)) if config else 1),
        bool(config.get("include_pedestal", False)) if config else False,
        float(config.get("pedestal_height", 6.0)) if config else 6.0,
        RASTER_N,
    )

    result = _compute_arrays(cfg_key)
    if result is None:
        return _empty_figure(
            "No plottable capacity grid. Please check Harbour_Cdyn115.csv "
            "(delimiter/decimals) or Page 1 settings."
        )

    xi, yi, Z_masked, hull_xy, include, sx, sy = result
    show_samples = "on" in (show_samples_value or [])
    return _build_figure(xi, yi, Z_masked, hull_xy, include, show_samples, sx, sy)
