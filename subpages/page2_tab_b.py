from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
from functools import lru_cache
from scipy.spatial import Delaunay, ConvexHull
from scipy.interpolate import LinearNDInterpolator

from lib.data_utils import (
    get_position_grids,   # XY grids aligned with Page 1 config
    load_value_grid,      # Harbour_Cdyn115.csv matrix loader
    interpolate_value_grid,
    flatten_with_values,
)

DATA_DIR    = "data"
VALUE_FILE  = "Harbour_Cdyn115.csv"
VALUE_LABEL = "Capacity [t]"

ISO_LEVELS  = [0, 35, 70, 105, 140]
COLORSCALE  = [
    [0.00, "#003b46"],
    [0.20, "#00b3c6"],
    [0.40, "#9ecf2a"],
    [0.60, "#ffcc33"],
    [0.80, "#ff8840"],
    [1.00, "#cc2f2f"],
]

# Lower = faster; you can bump to 320/384 if you want higher fidelity
RASTER_N = 240

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
                html.Div("Aligned with Page 1 interpolation & subdivisions.",
                         className="text-muted small mt-2"),
            ]
        ),
        className="mt-3",
        style={"maxWidth": "260px"},
    )

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
    mode, main_f, fold_f, include, ped_h, raster_n = cfg_key
    config = dict(
        interp_mode=mode,
        main_factor=main_f,
        folding_factor=fold_f,
        include_pedestal=include,
        pedestal_height=ped_h,
    )

    # 1) XY from Page 1
    Xgrid, Ygrid, new_main, new_fold = get_position_grids(config=config, data_dir=DATA_DIR)

    # 2) Capacity grid aligned to angles
    V_orig = load_value_grid(VALUE_FILE, data_dir=DATA_DIR)
    Vgrid  = interpolate_value_grid(V_orig, new_main, new_fold, mode=mode)

    # 3) Flatten to scattered
    df = flatten_with_values(Xgrid, Ygrid, Vgrid, value_name=VALUE_LABEL)

    m = np.isfinite(df[VALUE_LABEL].to_numpy())
    x = df.loc[m, "Outreach [m]"].to_numpy()
    y = df.loc[m, "Height [m]"].to_numpy()
    z = df.loc[m, VALUE_LABEL].to_numpy()

    pts = np.column_stack([x, y])
    tri = Delaunay(pts)

    # Use hull bbox to limit raster canvas (faster)
    hull = ConvexHull(pts)
    hull_xy = pts[hull.vertices]
    xmin, xmax = hull_xy[:, 0].min(), hull_xy[:, 0].max()
    ymin, ymax = hull_xy[:, 1].min(), hull_xy[:, 1].max()
    x_pad = 0.03 * (xmax - xmin + 1e-9)
    y_pad = 0.03 * (ymax - ymin + 1e-9)

    xi = np.linspace(xmin - x_pad, xmax + x_pad, raster_n)
    yi = np.linspace(ymin - y_pad, ymax + y_pad, raster_n)
    XI, YI = np.meshgrid(xi, yi)
    XY = np.column_stack([XI.ravel(), YI.ravel()])

    lin = LinearNDInterpolator(tri, z, fill_value=np.nan)
    Z = lin(XY).reshape(XI.shape)

    inside = tri.find_simplex(XY) >= 0
    Z_masked = Z.copy()
    Z_masked[~inside.reshape(Z.shape)] = np.nan

    # Optional sample thin
    if len(x) > 3000:
        idx = np.linspace(0, len(x) - 1, 3000).astype(int)
        sx, sy = x[idx], y[idx]
    else:
        sx, sy = x, y

    return xi, yi, Z_masked, hull_xy, include, sx, sy

def _build_figure(xi, yi, Z_masked, hull_xy, include, show_samples, sx, sy):
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
        zauto=False, zmin=min(ISO_LEVELS), zmax=max(ISO_LEVELS),
    ))

    for level in (70, 105):
        fig.add_trace(go.Contour(
            x=xi, y=yi, z=Z_masked,
            contours=dict(coloring="none", showlines=True, start=level, end=level, size=1e-6),
            line=dict(color="#ffd000", width=2.5),
            showscale=False, hoverinfo="skip",
            name=f"{level} t line",
            zauto=False, zmin=min(ISO_LEVELS), zmax=max(ISO_LEVELS),
        ))

    fig.add_trace(go.Scatter(
        x=np.r_[hull_xy[:, 0], hull_xy[0, 0]],
        y=np.r_[hull_xy[:, 1], hull_xy[0, 1]],
        mode="lines",
        line=dict(color="#1b1b1b", width=2.0),
        name="Envelope", hoverinfo="skip",
    ))

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
        uirevision="tab2b",
    )
    return fig

layout = html.Div(
    [
        html.H5("Page 2 – Sub B: Harbour lift (iso-capacity hulls)"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Checklist(
                        id="tabb-show-samples",
                        options=[{"label": "Show sample points overlay", "value": "on"}],
                        value=[], switch=True,
                    ),
                    md="auto",
                ),
            ],
            className="mb-2",
        ),
        dcc.Loading(
            id="tabb-loading",
            type="dot",  # reliable spinner style
            children=[dcc.Graph(id="harbour-cdyn115-contours")],
        ),
        _legend_card(),
    ]
)

@callback(
    Output("harbour-cdyn115-contours", "figure"),
    Input("app-config", "data"),
    Input("tabb-show-samples", "value"),
)
def update_iso_hulls(config, show_samples_value):
    xi, yi, Z_masked, hull_xy, include, sx, sy = _compute_arrays(_cfg_key(config))
    show_samples = "on" in (show_samples_value or [])
    return _build_figure(xi, yi, Z_masked, hull_xy, include, show_samples, sx, sy)
