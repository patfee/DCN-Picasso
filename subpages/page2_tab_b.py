from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from scipy.interpolate import griddata  # rasterize scattered XY->rect grid

from lib.data_utils import (
    get_position_grids,       # XY grids aligned with Page 1 (mode, factors, pedestal)
    load_value_grid,          # loads angle-grid CSV (Harbour_Cdyn115.csv)
    interpolate_value_grid,   # interpolates that value grid to current angle grid
    flatten_with_values,      # flattens XY+values to a tidy table
)

VALUE_FILE  = "Harbour_Cdyn115.csv"
VALUE_LABEL = "Capacity [t]"

# Levels (t) – tweak if you want other banding
ISO_LEVELS = [0, 35, 70, 105, 140]  # edges for filled bands (uniform spacing)

# Colors roughly inspired by your screenshot
COLORSCALE = [
    [0.00, "#003b46"],  # deep teal (low)
    [0.20, "#00b3c6"],  # cyan
    [0.40, "#9ecf2a"],  # green-yellow
    [0.60, "#ffcc33"],  # yellow
    [0.80, "#ff8840"],  # orange
    [1.00, "#cc2f2f"],  # red (high)
]

# Discrete legend swatches for the bands between ISO_LEVELS
# (length should be len(ISO_LEVELS)-1)
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
                html.Div(className="text-muted small mt-2",
                         children="Bands follow your Page 1 interpolation & subdivisions."),
            ]
        ),
        className="mt-3",
        style={"maxWidth": "260px"}
    )


def _build_contour_figure(df: pd.DataFrame, include_pedestal: bool, show_samples: bool) -> go.Figure:
    """Rasterize scattered points to a regular XY grid and draw filled iso-bands."""
    x = df["Outreach [m]"].to_numpy()
    y = df["Height [m]"].to_numpy()
    z = df[VALUE_LABEL].to_numpy()

    # Define rectilinear canvas
    x_pad = 0.03 * (x.max() - x.min() + 1e-9)
    y_pad = 0.03 * (y.max() - y.min() + 1e-9)
    xi = np.linspace(x.min() - x_pad, x.max() + x_pad, 300)
    yi = np.linspace(y.min() - y_pad, y.max() + y_pad, 300)
    XI, YI = np.meshgrid(xi, yi)

    # Interpolate scattered to grid; linear gives NaN outside convex hull (good mask)
    Z_lin = griddata(points=np.column_stack([x, y]), values=z, xi=(XI, YI), method="linear")
    # Fill small holes with nearest (keeps the hull edges crisp)
    Z_near = griddata(points=np.column_stack([x, y]), values=z, xi=(XI, YI), method="nearest")
    Z = np.where(np.isnan(Z_lin), Z_near, Z_lin)

    # Build filled contours (uniform band size due to uniform ISO_LEVELS spacing)
    band_step = ISO_LEVELS[1] - ISO_LEVELS[0] if len(ISO_LEVELS) > 1 else 10
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=xi, y=yi, z=Z,
        contours=dict(
            coloring="fill",
            showlines=True,
            start=min(ISO_LEVELS),
            end=max(ISO_LEVELS),
            size=band_step,
        ),
        colorscale=COLORSCALE,
        colorbar=dict(title=VALUE_LABEL),
        line=dict(width=1),
        hovertemplate=("Outreach: %{x:.2f} m<br>"
                       "Height: %{y:.2f} m<br>"
                       f"{VALUE_LABEL}: %{z:.1f}<extra></extra>"),
        showscale=True,
        name="Iso-bands",
    ))

    if show_samples:
        fig.add_trace(go.Scattergl(
            x=x, y=y, mode="markers",
            marker=dict(size=3, opacity=0.3),
            name="Samples",
            hoverinfo="skip",
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
