import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, Input, Output, callback, ctx

from lib.data_utils import (
    get_position_grids,
    load_value_grid,
    flatten_with_values,
)

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
DATA_DIR = "data"
VALUE_FILE = "Harbour_Cdyn115.csv"
VALUE_LABEL = "Capacity [t]"

# -------------------------------------------------------------------
# Build contour figure
# -------------------------------------------------------------------

def _build_contour_figure(df, include_pedestal=False, show_samples=False):
    """
    Build an iso-capacity contour (or hull) plot from a Harbour Cdyn grid.
    Displays only regions where valid data exist.
    """
    # Mask to keep only valid (finite) values
    mask = np.isfinite(df[VALUE_LABEL].to_numpy())
    df_valid = df.loc[mask]

    x = df_valid["Outreach [m]"]
    y = df_valid["Height [m]"]
    z = df_valid[VALUE_LABEL]

    fig = go.Figure()

    # Contour bands (only where data exist)
    fig.add_trace(go.Contour(
        x=x,
        y=y,
        z=z,
        colorscale="Turbo",
        contours=dict(
            showlines=True,
            start=np.nanmin(z),
            end=np.nanmax(z),
            size=5,
            coloring="lines",
        ),
        hovertemplate=(
            f"Outreach: %{x:.2f} m<br>"
            f"Height: %{y:.2f} m<br>"
            f"{VALUE_LABEL}: %{z:.1f}<extra></extra>"
        ),
    ))

    # Optionally overlay scatter of sample points
    if show_samples:
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(size=4, color=z, colorscale="Turbo",
                        showscale=False, line=dict(width=0.2, color="black")),
            name="Data points",
            hovertemplate=(
                f"Outreach: %{x:.2f} m<br>"
                f"Height: %{y:.2f} m<br>"
                f"{VALUE_LABEL}: %{marker.color:.1f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Harbour lift — Cdyn 1.15 (iso-capacity hulls)",
        xaxis_title="Outreach [m]",
        yaxis_title=("Jib head above deck level [m]"
                     if include_pedestal
                     else "Jib head above pedestal flange [m]"),
        template="plotly_white",
        height=760,
        margin=dict(l=40, r=20, t=60, b=40),
        uirevision="tab2b",  # persist zoom/pan on update
    )

    return fig


# -------------------------------------------------------------------
# Page layout
# -------------------------------------------------------------------

layout = html.Div([
    html.H4("Page 2 – Tab B  |  Iso-Capacity Hulls (Harbour Cdyn 1.15)"),

    html.Div([
        dcc.Checklist(
            id="include-pedestal-2b",
            options=[{"label": "Include pedestal height", "value": "yes"}],
            value=["yes"],
            inline=True,
        ),
        dcc.Checklist(
            id="show-samples-2b",
            options=[{"label": "Show sample points", "value": "yes"}],
            value=[],
            inline=True,
        ),
    ], style={"marginBottom": "0.5rem"}),

    html.Div(
        children=[
            dcc.Graph(id="harbour-cdyn115-contours",
                      style={"cursor": "progress"})  # shows spinner on compute
        ],
        style={"height": "780px"},
    ),
])


# -------------------------------------------------------------------
# Callback
# -------------------------------------------------------------------

@callback(
    Output("harbour-cdyn115-contours", "figure"),
    Input("include-pedestal-2b", "value"),
    Input("show-samples-2b", "value"),
    prevent_initial_call=False,
)
def update_iso_hulls(include_vals, show_vals):
    include_pedestal = "yes" in include_vals
    show_samples = "yes" in show_vals

    # compute base geometry grids (same interpolation as Tab A)
    outreach_grid, height_grid, main_deg, fold_deg = get_position_grids(
        config={"interp_mode": "linear", "main_factor": 1, "folding_factor": 1,
                "include_pedestal": include_pedestal},
        data_dir=DATA_DIR
    )

    # load the Harbour capacity grid (same angular layout)
    cap_df = load_value_grid(VALUE_FILE, data_dir=DATA_DIR)

    # align sizes
    cap_interp = np.asarray(cap_df)
    cap_interp = np.resize(cap_interp, outreach_grid.shape)

    df = flatten_with_values(outreach_grid, height_grid,
                             cap_interp, value_name=VALUE_LABEL)

    return _build_contour_figure(df, include_pedestal, show_samples)
