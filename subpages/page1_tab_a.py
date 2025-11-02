from dash import html, dcc, Input, Output
import dash
import plotly.graph_objects as go
import pandas as pd

from lib.data_utils import load_matrix_csv, stack_height_outreach

# Load matrices once (Dash hot-reload will reimport on file save)
HEIGHT_FN = "height.csv"
OUTREACH_FN = "outreach.csv"

# Attempt to load; if missing, show a friendly message in the layout
try:
    height_df = load_matrix_csv(HEIGHT_FN)
    outreach_df = load_matrix_csv(OUTREACH_FN)
    data_ok = not (height_df.empty or outreach_df.empty)
except Exception:
    height_df = pd.DataFrame()
    outreach_df = pd.DataFrame()
    data_ok = False

if data_ok:
    cloud = stack_height_outreach(height_df, outreach_df)
    # Build dropdown options from available angles
    main_options = [{"label": f"{int(a)}°", "value": float(a)} for a in sorted(cloud["main_deg"].unique())]
    fold_options = [{"label": f"{int(a)}°", "value": float(a)} for a in sorted(cloud["folding_deg"].unique())]
else:
    cloud = pd.DataFrame(columns=["main_deg","folding_deg","height_m","outreach_m"])
    main_options = []
    fold_options = []

# Controls + Graph
layout = html.Div(
    className="tab-wrap",
    children=[
        html.H4("Page 1 – Sub A: Height vs Outreach"),
        html.Div(
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
            children=[
                dcc.Dropdown(
                    id="p1a-main-angles",
                    options=main_options,
                    multi=True,
                    placeholder="Filter by Main-jib angle(s)",
                    style={"minWidth": "260px"},
                ),
                dcc.Dropdown(
                    id="p1a-fold-angles",
                    options=fold_options,
                    multi=True,
                    placeholder="Filter by Folding-jib angle(s)",
                    style={"minWidth": "260px"},
                ),
                dcc.Checklist(
                    id="p1a-show-envelope",
                    options=[{"label": "Show outer envelope", "value": "env"}],
                    value=["env"],
                    style={"alignSelf": "center"},
                ),
            ],
        ),
        dcc.Graph(id="p1a-scatter"),
        html.Small(
            "Hover shows: Main Jib, Folding Jib, Outreach (m), Height (m). "
            "Data source: data/height.csv & data/outreach.csv."
        ),
    ],
)

@dash.callback(
    Output("p1a-scatter", "figure"),
    Input("p1a-main-angles", "value"),
    Input("p1a-fold-angles", "value"),
    Input("p1a-show-envelope", "value"),
)
def render_scatter(main_filter, fold_filter, show_env):
    if cloud.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No data found (ensure data/height.csv and data/outreach.csv exist)",
            xaxis_title="Outreach (m)",
            yaxis_title="Height (m)",
        )
        return fig

    df = cloud
    if main_filter and len(main_filter) > 0:
        df = df[df["main_deg"].isin(main_filter)]
    if fold_filter and len(fold_filter) > 0:
        df = df[df["folding_deg"].isin(fold_filter)]

    # Scatter of all grid points
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=df["outreach_m"],
            y=df["height_m"],
            mode="markers",
            name="Grid points",
            hovertemplate=(
                "Main Jib: %{customdata[0]}°<br>"
                "Folding Jib: %{customdata[1]}°<br>"
                "Outreach: %{x:.2f} m<br>"
                "Height: %{y:.2f} m"
            ),
            customdata=df[["main_deg","folding_deg"]],
        )
    )

    # Optional: draw outer envelope (convex hull on X/Y)
    if show_env and "env" in show_env and len(df) > 2:
        try:
            import numpy as np
            from scipy.spatial import ConvexHull

            pts = np.c_[df["outreach_m"].values, df["height_m"].values]
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            # Sort hull by X then Y to look tidy (not strictly necessary)
            hull_pts = hull_pts[np.lexsort((hull_pts[:,1], hull_pts[:,0]))]
            fig.add_trace(
                go.Scatter(
                    x=hull_pts[:,0],
                    y=hull_pts[:,1],
                    mode="lines+markers",
                    name="Envelope",
                )
            )
        except Exception:
            # scipy not installed or hull failed; skip envelope
            pass

    fig.update_layout(
        title="Crane Height vs Outreach",
        xaxis_title="Outreach (m)",
        yaxis_title="Height (m)",
        legend=dict(orientation="h"),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig
