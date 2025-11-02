from dash import html, dcc, Input, Output
import dash
import plotly.graph_objects as go
import pandas as pd

from lib.data_utils import (
    load_matrix_csv_flexible,
    stack_height_outreach,
    list_data_files,
    DATA_DIR,
    data_dir_exists,
)

# Try common filenames (adjust/add if yours differ)
HEIGHT_CANDIDATES = ("height.csv", "Height.csv", "height_matrix.csv")
OUTREACH_CANDIDATES = ("outreach.csv", "Outreach.csv", "outreach_matrix.csv")

# Load
height_df = load_matrix_csv_flexible(HEIGHT_CANDIDATES)
outreach_df = load_matrix_csv_flexible(OUTREACH_CANDIDATES)
data_ok = not (height_df.empty or outreach_df.empty)

if data_ok:
    cloud = stack_height_outreach(height_df, outreach_df)
    main_options = [{"label": f"{int(a)}°", "value": float(a)} for a in sorted(cloud["main_deg"].unique())]
    fold_options = [{"label": f"{int(a)}°", "value": float(a)} for a in sorted(cloud["folding_deg"].unique())]
else:
    cloud = pd.DataFrame(columns=["main_deg", "folding_deg", "height_m", "outreach_m"])
    main_options = []
    fold_options = []

def _diagnostics_div():
    files = list_data_files()
    info = [
        html.Li(f"Data dir: {DATA_DIR} {'(exists)' if data_dir_exists() else '(NOT FOUND!)'}"),
        html.Li(f"Files in data/: {', '.join(files) if files else '(none)'}"),
        html.Li(f"Height candidates: {', '.join(HEIGHT_CANDIDATES)}"),
        html.Li(f"Outreach candidates: {', '.join(OUTREACH_CANDIDATES)}"),
        html.Li(f"Loaded height shape: {height_df.shape}"),
        html.Li(f"Loaded outreach shape: {outreach_df.shape}"),
        html.Li(f"Cloud rows: {len(cloud)}"),
    ]
    return html.Details([
        html.Summary("Diagnostics"),
        html.Ul(info, style={"marginTop": "8px"})
    ], open=not data_ok)

layout = html.Div(
    className="tab-wrap",
    children=[
        html.H4("Page 1 – Sub A: Height vs Outreach"),
        _diagnostics_div(),
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
            ],
        ),
        dcc.Graph(id="p1a-scatter"),
        html.Small(
            "Hover shows: Main Jib, Folding Jib, Outreach (m), Height (m). "
            "Files are loaded from the local 'data/' directory."
        ),
    ],
)

@dash.callback(
    Output("p1a-scatter", "figure"),
    Input("p1a-main-angles", "value"),
    Input("p1a-fold-angles", "value"),
)
def render_scatter(main_filter, fold_filter):
    fig = go.Figure()
    fig.update_layout(
        title="Crane Height vs Outreach",
        xaxis_title="Outreach (m)",
        yaxis_title="Height (m)",
        legend=dict(orientation="h"),
        margin=dict(l=10, r=10, t=60, b=10),
    )

    if cloud.empty:
        fig.update_layout(title="No data found — check Diagnostics above (are CSVs in data/?)")
        return fig

    df = cloud
    if main_filter:
        df = df[df["main_deg"].isin(main_filter)]
    if fold_filter:
        df = df[df["folding_deg"].isin(fold_filter)]

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
            customdata=df[["main_deg", "folding_deg"]],
        )
    )

    return fig
