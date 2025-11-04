from dash import html, dcc, callback, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from lib.data_utils import get_crane_points


def _format_df_for_view(df: pd.DataFrame, include_pedestal: bool) -> pd.DataFrame:
    """
    Return a copy of df with nice column order and rounding for display.
    """
    dfv = df.copy()
    # Order: Outreach, Height, Main, Folding
    dfv = dfv[["Outreach [m]", "Height [m]", "main_deg", "folding_deg"]]
    # Round values for display
    dfv["Outreach [m]"] = dfv["Outreach [m]"].round(2)
    dfv["Height [m]"] = dfv["Height [m]"].round(2)
    dfv["main_deg"] = dfv["main_deg"].round(0).astype(int)
    dfv["folding_deg"] = dfv["folding_deg"].round(0).astype(int)

    # Optionally relabel Height column header for the table (keeps data key the same)
    if include_pedestal:
        dfv = dfv.rename(columns={"Height [m]": "Height [m] (deck level)"})
    else:
        dfv = dfv.rename(columns={"Height [m]": "Height [m] (above pedestal flange)"})
    return dfv


def make_figure(df: pd.DataFrame, include_pedestal: bool) -> go.Figure:
    """
    Build the scatter with rich hover text using angles from the dataset.
    """
    # customdata will carry [main_deg, folding_deg] to use in the hovertemplate
    custom = np.stack([df["main_deg"].to_numpy(), df["folding_deg"].to_numpy()], axis=-1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Outreach [m]"],
        y=df["Height [m]"],
        mode="markers",
        marker=dict(size=6),
        name="Data points",
        customdata=custom,
        hovertemplate=(
            "Outreach: %{x:.2f} m — Height: %{y:.2f} m<br>"
            "Main Jib: %{customdata[0]:.0f}° — Folding Jib: %{customdata[1]:.0f}°"
        )
    ))

    fig.update_layout(
        title="Tabular data points Main Hoist",
        xaxis_title="Outreach [m]",
        yaxis_title=("Jib head above deck level [m]" if include_pedestal
                     else "Jib head above pedestal flange [m]"),
        template="plotly_white",
        height=720,
    )

    if not df.empty:
        x_min, x_max = df["Outreach [m]"].min(), df["Outreach [m]"].max()
        y_min, y_max = df["Height [m]"].min(),   df["Height [m]"].max()
        x_pad = max(0.5, 0.03 * (x_max - x_min))
        y_pad = max(0.5, 0.03 * (y_max - y_min))
        fig.update_xaxes(range=[x_min - x_pad, x_max + x_pad], zeroline=True)
        fig.update_yaxes(range=[y_min - y_pad, y_max + y_pad], zeroline=True)

    return fig


layout = html.Div(
    [
        html.H5("Page 1 – Sub A: Height vs Outreach"),

        # Controls
        dbc.Row(
            [
                dbc.Col(
                    dbc.Switch(
                        id="toggle-pedestal",
                        label="Add pedestal height",
                        value=False,  # default off; will sync from store
                    ),
                    md=3,
                ),
                dbc.Col(
                    dbc.Input(
                        id="pedestal-height",
                        type="number",
                        value=6.0,   # default; will sync from store
                        step=0.1,
                        min=0
                    ),
                    md=2,
                ),
                dbc.Col(
                    dbc.Button("Download CSV", id="download-csv-btn", n_clicks=0, color="primary"),
                    md=2,
                ),
            ],
            className="g-3 mb-3",
        ),

        # Graph
        dcc.Graph(id="crane-graph"),

        # Table
        html.Div(
            [
                html.H6("Height/Outreach data"),
                dash_table.DataTable(
                    id="crane-table",
                    columns=[],  # filled dynamically
                    data=[],     # filled dynamically
                    sort_action="native",
                    filter_action="native",
                    page_size=15,
                    style_table={"height": "420px", "overflowY": "auto"},
                    style_cell={"padding": "6px", "fontSize": "14px"},
                    style_header={"fontWeight": "600"},
                ),
            ],
            className="mt-3",
        ),

        # Shared store lives in app.py
        dcc.Store(id="app-config-proxy", storage_type="session"),

        # Download component
        dcc.Download(id="download-csv"),
    ]
)

# -------- Sync controls from session store on first load
@callback(
    Output("toggle-pedestal", "value"),
    Output("pedestal-height", "value"),
    Input("app-config", "data"),
    prevent_initial_call=False
)
def sync_controls_from_store(config):
    include = bool(config.get("include_pedestal", False)) if config else False
    pedestal = float(config.get("pedestal_height", 6.0)) if config else 6.0
    return include, pedestal


# -------- Update the shared store when user changes controls
@callback(
    Output("app-config", "data"),
    Input("toggle-pedestal", "value"),
    Input("pedestal-height", "value"),
    State("app-config", "data"),
)
def write_store(include_pedestal, pedestal_height, current):
    current = current or {}
    try:
        ph = float(pedestal_height) if pedestal_height is not None else current.get("pedestal_height", 6.0)
        if not np.isfinite(ph):
            ph = current.get("pedestal_height", 6.0)
    except Exception:
        ph = current.get("pedestal_height", 6.0)

    current.update({
        "include_pedestal": bool(include_pedestal),
        "pedestal_height": ph,
    })
    return current


# -------- Build the figure and the table from the effective dataset
@callback(
    Output("crane-graph", "figure"),
    Output("crane-table", "columns"),
    Output("crane-table", "data"),
    Input("app-config", "data"),
)
def update_outputs_from_store(config):
    df = get_crane_points(config=config, data_dir="data")
    include = bool(config.get("include_pedestal", False)) if config else False

    # Figure
    fig = make_figure(df, include)

    # Table
    df_view = _format_df_for_view(df, include)
    columns = [{"name": c, "id": c} for c in df_view.columns]
    data = df_view.to_dict("records")

    return fig, columns, data


# -------- Download the currently effective dataset
@callback(
    Output("download-csv", "data"),
    Input("download-csv-btn", "n_clicks"),
    State("app-config", "data"),
    prevent_initial_call=True
)
def download_csv(n_clicks, config):
    if not n_clicks:
        return None
    df = get_crane_points(config=config, data_dir="data")
    include = bool(config.get("include_pedestal", False)) if config else False
    df_out = _format_df_for_view(df, include)

    # Use a stable filename reflecting pedestal choice
    fname = "crane_points_with_pedestal.csv" if include else "crane_points_without_pedestal.csv"
    return dcc.send_data_frame(df_out.to_csv, filename=fname, index=False)
