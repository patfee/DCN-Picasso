from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
from lib.data_utils import get_crane_points

def make_figure(df, include_pedestal: bool):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Outreach [m]"],
        y=df["Height [m]"],
        mode="markers",
        marker=dict(size=6),
        name="Data points"
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
        dbc.Row(
            [
                dbc.Col(
                    dbc.Switch(
                        id="toggle-pedestal",
                        label="Add pedestal height",
                        value=False,  # default off; will be synced from store on first draw
                    ),
                    md=3,
                ),
                dbc.Col(
                    dbc.Input(
                        id="pedestal-height",
                        type="number",
                        value=6.0,   # default; will be synced from store
                        step=0.1,
                        min=0
                    ),
                    md=2,
                ),
            ],
            className="g-3 mb-3",
        ),
        # The shared store lives in app.py; we just link to it here
        dcc.Store(id="app-config-proxy", storage_type="session"),
        dcc.Graph(id="crane-graph"),
    ]
)

# --- Sync the UI controls with the session store on first load
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


# --- Update the shared store when user changes controls
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


# --- Build the figure from the *effective* dataset (uses store config)
@callback(
    Output("crane-graph", "figure"),
    Input("app-config", "data"),
)
def update_graph_from_store(config):
    df = get_crane_points(config=config, data_dir="data")
    include = bool(config.get("include_pedestal", False)) if config else False
    return make_figure(df, include)
