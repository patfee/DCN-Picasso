from dash import html, dcc, callback, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from lib.data_utils import get_crane_points


# ----------------------------------------------------------------------
# UI option lists
# ----------------------------------------------------------------------
MAIN_OPTIONS = [
    {"label": "1× per-interval (original)", "value": 1},
    {"label": "2× per-interval", "value": 2},
    {"label": "4× per-interval", "value": 4},
    {"label": "8× per-interval", "value": 8},
    {"label": "16× per-interval", "value": 16},
]
FOLD_OPTIONS = MAIN_OPTIONS + [{"label": "32× per-interval", "value": 32}]

MODE_OPTIONS = [
    {"label": "Linear (bilinear)", "value": "linear"},
    {"label": "Spline (smoother)", "value": "spline"},
    {"label": "Kinematic (2-link model)", "value": "kinematic"},
]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _format_df_for_view(df: pd.DataFrame, include_pedestal: bool) -> pd.DataFrame:
    dfv = df[["Outreach [m]", "Height [m]", "main_deg", "folding_deg"]].copy()
    dfv["Outreach [m]"] = dfv["Outreach [m]"].round(2)
    dfv["Height [m]"] = dfv["Height [m]"].round(2)
    dfv["main_deg"] = dfv["main_deg"].round(2)
    dfv["folding_deg"] = dfv["folding_deg"].round(2)
    dfv = dfv.rename(columns={
        "main_deg": "Main jib [°]",
        "folding_deg": "Folding jib [°]",
    })
    if include_pedestal:
        dfv = dfv.rename(columns={"Height [m]": "Height [m] (deck level)"})
    else:
        dfv = dfv.rename(columns={"Height [m]": "Height [m] (above pedestal flange)"})
    return dfv


def make_figure(df: pd.DataFrame, include_pedestal: bool) -> go.Figure:
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
            "Main Jib: %{customdata[0]:.2f}° — Folding Jib: %{customdata[1]:.2f}°"
        )
    ))
    fig.update_layout(
        title="Tabular data points Main Hoist",
        xaxis_title="Outreach [m]",
        yaxis_title=("Jib head above deck level [m]" if include_pedestal
                     else "Jib head above pedestal flange [m]"),
        template="plotly_white",
        height=720,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    if not df.empty:
        x_min, x_max = df["Outreach [m]"].min(), df["Outreach [m]"].max()
        y_min, y_max = df["Height [m]"].min(), df["Height [m]"].max()
        x_pad = max(0.5, 0.03 * (x_max - x_min))
        y_pad = max(0.5, 0.03 * (y_max - y_min))
        fig.update_xaxes(range=[x_min - x_pad, x_max + x_pad], zeroline=True)
        fig.update_yaxes(range=[y_min - y_pad, y_max + y_pad], zeroline=True)
    return fig


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
layout = html.Div(
    [
        html.H5("Page 1 – Sub A: Height vs Outreach"),
        dbc.Row(
            [
                dbc.Col(dbc.Switch(id="toggle-pedestal", label="Add pedestal height", value=False), md=3),
                dbc.Col(dbc.Input(id="pedestal-height", type="number", value=6.0, step=0.1, min=0), md=2),
                dbc.Col(dcc.Dropdown(id="main-factor", options=MAIN_OPTIONS, value=1, clearable=False), md=3),
                dbc.Col(dcc.Dropdown(id="folding-factor", options=FOLD_OPTIONS, value=1, clearable=False), md=3),
            ],
            className="g-3 mb-2",
        ),
        dbc.Row(
            [
                dbc.Col(html.Div(""), md=3),
                dbc.Col(html.Div(""), md=2),
                dbc.Col(html.Div("Main-jib subdivision"), md=3),
                dbc.Col(html.Div("Folding-jib subdivision"), md=3),
            ],
            className="mb-3 small text-muted",
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Dropdown(id="interp-mode", options=MODE_OPTIONS, value="linear", clearable=False), md=6),
                dbc.Col(html.Div("Interpolation mode", className="small text-muted"), md=3),
            ],
            className="g-3 mb-2",
        ),
        dcc.Graph(id="crane-graph"),
        html.Div(
            [
                html.H6("Height / Outreach data"),
                dash_table.DataTable(
                    id="crane-table",
                    columns=[], data=[],
                    sort_action="native", filter_action="native",
                    page_size=20,
                    style_table={"height": "420px", "overflowY": "auto"},
                    style_cell={"padding": "6px", "fontSize": "14px"},
                    style_header={"fontWeight": "600"},
                ),
                dbc.Button("Download CSV", id="download-csv-btn", n_clicks=0, color="primary", className="mt-3"),
            ],
            className="mt-3",
        ),
        dcc.Download(id="download-csv"),
    ]
)


# ----------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------
@callback(
    Output("toggle-pedestal", "value"),
    Output("pedestal-height", "value"),
    Output("main-factor", "value"),
    Output("folding-factor", "value"),
    Output("interp-mode", "value"),
    Input("app-config", "data"),
    prevent_initial_call=False,
)
def sync_controls_from_store(config):
   
