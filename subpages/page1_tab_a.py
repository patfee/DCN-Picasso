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
    {"label": "2× per-interval",            "value": 2},
    {"label": "4× per-interval",            "value": 4},
    {"label": "8× per-interval",            "value": 8},
    {"label": "16× per-interval",           "value": 16},
]
FOLD_OPTIONS = MAIN_OPTIONS + [{"label": "32× per-interval", "value": 32}]

MODE_OPTIONS = [
    {"label": "Linear (bilinear)",         "value": "linear"},
    {"label": "Spline (smoother)",         "value": "spline"},
    {"label": "Kinematic (2-link model)",  "value": "kinematic"},
]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _format_df_for_view(df: pd.DataFrame, include_pedestal: bool) -> pd.DataFrame:
    """
    Prepare a user-facing table: order/round columns and label Height based on pedestal toggle.
    """
    dfv = df[["Outreach [m]", "Height [m]", "main_deg", "folding_deg"]].copy()
    dfv["Outreach [m]"] = dfv["Outreach [m]"].round(2)
    dfv["Height [m]"]   = dfv["Height [m]"].round(2)
    dfv["main_deg"]     = dfv["main_deg"].round(2)
    dfv["folding_deg"]  = dfv["folding_deg"].round(2)

    dfv = dfv.rename(columns={
        "main_deg": "Main jib [°]",
        "folding_deg": "Folding jib [°]",
    })

    # Relabel height column header depending on pedestal inclusion
    if include_pedestal:
        dfv = dfv.rename(columns={"Height [m]": "Height [m] (deck level)"})
    else:
        dfv = dfv.rename(columns={"Height [m]": "Height [m] (above pedestal flange)"})

    return dfv


def make_figure(df: pd.DataFrame, include_pedestal: bool) -> go.Figure:
    """
    Build scatter plot with rich hover:
      Line 1: Outreach & Height (m)
      Line 2: Main jib (°) & Folding jib (°)
    """
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
        y_min, y_max = df["Height [m]"].min(),   df["Height [m]"].max()
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

        # Controls (row 1)
        dbc.Row(
            [
                dbc.Col(
                    dbc.Switch(
                        id="toggle-pedestal",
                        label="Add pedestal height",
                        value=False,  # will sync from store at load
                    ),
                    md=3,
                ),
                dbc.Col(
                    dbc.Input(
                        id="pedestal-height",
                        type="number",
                        value=6.0,   # will sync from store
                        step=0.1,
                        min=0
                    ),
                    md=2,
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="main-factor",
                        options=MAIN_OPTIONS,
                        value=1,
                        clearable=False,
                    ),
                    md=3,
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="folding-factor",
                        options=FOLD_OPTIONS,
                        value=1,
                        clearable=False,
                    ),
                    md=3,
                ),
            ],
            className="g-3 mb-2",
        ),

        # Labels row (under the two dropdowns)
        dbc.Row(
            [
                dbc.Col(html.Div(""), md=3),
                dbc.Col(html.Div(""), md=2),
                dbc.Col(html.Div("Main-jib subdivision"), md=3),
                dbc.Col(html.Div("Folding-jib subdivision"), md=3),
            ],
            className="mb-3 small text-muted",
        ),

        # Controls (row 2) — interpolation mode
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        id="interp-mode",
                        options=MODE_OPTIONS,
                        value="linear",
                        clearable=False,
                    ),
                    md=6,
                ),
                dbc.Col(
                    html.Div("Interpolation mode", className="small text-muted"),
                    md=3,
                ),
            ],
            className="g-3 mb-2",
        ),

        # Graph
        dcc.Graph(id="crane-graph"),

        # Table + download
        html.Div(
            [
                html.H6("Height / Outreach data"),
                dash_table.DataTable(
                    id="crane-table",
                    columns=[],  # set dynamically
                    data=[],     # set dynamically
                    sort_action="native",
                    filter_action="native",
                    page_size=20,
                    style_table={"height": "420px", "overflowY": "auto"},
                    style_cell={"padding": "6px", "fontSize": "14px"},
                    style_header={"fontWeight": "600"},
                ),
                dbc.Button(
                    "Download CSV",
                    id="download-csv-btn",
                    n_clicks=0,
                    color="primary",
                    className="mt-3",
                ),
            ],
            className="mt-3",
        ),

        dcc.Download(id="download-csv"),
    ]
)


# ----------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------

# Sync controls from session store on first load
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
    include  = bool(config.get("include_pedestal", False)) if config else False
    pedestal = float(config.get("pedestal_height", 6.0))   if config else 6.0
    main_f   = int(config.get("main_factor", 1))           if config else 1
    fold_f   = int(config.get("folding_factor", 1))        if config else 1
    mode     = (config.get("interp_mode") or "linear")     if config else "linear"
    return include, pedestal, main_f, fold_f, mode


# Write changes back to the session store
@callback(
    Output("app-config", "data"),
    Input("toggle-pedestal", "value"),
    Input("pedestal-height", "value"),
    Input("main-factor", "value"),
    Input("folding-factor", "value"),
    Input("interp-mode", "value"),
    State("app-config", "data"),
)
def write_store(include_pedestal, pedestal_height, main_factor, folding_factor, interp_mode, current):
    current = current or {}

    # keep pedestal input sane
    try:
        ph = float(pedestal_height) if pedestal_height is not None else current.get("pedestal_height", 6.0)
        if not np.isfinite(ph):
            ph = current.get("pedestal_height", 6.0)
    except Exception:
        ph = current.get("pedestal_height", 6.0)

    current.update({
        "include_pedestal": bool(include_pedestal),
        "pedestal_height": ph,
        "main_factor": int(main_factor) if main_factor else 1,
        "folding_factor": int(folding_factor) if folding_factor else 1,
        "interp_mode": (interp_mode or "linear"),
    })
    return current


# Build the figure + table from the effective dataset (respects all settings)
@callback(
    Output("crane-graph", "figure"),
    Output("crane-table", "columns"),
    Output("crane-table", "data"),
    Input("app-config", "data"),
)
def update_outputs_from_store(config):
    df = get_crane_points(config=config, data_dir="data")
    include = bool(config.get("include_pedestal", False)) if config else False

    fig = make_figure(df, include)

    df_view = _format_df_for_view(df, include)
    columns = [{"name": c, "id": c} for c in df_view.columns]
    data = df_view.to_dict("records")

    return fig, columns, data


# Download the currently effective dataset (interpolated + pedestal-configured)
@callback(
    Output("download-csv", "data"),
    Input("download-csv-btn", "n_clicks"),
    State("app-config", "data"),
    prevent_initial_call=True,
)
def download_csv(n_clicks, config):
    if not n_clicks:
        return None
    df = get_crane_points(config=config, data_dir="data")
    include = bool(config.get("include_pedestal", False)) if config else False
    df_out = _format_df_for_view(df, include)

    fname = (
        f"crane_points_m{config.get('main_factor',1)}"
        f"_f{config.get('folding_factor',1)}"
        f"_{config.get('interp_mode','linear')}"
        f"{'_with_pedestal' if include else '_without_pedestal'}.csv"
    )
    return dcc.send_data_frame(df_out.to_csv, filename=fname, index=False)
