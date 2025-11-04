from dash import html, dcc, callback, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
import pandas as pd

from lib.data_utils import (
    get_position_grids,       # X/Y grids + angle arrays (aligned with Page 1)
    load_value_grid,          # loads Harbour_Cdyn115.csv (angle grid)
    interpolate_value_grid,   # -> ndarray on current angle grid
    flatten_with_values,      # X/Y + values -> tidy df (no angles)
)

VALUE_FILE  = "Harbour_Cdyn115.csv"
VALUE_LABEL = "Capacity [t]"


def _make_figure(df: pd.DataFrame, include_pedestal: bool) -> go.Figure:
    # customdata holds angles for the hover
    custom = np.stack(
        [df["main_deg"].to_numpy(), df["folding_deg"].to_numpy()],
        axis=-1,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Outreach [m]"],
        y=df["Height [m]"],
        mode="markers",
        marker=dict(
            size=6,
            color=df[VALUE_LABEL],
            showscale=True,
            colorbar=dict(title=VALUE_LABEL),
        ),
        name="Harbour Cdyn 1.15",
        customdata=custom,
        hovertemplate=(
            "Outreach: %{x:.2f} m — Height: %{y:.2f} m<br>"
            "Main Jib: %{customdata[0]:.2f}° — Folding Jib: %{customdata[1]:.2f}°<br>"
            f"{VALUE_LABEL}: %{{marker.color:.2f}}"
        ),
    ))

    fig.update_layout(
        title="Harbour lift — Cdyn = 1.15 (capacity map)",
        xaxis_title="Outreach [m]",
        yaxis_title=("Jib head above deck level [m]" if include_pedestal
                     else "Jib head above pedestal flange [m]"),
        template="plotly_white",
        height=760,
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


def _format_df_for_table(df: pd.DataFrame, include_pedestal: bool) -> pd.DataFrame:
    dfv = df[["Outreach [m]", "Height [m]", "main_deg", "folding_deg", VALUE_LABEL]].copy()
    dfv["Outreach [m]"] = dfv["Outreach [m]"].round(2)
    dfv["Height [m]"]   = dfv["Height [m]"].round(2)
    dfv["main_deg"]     = dfv["main_deg"].round(2)
    dfv["folding_deg"]  = dfv["folding_deg"].round(2)

    dfv = dfv.rename(columns={
        "main_deg": "Main jib [°]",
        "folding_deg": "Folding jib [°]",
    })
    if include_pedestal:
        dfv = dfv.rename(columns={"Height [m]": "Height [m] (deck level)"})
    else:
        dfv = dfv.rename(columns={"Height [m]": "Height [m] (above pedestal flange)"})
    return dfv


layout = html.Div(
    [
        html.H5("Page 2 – Sub A: Harbour lift capacity (Cdyn 1.15)"),
        html.Div(className="mb-2 small text-muted",
                 children="Aligned with Page 1 settings (linear/spline, subdivisions, pedestal)."),

        dcc.Graph(id="harbour-cdyn115-graph"),

        html.Div(
            [
                html.H6("Harbour Cdyn 1.15 — data"),
                dash_table.DataTable(
                    id="harbour-cdyn115-table",
                    columns=[], data=[],
                    sort_action="native", filter_action="native",
                    page_size=20,
                    style_table={"height": "420px", "overflowY": "auto"},
                    style_cell={"padding": "6px", "fontSize": "14px"},
                    style_header={"fontWeight": "600"},
                ),
                dbc.Button("Download CSV", id="harbour-cdyn115-download-btn",
                           n_clicks=0, color="primary", className="mt-3"),
            ],
            className="mt-3",
        ),
        dcc.Download(id="harbour-cdyn115-download"),
    ]
)


@callback(
    Output("harbour-cdyn115-graph", "figure"),
    Output("harbour-cdyn115-table", "columns"),
    Output("harbour-cdyn115-table", "data"),
    Input("app-config", "data"),
)
def update_harbour_view(config):
    include = bool(config.get("include_pedestal", False)) if config else False
    mode = (config.get("interp_mode") or "linear").lower() if config else "linear"
    if mode not in {"linear", "spline"}:
        mode = "linear"

    # 1) Build XY grids that match Page 1
    Xgrid, Ygrid, new_main, new_fold = get_position_grids(config=config, data_dir="data")

    # 2) Interpolate value grid to the same angles
    V_orig = load_value_grid(VALUE_FILE, data_dir="data")     # DataFrame on original angles
    Vgrid  = interpolate_value_grid(V_orig, new_main, new_fold, mode=mode)  # ndarray on new grid

    # 3) Make tidy df (X/Y/value) and add angles for hover/table
    df = flatten_with_values(Xgrid, Ygrid, Vgrid, value_name=VALUE_LABEL)
    F, M = np.meshgrid(new_fold, new_main, indexing="ij")
    df["main_deg"]    = M.ravel()
    df["folding_deg"] = F.ravel()

    # 4) Plot and table
    fig = _make_figure(df, include_pedestal=include)
    df_view = _format_df_for_table(df, include)
    columns = [{"name": c, "id": c} for c in df_view.columns]
    data = df_view.to_dict("records")
    return fig, columns, data


@callback(
    Output("harbour-cdyn115-download", "data"),
    Input("harbour-cdyn115-download-btn", "n_clicks"),
    State("app-config", "data"),
    prevent_initial_call=True,
)
def download_harbour_csv(n_clicks, config):
    if not n_clicks:
        return None

    include = bool(config.get("include_pedestal", False)) if config else False
    mode = (config.get("interp_mode") or "linear").lower() if config else "linear"
    if mode not in {"linear", "spline"}:
        mode = "linear"

    Xgrid, Ygrid, new_main, new_fold = get_position_grids(config=config, data_dir="data")
    V_orig = load_value_grid(VALUE_FILE, data_dir="data")
    Vgrid  = interpolate_value_grid(V_orig, new_main, new_fold, mode=mode)

    df = flatten_with_values(Xgrid, Ygrid, Vgrid, value_name=VALUE_LABEL)
    F, M = np.meshgrid(new_fold, new_main, indexing="ij")
    df["main_deg"]    = M.ravel()
    df["folding_deg"] = F.ravel()

    df_view = _format_df_for_table(df, include)
    fname = f"Harbour_Cdyn115_m{config.get('main_factor',1)}_f{config.get('folding_factor',config.get('fold_factor',1))}_{mode}" \
            f"{'_with_pedestal' if include else '_without_pedestal'}.csv"
    return dcc.send_data_frame(df_view.to_csv, filename=fname, index=False)

