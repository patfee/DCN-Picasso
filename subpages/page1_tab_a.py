from dash import html, dcc, Input, Output, State
import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lib.data_utils import load_matrix_csv_flexible
from lib.geo_utils import (
    build_interpolators, resample_grid_by_factors, compute_boundary_curve, _sample_points
)

# ------- Settings -------
PEDESTAL_HEIGHT_M = 6.0
HEIGHT_CANDIDATES = ("height.csv", "Height.csv")
OUTREACH_CANDIDATES = ("outreach.csv", "Outreach.csv")
# Optional 2D load matrix (same angle grid). If absent, plot still works:
LOAD_CANDIDATES = ("Harbour_Cdyn115.csv", "harbour_Cdyn115.csv")

# ------- Load matrices -------
height_df = load_matrix_csv_flexible(HEIGHT_CANDIDATES)
outreach_df = load_matrix_csv_flexible(OUTREACH_CANDIDATES)
load_df = load_matrix_csv_flexible(LOAD_CANDIDATES) if not height_df.empty else pd.DataFrame()

if height_df.empty or outreach_df.empty:
    layout = html.Div(
        className="tab-wrap",
        children=[
            html.H4("Page 1 – Sub A: Height vs Outreach"),
            html.P("No matrix CSVs loaded. Ensure files exist in /data "
                   f"(tried {HEIGHT_CANDIDATES} and {OUTREACH_CANDIDATES}).")
        ],
    )
else:
    f_angles, m_angles, height_itp, outre_itp, _ = build_interpolators(height_df, outreach_df, load_df)

    # Originals (for orange points + hover)
    F0, M0 = np.meshgrid(f_angles, m_angles, indexing="ij")
    orig_xy = np.column_stack([outreach_df.values.ravel(), height_df.values.ravel()])
    orig_custom = np.column_stack([F0.ravel(), M0.ravel()])

    controls = html.Div(
        style={"flex": "0 0 340px", "overflowY": "auto", "height": "78vh",
               "paddingRight": "12px", "borderRight": "1px solid #eee"},
        children=[
            html.Label("Main-jib subdivision", style={"fontWeight": 600}),
            dcc.Dropdown(
                id="p1a-main-factor",
                options=[{"label": f"{o}× per-interval" + (" (original)" if o == 1 else ""), "value": o}
                         for o in [1, 2, 4, 8]],
                value=1, clearable=False, style={"marginBottom": "10px"}
            ),
            html.Label("Folding-jib subdivision", style={"fontWeight": 600}),
            dcc.Dropdown(
                id="p1a-fold-factor",
                options=[{"label": f"{o}× per-interval" + (" (original)" if o == 1 else ""), "value": o}
                         for o in [1, 2, 4, 8, 16]],
                value=1, clearable=False, style={"marginBottom": "10px"}
            ),
            html.Div(id="p1a-main-range", style={"fontFamily": "monospace"}),
            html.Div(id="p1a-fold-range", style={"fontFamily": "monospace", "marginBottom": "10px"}),

            dcc.Checklist(
                id="p1a-pedestal-toggle",
                options=[{"label": f"Include pedestal height (+{PEDESTAL_HEIGHT_M:.1f} m)", "value": "on"}],
                value=["on"]
            ),

            html.Label("Envelope type", style={"fontWeight": 600, "marginTop": "8px"}),
            dcc.RadioItems(
                id="p1a-envelope-type",
                options=[
                    {"label": "None (points only)", "value": "none"},
                    {"label": "Concave (fast approx)", "value": "concave"},
                    {"label": "Convex hull (fast, stable)", "value": "convex"},
                ],
                value="convex",
                style={"marginTop": "4px"}
            ),
            html.Div(id="p1a-envelope-help", style={"fontSize": "12px", "color": "#555"}),

            html.Div([
                html.Button("Download interpolated CSV", id="p1a-btn-download", n_clicks=0,
                            style={"marginTop": "10px", "width": "100%"}),
                dcc.Download(id="p1a-download-data"),
            ]),
        ],
    )

    graph = html.Div(
        style={"flex": "1 1 auto", "paddingLeft": "16px"},
        children=[dcc.Graph(id="p1a-graph", style={"height": "78vh"})],
    )

    layout = html.Div(
        className="tab-wrap",
        children=[html.H4("Page 1 – Sub A: Height vs Outreach"),
                  html.Div([controls, graph], style={"display": "flex", "gap": "10px"})]
    )

    @dash.callback(
        Output("p1a-main-range", "children"),
        Output("p1a-fold-range", "children"),
        Input("p1a-main-factor", "value"),
        Input("p1a-fold-factor", "value"),
    )
    def _show_ranges(main_factor, fold_factor):
        return (
            f"Main-jib angle range: {m_angles.min():.2f}° – {m_angles.max():.2f}° "
            f"(original points: {len(np.unique(m_angles))})",
            f"Folding-jib angle range: {f_angles.min():.2f}° – {f_angles.max():.2f}° "
            f"(original points: {len(np.unique(f_angles))})",
        )

    @dash.callback(Output("p1a-envelope-help", "children"), Input("p1a-envelope-type", "value"))
    def _explain(kind):
        if kind == "none":
            return "Displays only points (no envelope). Useful for checking interpolation coverage."
        if kind == "concave":
            return "Concave alpha shape: tighter, realistic shape but sensitive to sparse data."
        return "Convex hull: stable and conservative outer boundary."

    @dash.callback(
        Output("p1a-graph", "figure"),
        Input("p1a-main-factor", "value"),
        Input("p1a-fold-factor", "value"),
        Input("p1a-pedestal-toggle", "value"),
        Input("p1a-envelope-type", "value"),
    )
    def _update_figure(main_factor, fold_factor, pedestal_value, envelope_value):
        include_pedestal = "on" in (pedestal_value or [])
        prefer_concave = (envelope_value == "concave")
        draw_envelope = (envelope_value != "none")

        main_factor = int(main_factor or 1)
        fold_factor = int(fold_factor or 1)

        _, _, _, _, pts = resample_grid_by_factors(f_angles, m_angles, fold_factor, main_factor)
        H_dense = height_itp(pts)
        R_dense = outre_itp(pts)
        if include_pedestal:
            H_dense = H_dense + PEDESTAL_HEIGHT_M

        dense_xy = np.column_stack([R_dense, H_dense])
        dense_xy = dense_xy[~np.isnan(dense_xy).any(axis=1)]

        envelope_xy = None
        if draw_envelope:
            envelope_xy = compute_boundary_curve(_sample_points(dense_xy, 6000),
                                                 prefer_concave=prefer_concave)

        fig = go.Figure()

        # Interpolated points (blue-ish)
        dense_for_plot = _sample_points(dense_xy, 15000)
        if dense_for_plot.size:
            hover_tmpl = (
                "Outreach: %{x:.2f} m<br>"
                "Height: %{y:.2f} m<extra></extra>"
            )
            fig.add_trace(go.Scatter(
                x=dense_for_plot[:, 0], y=dense_for_plot[:, 1],
                mode="markers", marker=dict(size=4, symbol="diamond", opacity=0.5),
                name="Interpolated points",
                hovertemplate=hover_tmpl
            ))

        # Original matrix points (orange)
        if orig_xy.size:
            fig.add_trace(go.Scatter(
                x=orig_xy[:, 0],
                y=orig_xy[:, 1] + (PEDESTAL_HEIGHT_M if include_pedestal else 0.0),
                mode="markers", marker=dict(size=8),
                name="Original matrix points",
                hovertemplate=hover_tmpl
            ))

        # Envelope (green)
        if envelope_xy is not None and len(envelope_xy) > 2:
            fig.add_trace(go.Scatter(
                x=envelope_xy[:, 0], y=envelope_xy[:, 1],
                mode="lines", line=dict(width=3), name="Envelope",
            ))

        fig.update_layout(
            xaxis_title="Outreach [m]",
            yaxis_title="Jib head height [m]" if include_pedestal
                       else "Jib head above pedestal flange [m]",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="left", x=0),
            margin=dict(l=40, r=20, t=60, b=40),
            uirevision="keep",
        )
        return fig

    @dash.callback(
        Output("p1a-download-data", "data"),
        Input("p1a-btn-download", "n_clicks"),
        State("p1a-main-factor", "value"),
        State("p1a-fold-factor", "value"),
        prevent_initial_call=True,
    )
    def _download_interpolated(n, main_factor, fold_factor):
        if not n:
            return dash.no_update
        main_factor = int(main_factor or 1)
        fold_factor = int(fold_factor or 1)
        _, _, _, _, pts = resample_grid_by_factors(f_angles, m_angles, fold_factor, main_factor)
        df = pd.DataFrame({
            "FoldingJib_deg": pts[:, 0],
            "MainJib_deg":    pts[:, 1],
            "Outreach_m":     outre_itp(pts),
            "Height_m":       height_itp(pts),
        }).dropna()
        return dcc.send_data_frame(df.to_csv, "interpolated_geometry.csv", index=False)
