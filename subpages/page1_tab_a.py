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

    # Originals -> orange points + hover (with angles)
    F0, M0 = np.meshgrid(f_angles, m_angles, indexing="ij")
    orig_outreach = outreach_df.values.ravel()
    orig_height   = height_df.values.ravel()
    orig_fold_deg = F0.ravel()
    orig_main_deg = M0.ravel()

    # Pack as XY and customdata for hover
    orig_xy = np.column_stack([orig_outreach, orig_height])
    orig_custom = np.column_stack([orig_fold_deg, orig_main_deg])

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

            html.Label("Concavity (concave mode only)", style={"fontWeight": 600, "marginTop": "8px"}),
            dcc.Slider(
                id="p1a-alpha-scale",
                min=0.25, max=3.0, step=0.05, value=1.0,
                tooltip={"placement": "bottom"},
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

    @dash.callback(
        Output("p1a-envelope-help", "children"),
        Input("p1a-envelope-type", "value"),
        Input("p1a-alpha-scale", "value"))
    def _explain(kind, a):
        if kind == "none":
            return "Points only."
        if kind == "concave":
            return f"Concave alpha shape (requires alphashape+shapely). Scale={a:.2f}×."
        return "Convex hull: stable outer boundary using SciPy only."

    @dash.callback(
        Output("p1a-graph", "figure"),
        Input("p1a-main-factor", "value"),
        Input("p1a-fold-factor", "value"),
        Input("p1a-pedestal-toggle", "value"),
        Input("p1a-envelope-type", "value"),
        Input("p1a-alpha-scale", "value"),
    )
    def _update_figure(main_factor, fold_factor, pedestal_value, envelope_value, alpha_scale):
        include_pedestal = "on" in (pedestal_value or [])
        prefer_concave = (envelope_value == "concave")
        draw_envelope = (envelope_value != "none")

        main_factor = int(main_factor or 1)
        fold_factor = int(fold_factor or 1)

        # --- Interpolated grid (for blue points) ---
        _, _, _, _, pts = resample_grid_by_factors(f_angles, m_angles, fold_factor, main_factor)
        H_dense = height_itp(pts)
        R_dense = outre_itp(pts)
        if include_pedestal:
            H_dense = H_dense + PEDESTAL_HEIGHT_M

        # Filter NaNs, then sample indices (so we can carry angles in hover)
        valid = ~(np.isnan(H_dense) | np.isnan(R_dense))
        pts_valid = pts[valid]
        H_valid = H_dense[valid]
        R_valid = R_dense[valid]

        idx_plot = np.linspace(0, len(pts_valid) - 1, num=min(len(pts_valid), 15000), dtype=int)
        pts_plot = pts_valid[idx_plot]                 # [:,0]=Folding, [:,1]=Main
        dense_xy = np.column_stack([R_valid[idx_plot], H_valid[idx_plot]])

        # --- Envelope from ORIGINAL matrix points (outer red dots) ---
        if include_pedestal:
            env_orig_xy = np.column_stack([orig_outreach, orig_height + PEDESTAL_HEIGHT_M])
        else:
            env_orig_xy = np.column_stack([orig_outreach, orig_height])

        fallback_note = ""
        envelope_xy = None
        if draw_envelope:
            envelope_xy = compute_boundary_curve(
                _sample_points(env_orig_xy, 6000),
                prefer_concave=prefer_concave,
                alpha_scale=(alpha_scale or 1.0)
            )
            if envelope_xy is None:
                fallback_note = " (envelope unavailable: try increasing concavity or install alphashape/shapely)"

        fig = go.Figure()

        # Interpolated points (blue) — with angles in hover
        if len(dense_xy):
            fig.add_trace(go.Scatter(
                x=dense_xy[:, 0], y=dense_xy[:, 1],
                mode="markers",
                marker=dict(size=4, symbol="diamond", opacity=0.5),
                name="Interpolated points",
                customdata=pts_plot,  # [Folding_deg, Main_deg]
                hovertemplate=(
                    "Main Jib: %{customdata[1]:.2f}°<br>"
                    "Folding Jib: %{customdata[0]:.2f}°<br>"
                    "Outreach: %{x:.2f} m<br>"
                    "Height: %{y:.2f} m<extra></extra>"
                )
            ))

        # Original matrix points (orange) — with angles in hover
        fig.add_trace(go.Scatter(
            x=env_orig_xy[:, 0],
            y=env_orig_xy[:, 1],
            mode="markers", marker=dict(size=8),
            name="Original matrix points",
            customdata=np.column_stack([orig_fold_deg, orig_main_deg]),
            hovertemplate=(
                "Main Jib: %{customdata[1]:.2f}°<br>"
                "Folding Jib: %{customdata[0]:.2f}°<br>"
                "Outreach: %{x:.2f} m<br>"
                "Height: %{y:.2f} m<extra></extra>"
            )
        ))

        # Envelope (green)
        if envelope_xy is not None and len(envelope_xy) > 2:
            fig.add_trace(go.Scatter(
                x=envelope_xy[:, 0], y=envelope_xy[:, 1],
                mode="lines", line=dict(width=3), name="Envelope",
            ))

        fig.update_layout(
            title="Crane Height vs Outreach" + fallback_note,
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
