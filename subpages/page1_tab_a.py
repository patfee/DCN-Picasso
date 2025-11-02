from dash import html, dcc, Input, Output, State
import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shapely.geometry import MultiPoint
from shapely.ops import unary_union
import alphashape

from lib.data_utils import load_matrix_csv_flexible
from lib.geo_utils import build_interpolators, resample_grid_by_factors

# ---------- Settings ----------
PEDESTAL_HEIGHT_M = 6.0
HEIGHT_CANDIDATES = ("height.csv", "Height.csv")
OUTREACH_CANDIDATES = ("outreach.csv", "Outreach.csv")
LOAD_CANDIDATES = ("Harbour_Cdyn115.csv", "harbour_Cdyn115.csv")

# ---------- Load matrices ----------
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

    # Originals for scatter + hover
    F0, M0 = np.meshgrid(f_angles, m_angles, indexing="ij")
    orig_xy = np.column_stack([outreach_df.values.ravel(), height_df.values.ravel()])
    orig_custom = np.column_stack([F0.ravel(), M0.ravel()])

    # ---------- Envelope helper ----------
    def compute_tight_outline(points, concavity_scale=1.0):
        """Smooth concave outline following outermost points."""
        pts = np.asarray(points, float)
        pts = pts[np.isfinite(pts).all(axis=1)]
        pts = np.unique(np.round(pts, 6), axis=0)
        if len(pts) < 3:
            return None
        try:
            alpha = alphashape.optimizealpha(pts)
            alpha = max(1e-6, alpha * concavity_scale)
            shape = alphashape.alphashape(pts, alpha)
            shape = unary_union(shape)
            return shape
        except Exception:
            return MultiPoint(pts).convex_hull

    # ---------- Controls ----------
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
                    {"label": "Concave (tight outline)", "value": "concave"},
                    {"label": "Convex hull (fast, stable)", "value": "convex"},
                ],
                value="none", style={"marginTop": "4px"}
            ),
            html.Label("Concavity scale (concave mode only)",
                       style={"fontWeight": 600, "marginTop": "10px"}),
            dcc.Slider(id="p1a-concavity-scale", min=0.5, max=3.0, step=0.1, value=1.0,
                       tooltip={"always_visible": True}),

            html.Div([
                html.Button("Download interpolated CSV", id="p1a-btn-download", n_clicks=0,
                            style={"marginTop": "10px", "width": "100%"}),
                dcc.Download(id="p1a-download-data"),
            ]),

            html.Div("© DCN Diving B.V.",
                     style={"fontSize": "12px", "color": "#666",
                            "marginTop": "15px", "textAlign": "center"})
        ],
    )

    graph = html.Div(
        style={"flex": "1 1 auto", "paddingLeft": "16px"},
        children=[dcc.Graph(id="p1a-graph", style={"height": "78vh"})],
    )

    layout = html.Div(
        className="tab-wrap",
        children=[
            html.H4("Page 1 – Sub A: Height vs Outreach"),
            html.Div([controls, graph], style={"display": "flex", "gap": "10px"})
        ]
    )

    # ---------- Callbacks ----------
    @dash.callback(
        Output("p1a-main-range", "children"),
        Output("p1a-fold-range", "children"),
        Input("p1a-main-factor", "value"),
        Input("p1a-fold-factor", "value"),
    )
    def _show_ranges(main_factor, fold_factor):
        return (
            f"Main-jib angle range: {m_angles.min():.2f}° – {m_angles.max():.2f}° (original points: {len(np.unique(m_angles))})",
            f"Folding-jib angle range: {f_angles.min():.2f}° – {f_angles.max():.2f}° (original points: {len(np.unique(f_angles))})",
        )

    @dash.callback(
        Output("p1a-graph", "figure"),
        Input("p1a-main-factor", "value"),
        Input("p1a-fold-factor", "value"),
        Input("p1a-pedestal-toggle", "value"),
        Input("p1a-envelope-type", "value"),
        Input("p1a-concavity-scale", "value"),
    )
    def _update_figure(main_factor, fold_factor, pedestal_value, envelope_value, concavity_scale):
        include_pedestal = "on" in (pedestal_value or [])

        main_factor = int(main_factor or 1)
        fold_factor = int(fold_factor or 1)

        # Interpolated points
        _, _, _, _, pts = resample_grid_by_factors(f_angles, m_angles, fold_factor, main_factor)
        H_dense = height_itp(pts)
        R_dense = outre_itp(pts)
        if include_pedestal:
            H_dense = H_dense + PEDESTAL_HEIGHT_M
        dense_xy = np.column_stack([R_dense, H_dense])
        dense_xy = dense_xy[~np.isnan(dense_xy).any(axis=1)]

        fig = go.Figure()

        # Interpolated points (light blue)
        if dense_xy.size:
            hover_tmpl = (
                "Main Jib: %{customdata[0]:.2f}°<br>"
                "Folding Jib: %{customdata[1]:.2f}°<br>"
                "Outreach: %{x:.2f} m<br>"
                "Height: %{y:.2f} m<extra></extra>"
            )
            fig.add_trace(go.Scatter(
                x=dense_xy[:, 0], y=dense_xy[:, 1],
                mode="markers", marker=dict(size=3, symbol="diamond", opacity=0.4, color="lightblue"),
                name="Interpolated points",
                hovertemplate=hover_tmpl,
                customdata=pts
            ))

        # Original matrix points (orange)
        if orig_xy.size:
            fig.add_trace(go.Scatter(
                x=orig_xy[:, 0],
                y=orig_xy[:, 1] + (PEDESTAL_HEIGHT_M if include_pedestal else 0.0),
                mode="markers", marker=dict(size=7, color="orange"),
                name="Original matrix points",
                hovertemplate=hover_tmpl,
                customdata=orig_custom
            ))

        # Envelope
        if envelope_value in ("concave", "convex"):
            envelope_pts = orig_xy.copy()
            if include_pedestal:
                envelope_pts[:, 1] += PEDESTAL_HEIGHT_M

            if envelope_value == "concave":
                geom = compute_tight_outline(envelope_pts, concavity_scale)
            else:
                geom = MultiPoint(envelope_pts).convex_hull

            if geom and hasattr(geom, "exterior"):
                x, y = geom.exterior.xy
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode="lines", name="Envelope",
                    line=dict(color="green", width=3)
                ))

        fig.update_layout(
            title="Crane Height vs Outreach",
            xaxis_title="Outreach [m]",
            yaxis_title="Jib head height [m]" if include_pedestal else "Jib head above pedestal flange [m]",
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
            "MainJib_deg": pts[:, 1],
            "Outreach_m": outre_itp(pts),
            "Height_m": height_itp(pts),
        }).dropna()
        return dcc.send_data_frame(df.to_csv, "interpolated_geometry.csv", index=False)
