import dash
dash.register_page(__name__, path="/page-1/sub-a", name="Page 1 – Sub A")

from dash import html, dcc, Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---- Optional deps (guarded) ----
try:
    from shapely.geometry import MultiPoint
    from shapely.ops import unary_union
    import alphashape
    _HAS_ALPHASHAPE = True
except Exception:
    MultiPoint = None
    unary_union = None
    alphashape = None
    _HAS_ALPHASHAPE = False

from scipy.spatial import ConvexHull

from lib.data_utils import load_matrix_csv_flexible
from lib.geo_utils import build_interpolators, resample_grid_by_factors

# ---------------- Settings ----------------
PEDESTAL_HEIGHT_M = 6.0
HEIGHT_CANDIDATES = ("height.csv", "Height.csv")
OUTREACH_CANDIDATES = ("outreach.csv", "Outreach.csv")
LOAD_CANDIDATES = ("Harbour_Cdyn115.csv", "harbour_Cdyn115.csv")

# ---------------- Safe helpers ----------------
def _safe_concave_outline(points_xy: np.ndarray, concavity_scale: float = 1.0):
    """
    Compute a concave outline if alphashape/shapely are present; otherwise fallback to convex hull.
    Returns Nx2 array or None.
    """
    pts = np.asarray(points_xy, float)
    if pts.size == 0:
        return None
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 3:
        return None

    # Try concave shape first (optional)
    if _HAS_ALPHASHAPE and MultiPoint is not None:
        try:
            mp = MultiPoint(pts.tolist())
            alpha = alphashape.optimizealpha(mp)
            alpha = float(alpha) * float(concavity_scale)
            poly = alphashape.alphashape(mp, alpha)
            if poly and not poly.is_empty:
                if poly.geom_type == "MultiPolygon" and unary_union is not None:
                    poly = unary_union(poly)
                if hasattr(poly, "exterior"):
                    x, y = poly.exterior.coords.xy
                    outline = np.column_stack([x, y])
                    if len(outline) >= 3:
                        return outline
        except Exception:
            pass

    # Fallback: convex hull
    try:
        hull = ConvexHull(pts)
        idx = np.append(hull.vertices, hull.vertices[0])
        return pts[idx]
    except Exception:
        return None

# ---------------- Default layout in case anything fails ----------------
# (Predefine so Dash's page registry always finds a layout.)
layout = html.Div(
    className="tab-wrap",
    children=[
        html.H4("Page 1 – Sub A: Height vs Outreach"),
        html.P("Initializing… If this persists, check your data files in /data and Python dependencies.")
    ],
)

# ---------------- Build the real layout (guarded) ----------------
try:
    # Load matrices (robust loader)
    height_df = load_matrix_csv_flexible(HEIGHT_CANDIDATES)
    outreach_df = load_matrix_csv_flexible(OUTREACH_CANDIDATES)
    load_df = load_matrix_csv_flexible(LOAD_CANDIDATES) if not height_df.empty else pd.DataFrame()

    if height_df.empty or outreach_df.empty:
        layout = html.Div(
            className="tab-wrap",
            children=[
                html.H4("Page 1 – Sub A: Height vs Outreach"),
                html.P(
                    "No matrix CSVs loaded. Ensure files exist in /data "
                    f"(tried {HEIGHT_CANDIDATES} and {OUTREACH_CANDIDATES})."
                ),
            ],
        )
    else:
        # Build interpolators from your lib
        f_angles, m_angles, height_itp, outre_itp, load_itp = build_interpolators(
            height_df, outreach_df, load_df
        )

        # Original grid points for plotting
        F0, M0 = np.meshgrid(f_angles, m_angles, indexing="ij")
        orig_xy = np.column_stack([outreach_df.values.ravel(), height_df.values.ravel()])
        orig_custom = np.column_stack([F0.ravel(), M0.ravel()])

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
                html.Label("Concavity scale (outline tightening)"),
                dcc.Slider(0.6, 1.8, step=0.05, value=1.0, id="p1a-concavity",
                           tooltip={"placement": "bottom", "always_visible": False}),
                html.Hr(),
                html.Div(id="p1a-main-range", style={"fontFamily": "monospace"}),
                html.Div(id="p1a-fold-range", style={"fontFamily": "monospace"}),
                html.Hr(),
                html.Button("Download interpolated CSV", id="p1a-btn-download", n_clicks=0,
                            style={"marginTop": "10px", "width": "100%"}),
                dcc.Download(id="p1a-download-data"),
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
                f"Main-jib angle range: {m_angles.min():.2f}° – {m_angles.max():.2f}° "
                f"(original points: {len(np.unique(m_angles))})",
                f"Folding-jib angle range: {f_angles.min():.2f}° – {f_angles.max():.2f}° "
                f"(original points: {len(np.unique(f_angles))})",
            )

        @dash.callback(
            Output("p1a-graph", "figure"),
            Input("p1a-main-factor", "value"),
            Input("p1a-fold-factor", "value"),
            Input("p1a-concavity", "value"),
        )
        def _update_chart(main_factor, fold_factor, concavity_scale):
            main_factor = int(main_factor or 1)
            fold_factor = int(fold_factor or 1)

            # Dense sampling
            F_dense, M_dense, R_dense, H_dense, pts = resample_grid_by_factors(
                f_angles, m_angles, fold_factor, main_factor,
                height_itp=height_itp, outre_itp=outre_itp
            )
            # Remove NaNs
            mask = (~np.isnan(R_dense)) & (~np.isnan(H_dense))
            R_dense = R_dense[mask]
            H_dense = H_dense[mask]
            pts_dense = pts[mask]

            # Build figure
            fig = go.Figure()

            # Interpolated points
            if R_dense.size:
                fig.add_trace(go.Scatter(
                    x=R_dense, y=H_dense, mode="markers",
                    marker=dict(size=4, symbol="diamond", opacity=0.45, color="royalblue"),
                    name="Subdivision points",
                    customdata=pts_dense[:, :2],  # (Folding, Main)
                    hovertemplate=(
                        "Main Jib: %{customdata[1]:.2f}°<br>"
                        "Folding Jib: %{customdata[0]:.2f}°<br>"
                        "Outreach: %{x:.2f} m<br>"
                        "Jib head above pedestal flange: %{y:.2f} m<extra></extra>"
                    ),
                ))

            # Original matrix points
            if orig_xy.size:
                fig.add_trace(go.Scatter(
                    x=orig_xy[:, 0], y=orig_xy[:, 1],
                    mode="markers", marker=dict(size=7, color="midnightblue"),
                    name="Original matrix points",
                    customdata=orig_custom,
                    hovertemplate=(
                        "Main Jib: %{customdata[1]:.2f}°<br>"
                        "Folding Jib: %{customdata[0]:.2f}°<br>"
                        "Outreach: %{x:.2f} m<br>"
                        "Jib head above pedestal flange: %{y:.2f} m<extra></extra>"
                    ),
                ))

            # Outline (concave or convex fallback)
            outline = _safe_concave_outline(orig_xy, concavity_scale=float(concavity_scale or 1.0))
            if outline is not None and outline.size:
                fig.add_trace(go.Scatter(
                    x=outline[:, 0], y=outline[:, 1],
                    mode="lines+markers",
                    line=dict(color="midnightblue", width=2),
                    marker=dict(size=5, color="midnightblue"),
                    name="Envelope",
                    hoverinfo="skip",
                ))

            fig.update_layout(
                title=dict(text="<b>Tabular data points Main Hoist</b>", x=0.5, xanchor="center"),
                xaxis=dict(title="Outreach [m]", zeroline=True, gridcolor="#D9D9D9", tickformat=".0f"),
                yaxis=dict(title="Jib head above pedestal flange [m]", gridcolor="#D9D9D9", tickformat=".0f"),
                plot_bgcolor="white",
                hovermode="closest",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=20, t=60, b=40),
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
            _, _, _, _, pts = resample_grid_by_factors(
                f_angles, m_angles, fold_factor, main_factor,
                height_itp=height_itp, outre_itp=outre_itp
            )
            df = pd.DataFrame({
                "FoldingJib_deg": pts[:, 0],
                "MainJib_deg": pts[:, 1],
                "Outreach_m": outre_itp(pts),
                "Height_m": height_itp(pts),
            }).dropna()
            return dcc.send_data_frame(df.to_csv, "interpolated_geometry.csv", index=False)

except Exception as e:
    # Hardening: if anything went wrong above, expose a safe layout so the app still boots.
    layout = html.Div(
        className="tab-wrap",
        children=[
            html.H4("Page 1 – Sub A: Height vs Outreach"),
            html.Pre(
                f"Failed to initialize this tab.\n\nReason: {type(e).__name__}: {e}\n"
                "Check that /data/height.csv and /data/outreach.csv exist and that required "
                "Python packages are installed (numpy, pandas, scipy, plotly; optional: shapely, alphashape)."
            ),
        ],
    )
