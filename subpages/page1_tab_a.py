import dash
dash.register_page(__name__, path="/page-1/sub-a", name="Page 1 – Sub A")

from dash import html, dcc, Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Optional deps (safe import)
try:
    from shapely.geometry import MultiPoint
    from shapely.ops import unary_union
    import alphashape
    _HAS_ALPHASHAPE = True
except Exception:
    _HAS_ALPHASHAPE = False

from scipy.spatial import ConvexHull
from lib.data_utils import load_matrix_csv_flexible
from lib.geo_utils import build_interpolators, resample_grid_by_factors

# Default layout so Dash always finds one
layout = html.Div([html.H4("Page 1 – Sub A: Height vs Outreach"), html.P("Initializing …")])

# ---------------- Helper ----------------
def _safe_outline(points_xy, concavity_scale=1.0):
    pts = np.asarray(points_xy, float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 3:
        return None
    if _HAS_ALPHASHAPE:
        try:
            mp = MultiPoint(pts)
            alpha = alphashape.optimizealpha(mp)
            alpha = float(alpha) * float(concavity_scale)
            poly = alphashape.alphashape(mp, alpha)
            if poly.is_empty:
                return None
            if poly.geom_type == "MultiPolygon":
                poly = unary_union(poly)
            x, y = poly.exterior.coords.xy
            return np.column_stack([x, y])
        except Exception:
            pass
    try:
        hull = ConvexHull(pts)
        ring = pts[hull.vertices]
        return np.vstack([ring, ring[0]])
    except Exception:
        return None

# ---------------- Main guarded logic ----------------
try:
    HEIGHT_CANDIDATES = ("height.csv", "Height.csv")
    OUTREACH_CANDIDATES = ("outreach.csv", "Outreach.csv")
    LOAD_CANDIDATES = ("Harbour_Cdyn115.csv", "harbour_Cdyn115.csv")

    height_df = load_matrix_csv_flexible(HEIGHT_CANDIDATES)
    outreach_df = load_matrix_csv_flexible(OUTREACH_CANDIDATES)
    load_df = load_matrix_csv_flexible(LOAD_CANDIDATES)

    if height_df.empty or outreach_df.empty:
        layout = html.Div([
            html.H4("Page 1 – Sub A: Height vs Outreach"),
            html.P(f"No matrix CSVs found in /data (tried {HEIGHT_CANDIDATES} and {OUTREACH_CANDIDATES}).")
        ])
    else:
        f_angles, m_angles, height_itp, outre_itp, load_itp = build_interpolators(height_df, outreach_df, load_df)

        F0, M0 = np.meshgrid(f_angles, m_angles, indexing="ij")
        orig_xy = np.column_stack([outreach_df.values.ravel(), height_df.values.ravel()])
        orig_custom = np.column_stack([F0.ravel(), M0.ravel()])

        controls = html.Div([
            html.Label("Main-jib subdivision"),
            dcc.Dropdown(
                id="p1a-main-factor",
                options=[{"label": f"{o}× per-interval" + (" (original)" if o==1 else ""), "value": o} for o in [1,2,4,8]],
                value=1, clearable=False),
            html.Label("Folding-jib subdivision"),
            dcc.Dropdown(
                id="p1a-fold-factor",
                options=[{"label": f"{o}× per-interval" + (" (original)" if o==1 else ""), "value": o} for o in [1,2,4,8,16]],
                value=1, clearable=False),
            html.Label("Concavity scale (outline tightening)"),
            dcc.Slider(0.6, 1.8, step=0.05, value=1.0, id="p1a-concavity"),
            html.Hr(),
            html.Button("Download interpolated CSV", id="p1a-btn-download", n_clicks=0),
            dcc.Download(id="p1a-download-data")
        ], style={"width":"300px","paddingRight":"12px"})

        graph = html.Div(dcc.Graph(id="p1a-graph", style={"height":"820px"}), style={"flex":"1"})

        layout = html.Div(className="tab-wrap",
            children=[
                html.H4("Page 1 – Sub A: Height vs Outreach"),
                html.Div([controls, graph], style={"display":"flex","gap":"10px"})
            ]
        )

        @dash.callback(
            Output("p1a-graph", "figure"),
            Input("p1a-main-factor", "value"),
            Input("p1a-fold-factor", "value"),
            Input("p1a-concavity", "value"),
            prevent_initial_call=False,
        )
        def _update_chart(main_factor, fold_factor, concavity_scale):
            main_factor, fold_factor = int(main_factor or 1), int(fold_factor or 1)
            F_dense, M_dense, R_dense, H_dense, pts = resample_grid_by_factors(
                f_angles, m_angles, fold_factor, main_factor
            )

            # Flatten to avoid shape mismatch
            Rf, Hf = np.ravel(R_dense), np.ravel(H_dense)
            Pf = np.reshape(pts, (-1, pts.shape[-1])) if pts.ndim > 1 else pts
            mask = (~np.isnan(Rf)) & (~np.isnan(Hf))
            Rf, Hf, Pf = Rf[mask], Hf[mask], Pf[mask]

            fig = go.Figure()
            if Rf.size:
                fig.add_trace(go.Scatter(
                    x=Rf, y=Hf, mode="markers",
                    marker=dict(size=4, symbol="diamond", opacity=0.45, color="royalblue"),
                    name="Subdivision points",
                    customdata=Pf[:, :2],
                    hovertemplate="Main Jib: %{customdata[1]:.2f}°<br>Folding Jib: %{customdata[0]:.2f}°<br>"
                                  "Outreach: %{x:.2f} m<br>Height: %{y:.2f} m<extra></extra>"
                ))
            fig.add_trace(go.Scatter(
                x=orig_xy[:,0], y=orig_xy[:,1],
                mode="markers", marker=dict(size=7, color="midnightblue"),
                name="Original matrix points",
                customdata=orig_custom,
                hovertemplate="Main Jib: %{customdata[1]:.2f}°<br>Folding Jib: %{customdata[0]:.2f}°<br>"
                              "Outreach: %{x:.2f} m<br>Height: %{y:.2f} m<extra></extra>"
            ))
            outline = _safe_outline(orig_xy, concavity_scale)
            if outline is not None and outline.size:
                fig.add_trace(go.Scatter(
                    x=outline[:,0], y=outline[:,1],
                    mode="lines+markers", line=dict(color="midnightblue", width=2),
                    marker=dict(size=5, color="midnightblue"), name="Envelope"
                ))
            fig.update_layout(
                title=dict(text="<b>Tabular data points Main Hoist</b>", x=0.5),
                xaxis_title="Outreach [m]", yaxis_title="Jib head above pedestal flange [m]",
                plot_bgcolor="white", hovermode="closest"
            )
            return fig

        @dash.callback(
            Output("p1a-download-data", "data"),
            Input("p1a-btn-download", "n_clicks"),
            State("p1a-main-factor", "value"),
            State("p1a-fold-factor", "value"),
            prevent_initial_call=True,
        )
        def _download(n, main_factor, fold_factor):
            if not n:
                return dash.no_update
            _, _, R_dense, H_dense, pts = resample_grid_by_factors(f_angles, m_angles, fold_factor, main_factor)
            Rf, Hf = np.ravel(R_dense), np.ravel(H_dense)
            Pf = np.reshape(pts, (-1, pts.shape[-1])) if pts.ndim > 1 else pts
            mask = (~np.isnan(Rf)) & (~np.isnan(Hf))
            df = pd.DataFrame({
                "FoldingJib_deg": Pf[mask, 0],
                "MainJib_deg": Pf[mask, 1],
                "Outreach_m": Rf[mask],
                "Height_m": Hf[mask],
            })
            return dcc.send_data_frame(df.to_csv, "interpolated_geometry.csv", index=False)

except Exception as e:
    layout = html.Div([
        html.H4("Page 1 – Sub A: Height vs Outreach"),
        html.Pre(f"Failed to initialize this tab.\nReason: {type(e).__name__}: {e}")
    ])
