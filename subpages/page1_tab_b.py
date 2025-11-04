import dash
dash.register_page(__name__, path="/page-1/sub-b", name="Page 1 – Sub B")

from dash import html, dcc
layout = html.Div([
    html.H4("Page 1 – Sub B : Tabular data points Main Hoist"),
    html.P("Initializing … Please wait or check /data CSV files if this persists.")
])

try:
    from dash import Input, Output
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from scipy.spatial import ConvexHull
    try:
        import alphashape
        from shapely.geometry import MultiPoint
        from shapely.ops import unary_union
        _HAS_ALPHASHAPE = True
    except Exception:
        _HAS_ALPHASHAPE = False

    from lib.data_utils import load_matrix_csv_flexible
    from lib.geo_utils import build_interpolators, resample_grid_by_factors

    HEIGHT_CANDIDATES = ("height.csv", "Height.csv")
    OUTREACH_CANDIDATES = ("outreach.csv", "Outreach.csv")

    height_df = load_matrix_csv_flexible(HEIGHT_CANDIDATES)
    outreach_df = load_matrix_csv_flexible(OUTREACH_CANDIDATES)

    if height_df.empty or outreach_df.empty:
        layout = html.Div([
            html.H4("Page 1 – Sub B : Tabular data points Main Hoist"),
            html.P(
                "No matrix CSVs found. Ensure files exist in /data "
                f"(tried {HEIGHT_CANDIDATES} and {OUTREACH_CANDIDATES})."
            )
        ])
    else:
        f_angles, m_angles, height_itp, outre_itp, _ = build_interpolators(
            height_df, outreach_df, pd.DataFrame()
        )
        F0, M0 = np.meshgrid(f_angles, m_angles, indexing="ij")
        orig_xy = np.column_stack([outreach_df.values.ravel(), height_df.values.ravel()])
        orig_custom = np.column_stack([F0.ravel(), M0.ravel()])

        def _safe_outline(points, concavity_scale=1.0):
            pts = np.asarray(points, float)
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

        layout = html.Div(
            className="tab-wrap",
            children=[
                html.H4("Page 1 – Sub B : Tabular data points Main Hoist"),
                html.Div(
                    className="two-col",
                    children=[
                        html.Div(
                            children=[
                                html.Label("Main-jib subdivision", style={"fontWeight": 600}),
                                dcc.Dropdown(
                                    id="p1b-main-factor",
                                    options=[{"label": f"{o}× per-interval" + (" (original)" if o == 1 else ""), "value": o}
                                             for o in [1, 2, 4, 8]],
                                    value=1, clearable=False),
                                html.Label("Folding-jib subdivision", style={"fontWeight": 600}),
                                dcc.Dropdown(
                                    id="p1b-fold-factor",
                                    options=[{"label": f"{o}× per-interval" + (" (original)" if o == 1 else ""), "value": o}
                                             for o in [1, 2, 4, 8, 16]],
                                    value=1, clearable=False),
                                html.Label("Concavity scale (outline tightening)"),
                                dcc.Slider(0.6, 1.8, step=0.05, value=1.0, id="p1b-concavity"),
                            ],
                            style={"minWidth": "260px", "maxWidth": "360px", "paddingRight": "12px"}
                        ),
                        dcc.Loading(
                            dcc.Graph(id="p1b-graph", style={"height": "820px"}),
                            type="dot"
                        ),
                    ],
                ),
                html.Div(id="p1b-range-info", style={"fontFamily": "monospace", "marginTop": "8px"}),
            ],
        )

        @dash.callback(
            Output("p1b-graph", "figure"),
            Output("p1b-range-info", "children"),
            Input("p1b-main-factor", "value"),
            Input("p1b-fold-factor", "value"),
            Input("p1b-concavity", "value"),
            prevent_initial_call=False,
        )
        def _update_chart(main_factor, fold_factor, concavity_scale):
            main_factor = int(main_factor or 1)
            fold_factor = int(fold_factor or 1)
            F_dense, M_dense, R_dense, H_dense, pts = resample_grid_by_factors(
                f_angles, m_angles, fold_factor, main_factor
            )
            dense_xy = np.column_stack([R_dense, H_dense])
            dense_xy = dense_xy[~np.isnan(dense_xy).any(axis=1)]

            fig = go.Figure()
            if dense_xy.size:
                fig.add_trace(go.Scatter(
                    x=dense_xy[:, 0], y=dense_xy[:, 1],
                    mode="markers",
                    marker=dict(size=4, symbol="diamond", opacity=0.45, color="royalblue"),
                    name="Subdivision points",
                    customdata=pts[:, :2],
                    hovertemplate=(
                        "Main Jib: %{customdata[1]:.2f}°<br>"
                        "Folding Jib: %{customdata[0]:.2f}°<br>"
                        "Outreach: %{x:.2f} m<br>"
                        "Jib head above pedestal flange: %{y:.2f} m<extra></extra>"
                    )
                ))
            if orig_xy.size:
                fig.add_trace(go.Scatter(
                    x=orig_xy[:, 0], y=orig_xy[:, 1],
                    mode="markers",
                    marker=dict(size=7, color="midnightblue"),
                    name="Original matrix points",
                    customdata=orig_custom,
                    hovertemplate=(
                        "Main Jib: %{customdata[1]:.2f}°<br>"
                        "Folding Jib: %{customdata[0]:.2f}°<br>"
                        "Outreach: %{x:.2f} m<br>"
                        "Jib head above pedestal flange: %{y:.2f} m<extra></extra>"
                    )
                ))
            outline = _safe_outline(orig_xy, concavity_scale)
            if outline is not None and outline.size:
                fig.add_trace(go.Scatter(
                    x=outline[:, 0], y=outline[:, 1],
                    mode="lines+markers",
                    line=dict(color="midnightblue", width=2),
                    marker=dict(size=5, color="midnightblue"),
                    name="Envelope"
                ))
            fig.update_layout(
                title=dict(text="<b>Tabular data points Main Hoist</b>", x=0.5),
                xaxis=dict(title="Outreach [m]", gridcolor="#D9D9D9"),
                yaxis=dict(title="Jib head above pedestal flange [m]", gridcolor="#D9D9D9"),
                plot_bgcolor="white",
                hovermode="closest"
            )
            info = (
                f"Main-jib subdivision: {main_factor}× • "
                f"Folding-jib subdivision: {fold_factor}× • "
                f"Angles Main: {m_angles.min():.2f}° – {m_angles.max():.2f}°, "
                f"Folding: {f_angles.min():.2f}° – {f_angles.max():.2f}°"
            )
            return fig, info

except Exception as e:
    layout = html.Div([
        html.H4("Page 1 – Sub B : Tabular data points Main Hoist"),
        html.Pre(f"Failed to initialize Sub B.\nReason: {type(e).__name__}: {e}")
    ])
