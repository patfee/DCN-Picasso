import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import os

# --- App ---
app = dash.Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    assets_folder="assets",
    title="DCN Picasso – Crane Tool",
)
server = app.server  # for gunicorn / Coolify

# --- Header (title left, logo right) ---
header = html.Header(
    [
        html.H2("DCN Picasso – Crane Visualization", style={"margin": 0}),
        # Use your existing logo file name here (logo.png). Change if different.
        html.Img(src="/assets/logo.png", alt="DCN Logo", height="48px"),
    ],
    style={
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
        "padding": "10px 16px",
        "borderBottom": "1px solid #eee",
        "position": "sticky",
        "top": 0,
        "zIndex": 1000,
        "background": "white",
    },
)

# --- Sidebar (left menu) ---
SIDEBAR_WIDTH = 240
sidebar = html.Nav(
    [
        html.Div("Menu", style={"fontWeight": 600, "marginBottom": 8}),
        html.Ul(
            [
                # Make sure these paths match your pages' registered paths.
                # Most common in Dash Pages: /page1, /page2, /page3
                html.Li(dcc.Link("Page 1", href="/page1")),
                html.Li(dcc.Link("Page 2", href="/page2")),
                html.Li(dcc.Link("Page 3", href="/page3")),
            ],
            style={"listStyle": "none", "padding": 0, "margin": 0, "lineHeight": "2.0"},
        ),
        html.Div("© DCN Diving B.V.", style={"marginTop": "auto", "color": "#777"}),
    ],
    style={
        "position": "fixed",
        "top": 62,  # just under header
        "left": 0,
        "bottom": 0,
        "width": f"{SIDEBAR_WIDTH}px",
        "padding": "12px 16px",
        "borderRight": "1px solid #eee",
        "background": "#fafafa",
        "overflowY": "auto",
        "display": "flex",
        "flexDirection": "column",
        "gap": "8px",
    },
)

# --- Content ---
content = html.Main(
    [dash.page_container],
    style={
        "marginLeft": f"{SIDEBAR_WIDTH + 16}px",
        "padding": "16px",
        "minHeight": "calc(100vh - 62px)",
    },
)

# --- Layout ---
app.layout = html.Div([header, sidebar, content])

if __name__ == "__main__":
    # Bind to all interfaces; default port 3000 so the app is reachable at 192.168.1.203:3000
    port = int(os.environ.get("PORT", 3000))
    app.run_server(host="0.0.0.0", port=port, debug=True)
