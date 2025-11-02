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
    title="My Application"
)
server = app.server  # for gunicorn / Coolify

# --- Header ---
header = html.Header(
    className="app-header",
    children=[
        html.Div("My Application", className="app-title"),
        html.Img(src="/assets/logo.png", className="app-logo", alt="Logo")
    ],
)

# --- Sidebar ---
sidebar = html.Nav(
    className="app-sidebar",
    children=[
        html.Div("Menu", className="sidebar-title"),
        html.Ul(
            className="sidebar-list",
            children=[
                html.Li(dcc.Link("Page 1", href="/page-1", className="sidebar-link")),
                html.Li(dcc.Link("Page 2", href="/page-2", className="sidebar-link")),
                html.Li(dcc.Link("Page 3", href="/page-3", className="sidebar-link")),
            ],
        ),
        html.Div(className="sidebar-footer", children="© Your Company")
    ],
)

# --- Content ---
content = html.Main(className="app-content", children=[dash.page_container])

# --- Layout ---
app.layout = html.Div(className="app-root", children=[header, sidebar, content])

if __name__ == "__main__":
    # Bind to all interfaces; default port 3000 so the app is reachable at 192.168.1.203:3000
    port = int(os.environ.get("PORT", 3000))
    app.run_server(host="0.0.0.0", port=port, debug=True)
