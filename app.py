import os
from dash import Dash, html, dcc, Input, Output, callback, no_update
import dash_bootstrap_components as dbc

# Import page modules
from pages import page1, page2, page3

# -----------------------------------------------------------------------------
# App / Server
# -----------------------------------------------------------------------------
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,   # allow callbacks defined in submodules
    title="Crane Portal",
)
server = app.server  # for gunicorn

# -----------------------------------------------------------------------------
# Global App Config store
# -----------------------------------------------------------------------------
# This Store holds the user selections that all pages rely on (interp mode, factors, pedestal, etc.)
# It should already be read by your page callbacks (id="app-config").
DEFAULT_CONFIG = {
    "interp_mode": "linear",       # or "spline"
    "main_factor": 1,              # 1, 2, 4, 8, 16
    "folding_factor": 1,           # 1, 2, 4, 8, 16, 32
    "include_pedestal": False,     # toggle
    "pedestal_height": 6.0,        # meters
}

# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------
def navbar():
    return dbc.Nav(
        [
            dbc.NavItem(dbc.NavLink("Page 1", href="/page1", id="link-page1")),
            dbc.NavItem(dbc.NavLink("Page 2", href="/page2", id="link-page2")),
            dbc.NavItem(dbc.NavLink("Page 3", href="/page3", id="link-page3")),
        ],
        pills=True,
        className="gap-2",
    )

app.layout = dbc.Container(
    fluid=True,
    children=[
        dcc.Location(id="url"),
        dcc.Store(id="app-config", data=DEFAULT_CONFIG, storage_type="memory"),
        # Top bar
        dbc.Row(
            [
                dbc.Col(html.H3("Crane Engineering Portal"), md="auto"),
                dbc.Col(navbar(), md="auto"),
            ],
            align="center",
            className="my-3 gy-2",
        ),
        # Pages are mounted once and then only shown/hidden -> state persists, no redraw
        html.Div(
            [
                html.Div(id="page1-container", children=page1.layout, style={"display": "block"}),
                html.Div(id="page2-container", children=page2.layout, style={"display": "none"}),
                html.Div(id="page3-container", children=page3.layout, style={"display": "none"}),
            ],
            id="pages-host",
        ),
    ],
)

# -----------------------------------------------------------------------------
# Routing: show/hide pages without destroying them (keeps figures & tables as-is)
# -----------------------------------------------------------------------------
@callback(
    Output("page1-container", "style"),
    Output("page2-container", "style"),
    Output("page3-container", "style"),
    Output("link-page1", "active"),
    Output("link-page2", "active"),
    Output("link-page3", "active"),
    Input("url", "pathname"),
)
def route(pathname):
    # default route
    if pathname in (None, "/", ""):
        pathname = "/page1"

    show = {"display": "block"}
    hide = {"display": "none"}

    if pathname.startswith("/page1"):
        return show, hide, hide, True, False, False
    if pathname.startswith("/page2"):
        return hide, show, hide, False, True, False
    if pathname.startswith("/page3"):
        return hide, hide, show, False, False, True

    # fallback
    return show, hide, hide, True, False, False

# -----------------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Respect your local Coolify port 3000
    port = int(os.environ.get("PORT", "3000"))
    app.run_server(host="0.0.0.0", port=port, debug=False)
