import os
from flask import Flask
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

# Import page modules
from pages import page1, page2, page3

# Flask + Dash
server = Flask(__name__)
app = Dash(
    __name__,
    server=server,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="My Application 2"
)

# Sidebar (left) + simple header look
sidebar = html.Div(
    [
        html.H5("Menu"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Page 1", href="/page1", active="exact"),
                dbc.NavLink("Page 2", href="/page2", active="exact"),
                dbc.NavLink("Page 3", href="/page3", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
        html.Div("© Your Company", className="mt-4 small text-muted"),
    ],
    className="bg-light border-end p-3 h-100"
)

content = html.Div(
    [
        html.Div(
            [
                html.H3("My Application 2", className="mb-3"),
                dcc.Location(id="url"),
                html.Div(id="page-content")
            ],
            className="p-3"
        )
    ],
    className="h-100"
)

app.layout = dbc.Container(
    dbc.Row(
        [
            dbc.Col(sidebar, width=2),
            dbc.Col(content, width=10),
        ],
        className="vh-100 g-0"
    ),
    fluid=True
)

# Simple router
from dash import Input, Output
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page_content(pathname):
    if pathname in ("/", "/page1"):
        return page1.layout
    if pathname == "/page2":
        return page2.layout
    if pathname == "/page3":
        return page3.layout
    return html.H1("404 - Page Not Found", className="text-danger text-center mt-5")

# Local run: bind to 0.0.0.0:3000 (or $PORT if provided)
if __name__ == "__main__":
    port = int(os.getenv("PORT", "3000"))
    app.run_server(host="0.0.0.0", port=port, debug=True)
