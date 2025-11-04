from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from flask import Flask

# Local imports
from pages import page1, page2, page3

# Create Flask + Dash
server = Flask(__name__)
app = Dash(__name__, server=server, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Sidebar
sidebar = html.Div(
    [
        html.H2("Menu"),
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
        html.Div("© Your Company", className="mt-auto small text-muted text-center")
    ],
    className="bg-light sidebar p-3",
)

# Layout
app.layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col(sidebar, width=2),
            dbc.Col(dcc.Location(id="url"), width=10),
        ]),
        dbc.Row(dbc.Col(html.Div(id="page-content"), width=10))
    ],
    fluid=True
)

# Callbacks for routing
from dash import Input, Output
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page_content(pathname):
    if pathname == "/page1":
        return page1.layout
    elif pathname == "/page2":
        return page2.layout
    elif pathname == "/page3":
        return page3.layout
    else:
        return html.H1("404 - Page Not Found", className="text-danger text-center mt-5")

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=3000, debug=True)
