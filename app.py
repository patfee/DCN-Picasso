import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# Initialize Dash app with multipage support
app = dash.Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    title="DCN Picasso – Crane Tool",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

server = app.server  # for gunicorn / Coolify

# ---------------------------------------------------------------------
# Header section
# ---------------------------------------------------------------------
header = html.Header(
    className="d-flex justify-content-between align-items-center p-2 border-bottom",
    children=[
        html.H2("DCN Picasso – Crane Visualization", className="m-0 ps-3"),
        html.Img(
            src="/assets/dcn_logo.png",  # place your logo file here
            height="48px",
            style={"marginRight": "16px"},
        ),
    ],
)

# ---------------------------------------------------------------------
# Sidebar / Menu
# ---------------------------------------------------------------------
sidebar = html.Div(
    className="border-end bg-light p-3",
    style={
        "width": "230px",
        "height": "calc(100vh - 120px)",
        "overflowY": "auto",
        "position": "fixed",
        "top": "70px",
        "left": "0",
    },
    children=[
        html.H5("Navigation", className="mb-3"),
        html.Div(
            [
                dcc.Link("Page 1 – Crane Geometry", href="/page1", className="d-block mb-2"),
                dcc.Link("Page 2 – Load Curves", href="/page2", className="d-block mb-2"),
                dcc.Link("Page 3 – Settings", href="/page3", className="d-block mb-2"),
            ]
        ),
    ],
)

# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
footer = html.Footer(
    "© DCN Diving B.V.",
    style={
        "textAlign": "center",
        "padding": "10px",
        "borderTop": "1px solid #ddd",
        "marginTop": "10px",
        "color": "#777",
    },
)

# ---------------------------------------------------------------------
# Main Layout
# ---------------------------------------------------------------------
app.layout = html.Div(
    [
        header,
        sidebar,
        html.Div(
            style={
                "marginLeft": "250px",
                "padding": "20px",
                "minHeight": "calc(100vh - 120px)",
            },
            children=[
                dash.page_container,  # holds the current page content
            ],
        ),
        footer,
    ]
)

# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=3000, debug=False)
