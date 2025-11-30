from app_instance import app
from dash import html, dcc
layout = html.Div(className="tab-wrap", children=[
    html.H4("Page 3 – Report"),
    dcc.Markdown("Render reports/tables here.")
])
