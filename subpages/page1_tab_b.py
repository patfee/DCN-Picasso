from app_instance import app
from dash import html, dcc
layout = html.Div(className="tab-wrap", children=[
    html.H4("Page 1 – Overview"),
    dcc.Markdown("- Put graphs here\n- KPIs\n- Separate from other tabs")
])

