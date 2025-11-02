from dash import html, dcc
layout = html.Div(className="tab-wrap", children=[
    html.H4("Page 2 – Overview"),
    dcc.Markdown("- Put graphs here\n- KPIs\n- Separate from other tabs")
])
