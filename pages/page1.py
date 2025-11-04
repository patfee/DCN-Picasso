from dash import html, dcc, callback, Input, Output
from subpages import page1_tab_a, page1_tab_b

layout = html.Div(
    [
        html.H3("Page 1"),
        dcc.Tabs(
            id="tabs-page1",
            value="tab-a",
            children=[
                dcc.Tab(label="Sub A", value="tab-a"),
                dcc.Tab(label="Sub B", value="tab-b"),
            ],
        ),
        html.Div(id="tabs-content-page1", className="mt-3"),
    ]
)

@callback(Output("tabs-content-page1", "children"), Input("tabs-page1", "value"))
def render_tab(tab):
    if tab == "tab-a":
        return page1_tab_a.layout
    if tab == "tab-b":
        return page1_tab_b.layout
    return html.Div("Select a tab")
