from dash import html, dcc, callback, Input, Output
from subpages import page2_tab_a, page2_tab_b

layout = html.Div(
    [
        html.H3("Page 2"),
        dcc.Tabs(
            id="tabs-page2",
            value="page2-tab-a",
            children=[
                dcc.Tab(label="Sub A", value="page2-tab-a"),  # coloured point cloud
                dcc.Tab(label="Sub B", value="page2-tab-b"),  # iso-capacity hulls
            ],
        ),
        html.Div(id="page2-tab-content"),
    ]
)

@callback(Output("page2-tab-content", "children"), Input("tabs-page2", "value"))
def _switch_tab(tab_value):
    if tab_value == "page2-tab-a":
        return page2_tab_a.layout
    if tab_value == "page2-tab-b":
        return page2_tab_b.layout
    return html.Div("Coming soon…")
