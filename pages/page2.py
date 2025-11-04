from dash import html, dcc, callback, Input, Output
from subpages import page2_tab_a, page2_tab_b, page2_tab_c  # add page2_tab_c

layout = html.Div(
    [
        html.H3("Page 2"),
cc.Tabs(
    id="page2-tabs",
    value="tab-a",
    children=[
        dcc.Tab(label="Tab A – Colored points", value="tab-a", children=page2_tab_a.layout),
        dcc.Tab(label="Tab B – Iso hulls", value="tab-b", children=page2_tab_b.layout),
        dcc.Tab(label="Tab C – Iso hulls (copy)", value="tab-c", children=page2_tab_c.layout),  # new
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
