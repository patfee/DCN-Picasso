from dash import html, dcc, callback, Input, Output
from subpages import page2_tab_a

layout = html.Div(
    [
        html.H3("Page 2"),
        dcc.Tabs(
            id="tabs-page2",
            value="page2-tab-a",
            children=[
                dcc.Tab(label="Sub A", value="page2-tab-a"),  # Harbour Cdyn 1.15
                # dcc.Tab(label="Sub B", value="page2-tab-b"),  # future
            ],
        ),
        html.Div(id="page2-tab-content"),
    ]
)

@callback(Output("page2-tab-content", "children"), Input("tabs-page2", "value"))
def _switch_tab(tab_value):
    if tab_value == "page2-tab-a":
        return page2_tab_a.layout
    return html.Div("Coming soon…")
