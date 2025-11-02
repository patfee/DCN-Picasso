import dash
from dash import html, dcc, Output, Input
from subpages.page2_tab_a import layout as tab_a_layout
from subpages.page2_tab_b import layout as tab_b_layout

dash.register_page(__name__, path="/page-2", name="Page 2", title="Page 2")

tabs = dcc.Tabs(
    id="page2-tabs",
    value="tab-a",
    children=[
        dcc.Tab(label="Overview", value="tab-a"),
        dcc.Tab(label="Details", value="tab-b"),
    ],
)

layout = html.Div(className="page-wrap", children=[
    html.H2("Page 2"),
    tabs,
    html.Div(id="page2-tab-content")
])

@dash.callback(Output("page2-tab-content", "children"), Input("page2-tabs", "value"))
def _render_page2_tab(tab_value):
    return tab_a_layout if tab_value == "tab-a" else tab_b_layout
