import dash
from dash import html, dcc, Output, Input
from subpages.page3_tab_a import layout as tab_a_layout
from subpages.page3_tab_b import layout as tab_b_layout

dash.register_page(__name__, path="/page-3", name="Page 3", title="Page 3")

tabs = dcc.Tabs(
    id="page3-tabs",
    value="tab-a",
    children=[
        dcc.Tab(label="Report", value="tab-a"),
        dcc.Tab(label="Export", value="tab-b"),
    ],
)

layout = html.Div(className="page-wrap", children=[
    html.H2("Page 3"),
    tabs,
    html.Div(id="page3-tab-content")
])

@dash.callback(Output("page3-tab-content", "children"), Input("page3-tabs", "value"))
def _render_page3_tab(tab_value):
    return tab_a_layout if tab_value == "tab-a" else tab_b_layout
