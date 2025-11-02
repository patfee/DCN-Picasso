import dash
from dash import html, dcc, Output, Input
from subpages.page1_tab_a import layout as tab_a_layout
from subpages.page1_tab_b import layout as tab_b_layout

dash.register_page(__name__, path="/page-1", name="Page 1", title="Page 1")

tabs = dcc.Tabs(
    id="page1-tabs",
    value="tab-a",
    children=[
        dcc.Tab(label="Sub A", value="tab-a"),
        dcc.Tab(label="Sub B", value="tab-b"),
    ],
)

layout = html.Div(className="page-wrap", children=[
    html.H2("Page 1"),
    tabs,
    html.Div(id="page1-tab-content")
])

@dash.callback(Output("page1-tab-content", "children"), Input("page1-tabs", "value"))
def _render_page1_tab(tab_value):
    return tab_a_layout if tab_value == "tab-a" else tab_b_layout
