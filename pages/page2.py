from dash import html, dcc
import dash_bootstrap_components as dbc
from subpages import page2_tab_a

layout = html.Div(
    [
        html.H3("Page 2"),
        dcc.Tabs(
            id="tabs-page2",
            value="page2-tab-a",
            children=[
                dcc.Tab(label="Sub A", value="page2-tab-a"),
                # You can add Sub B later if needed
            ],
        ),
        html.Div(id="page2-tab-content"),
    ]
)

def render_tab_content(tab_value):
    if tab_value == "page2-tab-a":
        return page2_tab_a.layout
    return html.Div("Coming soon…")

# Dash callback to switch tab content (this is usually in app.py or here if you wired it this way)
from dash import callback, Input, Output
@callback(Output("page2-tab-content", "children"), Input("tabs-page2", "value"))
def _switch_tab(tab_value):
    return render_tab_content(tab_value)
