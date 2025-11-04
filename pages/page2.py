from dash import html, dcc
import dash_bootstrap_components as dbc

# Import sub-tabs
from subpages import page2_tab_a, page2_tab_b, page2_tab_c


# ----------------------------- Layout --------------------------------

layout = html.Div(
    [
        html.H4("Page 2 – Harbour Lift Visualization", className="mb-3"),

        html.Div(
            "This page visualizes crane capacity data aligned with Page 1 interpolation settings "
            "(main/folding jib subdivisions, pedestal height, interpolation mode).",
            className="text-muted mb-4",
        ),

        # Tab control
        dcc.Tabs(
            id="page2-tabs",
            value="tab-a",
            children=[
                dcc.Tab(
                    label="Tab A – Harbour Cdyn 1.15 (Colored Points)",
                    value="tab-a",
                    children=page2_tab_a.layout,
                    className="p-2",
                ),
                dcc.Tab(
                    label="Tab B – Harbour Cdyn 1.15 (Iso Hulls)",
                    value="tab-b",
                    children=page2_tab_b.layout,
                    className="p-2",
                ),
                dcc.Tab(
                    label="Tab C – Harbour Cdyn 1.15 (Iso Hulls Copy)",
                    value="tab-c",
                    children=page2_tab_c.layout,
                    className="p-2",
                ),
            ],
        ),

        html.Hr(className="mt-4 mb-3"),

        dbc.Alert(
            [
                html.Strong("Tip: "),
                "switch between tabs to compare coloured scatter data (Tab A) "
                "and filled iso-capacity hulls (Tab B / C).",
            ],
            color="info",
            className="mt-2",
        ),
    ]
)

