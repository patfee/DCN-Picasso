# ... existing imports ...
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

# ... existing app/server code ...

content = html.Div(
    [
        # Session-scoped application config available to all pages
        dcc.Store(
            id="app-config",
            storage_type="session",
            data={
                "include_pedestal": False,  # default off
                "pedestal_height": 6.0,     # default 6 m
            },
        ),
        html.Div(
            [
                html.H3("My Application 2", className="mb-3"),
                dcc.Location(id="url"),
                html.Div(id="page-content")
            ],
            className="p-3"
        )
    ],
    className="h-100"
)
