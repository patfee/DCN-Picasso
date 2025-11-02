from dash import html, dcc
from lib.data_utils import load_csv

df = load_csv("sample.csv")

layout = html.Div(className="tab-wrap", children=[
    html.H4("Page 1 – Sub A"),
    html.P("Reading shared CSV from /data via lib.data_utils."),
    dcc.Textarea(
        value=df.head().to_string(index=False),
        readOnly=True,
        style={"width": "100%", "height": "200px", "fontFamily": "monospace"}
    )
])
