from dash import html, dcc
import plotly.graph_objs as go
from lib.data_utils import load_crane_data

# Load once (small CSVs); if you want hot-reload, move this into a callback later
df = load_crane_data()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["Outreach [m]"],
    y=df["Height [m]"],
    mode="markers+lines",
    marker=dict(size=6),
    line=dict(width=1.5),
    name="Main Hoist"
))

fig.update_layout(
    title="Tabular data points Main Hoist",
    xaxis_title="Outreach [m]",
    yaxis_title="Jib head above pedestal flange [m]",
    template="plotly_white",
    height=650,
)

layout = html.Div(
    [
        html.H5("Page 1 – Sub A: Height vs Outreach"),
        dcc.Graph(figure=fig),
    ]
)
