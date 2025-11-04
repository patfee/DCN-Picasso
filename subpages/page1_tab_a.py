from dash import html, dcc
import plotly.graph_objs as go
from lib.data_utils import load_crane_points

df = load_crane_points()

# Scatter points (you can add hull/outline later)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["Outreach [m]"],
    y=df["Height [m]"],
    mode="markers",
    marker=dict(size=6),
    name="Data points"
))

fig.update_layout(
    title="Tabular data points Main Hoist",
    xaxis_title="Outreach [m]",
    yaxis_title="Jib head above pedestal flange [m]",
    template="plotly_white",
    height=700,
)

# Optional: set sensible ranges from data (respects negatives)
if not df.empty:
    xpad = max(1.0, 0.03 * (df["Outreach [m]"].max() - df["Outreach [m]"].min()))
    ypad = max(1.0, 0.03 * (df["Height [m]"].max() - df["Height [m]"].min()))
    fig.update_xaxes(range=[df["Outreach [m]"].min() - xpad, df["Outreach [m]"].max() + xpad], zeroline=True)
    fig.update_yaxes(range=[df["Height [m]"].min() - ypad, df["Height [m]"].max() + ypad], zeroline=True)

layout = html.Div(
    [
        html.H5("Page 1 – Sub A: Height vs Outreach"),
        dcc.Graph(figure=fig),
    ]
)
