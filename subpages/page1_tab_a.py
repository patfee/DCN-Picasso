from dash import html, dcc
import plotly.graph_objs as go
from lib.data_utils import load_crane_points

# Build tidy (Outreach, Height) pairs from the angle grids
df = load_crane_points()

fig = go.Figure()

# Points cloud
fig.add_trace(go.Scatter(
    x=df["Outreach [m]"],
    y=df["Height [m]"],
    mode="markers",
    marker=dict(size=6),
    name="Data points"
))

# Axes, title, grid
fig.update_layout(
    title="Tabular data points Main Hoist",
    xaxis_title="Outreach [m]",
    yaxis_title="Jib head above pedestal flange [m]",
    template="plotly_white",
    height=720,
)

# Sensible ranges (respect negatives)
if not df.empty:
    x_min, x_max = df["Outreach [m]"].min(), df["Outreach [m]"].max()
    y_min, y_max = df["Height [m]"].min(),   df["Height [m]"].max()
    x_pad = max(0.5, 0.03 * (x_max - x_min))
    y_pad = max(0.5, 0.03 * (y_max - y_min))
    fig.update_xaxes(range=[x_min - x_pad, x_max + x_pad], zeroline=True)
    fig.update_yaxes(range=[y_min - y_pad, y_max + y_pad], zeroline=True)

layout = html.Div(
    [
        html.H5("Page 1 – Sub A: Height vs Outreach"),
        dcc.Graph(figure=fig),
    ]
)
