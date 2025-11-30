"""
Shared app instance for callbacks to register against.
This file exists to avoid circular import issues.
"""
from flask import Flask
from dash import Dash
import dash_bootstrap_components as dbc

server = Flask(__name__)
app = Dash(
    __name__,
    server=server,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="DCN Picasso Engineering Data"
)
