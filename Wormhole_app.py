# Imports
import dash
from dash import html, dcc
import plotly.graph_objs as go
import numpy as np
from dash.dependencies import Input, Output
from waitress import serve
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# Enhanced Morris-Thorne Metric Function
def enhanced_morris_thorne_metric(r, b0, phi0, spin, exotic_factor):
    eps = 1e-6  # Small number to avoid division by zero
    r = np.maximum(r, eps)  # Ensure r is not zero
    b_r = b0 * np.exp(-r**2 / b0**2) + exotic_factor * np.sin(r)
    phi_r = phi0 * np.exp(-r**2 / b0**2)
    omega = spin * r**2

    g_tt = -np.exp(2 * phi_r)
    g_rr = 1 / (1 - np.maximum(b_r/r, eps))
    g_thth = r**2
    g_phiphi = g_thth * (np.sin(np.pi / 2)**2 + omega)

    return np.array([[g_tt, 0, 0, 0], [0, g_rr, 0, 0], [0, 0, g_thth, 0], [0, 0, 0, g_phiphi]])

# Visualization Function
def create_update_wormhole(b0, phi0, spin, exotic_factor):
    r_range = np.linspace(-10, 10, 100)
    guu_tensors = [enhanced_morris_thorne_metric(r, b0, phi0, spin, exotic_factor) for r in r_range]

    fig = go.Figure()
    for i in range(4):
        for j in range(4):
            zs = [guu[i, j] for guu in guu_tensors if guu is not None]
            fig.add_trace(go.Scatter3d(x=r_range[:len(zs)], y=[i]*len(zs), z=zs, mode='lines', name=f'g{i}{j}'))

    fig.update_layout(scene=dict(xaxis_title='R', yaxis_title='GUU Component', zaxis_title='Value'), margin=dict(l=0, r=0, b=0, t=0))
    return fig

# Feasibility Score Calculation
def compute_feasibility(metric):
    feasibility_score = np.mean(np.abs(metric))
    return feasibility_score

# Bayesian Optimization Setup
space = [Real(1, 10, name='b0'), Real(0, 10, name='phi0'), Real(0, 10, name='spin'), Real(0, 10, name='exotic_factor')]

@use_named_args(space)
def objective(b0, phi0, spin, exotic_factor):
    metric = enhanced_morris_thorne_metric(0, b0, phi0, spin, exotic_factor)
    return -compute_feasibility(metric)

# Dash App Layout
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id='wormhole-plot'),
    html.Label('B0:'), dcc.Slider(id='b0-slider', min=1, max=10, value=1, step=0.1, marks={i: str(i) for i in range(1, 11)}),
    html.Label('Phi0:'), dcc.Slider(id='phi0-slider', min=0, max=10, value=0, step=0.1, marks={i: str(i) for i in range(11)}),
    html.Label('Spin:'), dcc.Slider(id='spin-slider', min=0, max=10, value=0, step=0.1, marks={i: str(i) for i in range(11)}),
    html.Label('Exotic Factor:'), dcc.Slider(id='exotic-factor-slider', min=0, max=10, value=0, step=0.1, marks={i: str(i) for i in range(11)}),
    html.Button('Run Bayesian Optimization', id='run-optimization-button'),
    html.Div(id='optimization-result')
])

# Callbacks
@app.callback(Output('wormhole-plot', 'figure'), [Input('b0-slider', 'value'), Input('phi0-slider', 'value'), Input('spin-slider', 'value'), Input('exotic-factor-slider', 'value')])
def update_graph(b0, phi0, spin, exotic_factor):
    return create_update_wormhole(b0, phi0, spin, exotic_factor)

@app.callback(Output('optimization-result', 'children'), [Input('run-optimization-button', 'n_clicks')])
def run_optimization(n_clicks):
    if n_clicks is None:
        return "Click the button to run optimization."
    result = gp_minimize(objective, space, n_calls=50, acq_func="EI")
    return f"Optimized Parameters: b0={result.x[0]}, phi0={result.x[1]}, spin={result.x[2]}, exotic_factor={result.x[3]}"

# Server Setup
if __name__ == '__main__':
    serve(app.server, host='0.0.0.0', port=8080)
