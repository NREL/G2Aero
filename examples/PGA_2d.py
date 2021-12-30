import os
import numpy as np
import pandas as pd
from src.g2aero import PGAspace
from src.g2aero import position_airfoil, add_tailedge_gap

import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash.dependencies import Input, Output


pga_dict = np.load(os.path.join(os.getcwd(), '../data', 'PGA_space', 'PGA_space.npz'))
pga = PGAspace(pga_dict['Vh'], pga_dict['M_mean'], pga_dict['b_mean'], pga_dict['karcher_mean'])
pga.M_mean = pga.M_mean.T
M = np.load(os.path.join(os.getcwd(), '../data', 'PGA_space', 'M_b.npz'))['M']
b = np.load(os.path.join(os.getcwd(), '../data', 'PGA_space', 'M_b.npz'))['b']
t = np.load(os.path.join(os.getcwd(), '../data', 'PGA_space', 't.npz'))['t']




t = np.hstack((t, M[:, 1:, 1]))
r_min = np.abs(np.quantile(t, 0.005, axis=0))
r_max = np.abs(np.quantile(t, 0.995, axis=0))
pga.radius = np.max(np.array([r_min, r_max]), axis=0)
pga.radius[-1] = (np.max(M[:, 1:, 1]) - np.min(M[:, 1:, 1]))/2
centers = [0, 0, 0, 0, (np.max(M[:, 1:, 1]) + np.min(M[:, 1:, 1]))/2]

airfoils = ['NACA64_A17', 'DU21_A17', 'DU25_A17', 'DU30_A17', 'DU35_A17', 'DU40_A17', 'DU00-W2-350', 'DU08-W-210',
            'FFA-W3-211',  'FFA-W3-241', 'FFA-W3-270blend', 'FFA-W3-301', 'FFA-W3-330blend', 'FFA-W3-360', 'SNL-FFA-W3-500']

colors_pallete = ['#FF00FF', '#00429d', '#2571b0', '#4a9fc3', '#6fced6', '#96ffea', '#008000', '#58ffd8',
                  '#ff9e7c', '#fa7779', '#e9546f', '#d13660', '#b51c4d', '#940638', '#720022']
dict_color = {airfoils[i]: colors_pallete[i] for i in range(len(airfoils))}
df = pd.DataFrame(data=t, columns=['t1', 't2', 't3', 't4', 's2'])
df['label'] = [airfoil for airfoil in airfoils for i in range(1001)]
df['colors'] = [dict_color[airfoil] for airfoil in airfoils for i in range(1001)]


karcher_mean = pga.PGA2shape([0, 0, 0, 0])
karcher_mean = position_airfoil(karcher_mean)
karcher_mean = add_tailedge_gap(karcher_mean, 0.01)
tmin = -pga.radius
tmin[-1] = np.min(M[:, 1:, 1])
tmax = pga.radius
tmax[-1] = np.max(M[:, 1:, 1])


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SKETCHY])

app.layout = html.Div([
    dbc.Row(dbc.Col(html.Div(html.H1(children='Airfoil perturbations', style={'textAlign': 'center'})))),
    dbc.Row([
        dbc.Col(
            html.Div([
                html.H2(children='PGA parameters',
                        style={'textAlign': 'center'}),

                html.Label(id='t1-update'),
                dcc.Slider(id='t1-slider', min=tmin[0], max=tmax[0], step=0.0001,
                           marks={i: f'{i:.2f}' for i in np.arange(np.round(tmin[0], 2), np.round(tmax[0], 2), 0.05)},
                           value=0.0),
                html.Br(),

                html.Label(id='t2-update'),
                dcc.Slider(id='t2-slider',  min=tmin[1], max=tmax[1], step=0.0001,
                           marks={i: f'{i:.2f}' for i in np.arange(np.round(tmin[0], 2), np.round(tmax[0], 2), 0.05)},
                           value=0.0),
                html.Br(),

                html.Label(id='t3-update'),
                dcc.Slider(id='t3-slider', min=tmin[2], max=tmax[2], step=0.0001,
                           marks={i: f'{i:.2f}' for i in np.arange(np.round(tmin[0], 2), np.round(tmax[0], 2), 0.05)},
                           value=0.0),
                html.Br(),

                html.Label(id='t4-update'),
                dcc.Slider(id='t4-slider', min=tmin[3], max=tmax[3], step=0.0001,
                           marks={i: f'{i:.2f}' for i in np.arange(np.round(tmin[0], 2), np.round(tmax[0], 2), 0.05)},
                           value=0.0),
                html.Br(),
                html.H2(children='Affine Transformation parameters',
                        style={'textAlign': 'center'}),

                html.Label(id='M22-update'),
                dcc.Slider(id='M22-slider',  min=tmin[4], max=tmax[4], step=0.001,
                           marks={i: f'{i:.2f}' for i in np.arange(np.round(tmin[4], 2), np.round(tmax[4], 2), 0.5)},
                           value=np.round(pga.M_mean[1, 1], 4)),
                html.Br(),
                dcc.Graph(id='airfoils'),
                html.Br(),
            ]), width={"size": 4, "offset": 1}),
        dbc.Col(
            html.Div([
                dcc.Graph(id='scatterplot'),
            ]), width=6, align="center")
    ])
])


@app.callback(Output('t1-update', 'children'), [Input('t1-slider', 'value')])
def display_value(value):
    return f't1 = {value}'


@app.callback(Output('t2-update', 'children'), [Input('t2-slider', 'value')])
def display_value(value):
    return f't2 = {value}'


@app.callback(Output('t3-update', 'children'), [Input('t3-slider', 'value')])
def display_value(value):
    return f't3 = {value}'


@app.callback(Output('t4-update', 'children'), [Input('t4-slider', 'value')])
def display_value(value):
    return f't3 = {value}'


@app.callback(Output('M22-update', 'children'), [Input('M22-slider', 'value')])
def display_value(value):
    return f'M22 = {value}'


@app.callback(Output('airfoils', 'figure'),
              [Input('t1-slider', 'value'),
               Input('t2-slider', 'value'),
               Input('t3-slider', 'value'),
               Input('t4-slider', 'value'),
               Input('M22-slider', 'value')])
def calculate_shape(t1, t2, t3, t4, M22):

    M = pga.M_mean.copy()
    M[1, 1] = M22
    shape = pga.PGA2shape([t1, t2, t3, t4], M=M)
    shape = position_airfoil(shape)
    shape = add_tailedge_gap(shape, 0.01)

    fig = go.Figure()
    fig.update_xaxes(range=[-0.01, 1.01], scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-0.3, 0.3], scaleanchor="x", scaleratio=1)
    fig.add_trace(go.Scatter(x=karcher_mean[:, 0], y=karcher_mean[:, 1],
                             line=dict(color="#FF0000", width=2), name='Karcher mean'))
    fig.add_trace(go.Scatter(x=shape[:, 0], y=shape[:, 1],
                             line=dict(color="#000000", width=3), name='Perturbation'))
                             # line=dict(color="#00CED1", width=3), name='Perturbation'))

    return fig


@app.callback(Output('scatterplot', 'figure'),
              [Input('t1-slider', 'value'),
               Input('t2-slider', 'value'),
               Input('t3-slider', 'value'),
               Input('t4-slider', 'value'),
               Input('M22-slider', 'value')])
def update_scatterplot(t1, t2, t3, t4, M22):
    fig_scatterplot = go.Figure(
        data=go.Splom(dimensions=[dict(label='t1', values=df.t1), dict(label='t2', values=df.t2),
                                  dict(label='t3', values=df.t3), dict(label='t4', values=df.t4),
                                  dict(label='s2', values=df.s2)],
                      showupperhalf=False, diagonal_visible=False,  # remove plots on diagonal
                      text=df['label'],
                      marker=dict(color=df['colors'], size=2)))

    fig_scatterplot.update_layout(
        width=1000,
        height=1000,
    )
    fig_scatterplot.add_trace(go.Splom(dimensions=[dict(label=r'$t_1$', values=[0]), dict(label='t2', values=[0]),
                                                   dict(label='t3', values=[0]), dict(label='t4', values=[0]),
                                                   dict(label=r'$s_2$', values=[pga.M_mean[1, 1]])],
                                       showupperhalf=False, diagonal_visible=False,  # remove plots on diagonal
                                       marker=dict(color="#FF0000", size=10), name='Karcher mean'))

    fig_scatterplot.add_trace(go.Splom(dimensions=[dict(label=r'$t_1$', values=[t1]), dict(label='t2', values=[t2]),
                                                   dict(label='t3', values=[t3]), dict(label='t4', values=[t4]),
                                                   dict(label=r'$s_2$', values=[M22])],
                                       showupperhalf=False, diagonal_visible=False,  # remove plots on diagonal
                                       marker=dict(color="#000000", size=10), name='Perturbation'))
    return fig_scatterplot


if __name__ == '__main__':
    app.run_server(debug=True)