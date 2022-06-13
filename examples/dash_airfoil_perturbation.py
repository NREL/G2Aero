import os
import numpy as np
import pandas as pd
from g2aero.perturbation import PGAspace
from g2aero.utils import position_airfoil, add_tailedge_gap

import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash.dependencies import Input, Output

pga_dict = np.load(os.path.join(os.getcwd(), '../data', 'PGA_space', 'PGA_space.npz'))
pga = PGAspace(pga_dict['Vh'], pga_dict['M_mean'], pga_dict['b_mean'], pga_dict['karcher_mean'])
pga.M_mean = pga.M_mean.T
s2 = np.load(os.path.join(os.getcwd(), '../data', 'PGA_space', 'M_b.npz'))['M'][:, 1:, 1]
b = np.load(os.path.join(os.getcwd(), '../data', 'PGA_space', 'M_b.npz'))['b']
t = np.load(os.path.join(os.getcwd(), '../data', 'PGA_space', 't.npz'))['t']


r_min = np.abs(np.quantile(t, 0.0, axis=0))
r_max = np.abs(np.quantile(t, 0.0, axis=0))
pga.radius = np.max(np.array([r_min, r_max]), axis=0).tolist()
t = np.hstack((t, s2))
pga.radius.append((np.max(s2[:-1001]) - np.min(s2[:-1001]))/2)
centers = [0, 0, 0, 0, (np.max(s2[:-1001]) + np.min(s2[:-1001]))/2]
centers = np.array(centers)
pga.radius = np.array(pga.radius)

x = [1.55359315, 1.74616371]
y = [0.0182, 0.02408]
m, l = np.polyfit(x, y, 1)

airfoils = ['NACA64_A17', 'DU21_A17', 'DU25_A17', 'DU30_A17', 'DU35_A17', 'DU40_A17', 'DU00-W2-350', 'DU08-W-210',
            'FFA-W3-211',  'FFA-W3-241', 'FFA-W3-270blend', 'FFA-W3-301', 'FFA-W3-330blend', 'FFA-W3-360', 'SNL-FFA-W3-500']

colors_pallete = ['#FF00FF', '#00429d', '#2571b0', '#4a9fc3', '#6fced6', '#96ffea', '#008000', '#58ffd8',
                  '#ff9e7c', '#fa7779', '#e9546f', '#d13660', '#b51c4d', '#940638', '#720022']
dict_color = {airfoils[i]: colors_pallete[i] for i in range(len(airfoils))}

gap_size = [0.0, 0.003878, 0.004262, 0.00492, 0.00566, 0.00694,  0.01, 0.00278,
            0.00131, 0.00751, 0.01265, 0.0182, 0.024079999999999997, 0.01368, 0.020200000000000003]
gap = np.array([g for g in gap_size for i in range(1001)])
t = np.hstack((t, gap.reshape(-1, 1)))
df = pd.DataFrame(data=t, columns=['t1', 't2', 't3', 't4', 's2', 'gap'])
df['label'] = [airfoil for airfoil in airfoils for i in range(1001)]
df['colors'] = [dict_color[airfoil] for airfoil in airfoils for i in range(1001)]

karcher_mean = pga.PGA2shape([0, 0, 0, 0])
karcher_mean = position_airfoil(karcher_mean)
karcher_mean = add_tailedge_gap(karcher_mean, 0.01)
tmin = -pga.radius
tmin[-1] = np.min(s2)
tmax = pga.radius
tmax[-1] = np.max(s2)

radius = np.max(np.array([r_min, r_max]), axis=0).tolist()


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
                html.H2(children='Tailedge gap',
                        style={'textAlign': 'center'}),
                html.Label(id='gap-update'),
                dcc.Slider(id='gap-slider',  min=0.002, max=np.round(np.max(gap), 3), step=0.0005,
                           marks={i: f'{i:.3f}' for i in np.arange(0.002, np.round(np.max(gap), 3), 0.002)},
                           value=0.01),
                html.Br(),

            ]), width={"size": 4, "offset": 1}),
        dbc.Col(
            html.Div([
                dcc.Graph(id='airfoils'),
                html.Br(),
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
    return f't4 = {value}'


@app.callback(Output('M22-update', 'children'), [Input('M22-slider', 'value')])
def display_value(value):
    return f's2 = {value}'


@app.callback(Output('gap-update', 'children'), [Input('gap-slider', 'value')])
def display_value(value):
    return f'gap = {value}'


@app.callback(Output('airfoils', 'figure'),
              [Input('t1-slider', 'value'),
               Input('t2-slider', 'value'),
               Input('t3-slider', 'value'),
               Input('t4-slider', 'value'),
               Input('M22-slider', 'value'),
               Input('gap-slider', 'value')])
def calculate_shape(t1, t2, t3, t4, M22, gap):

    M = pga.M_mean.copy()
    M[1, 1] = M22
    shape = pga.PGA2shape([t1, t2, t3, t4], M=M)
    shape = position_airfoil(shape)
    shape = add_tailedge_gap(shape, gap)

    fig = go.Figure()
    fig.update_xaxes(range=[-0.01, 1.01], scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-0.3, 0.3], scaleanchor="x", scaleratio=1)
    fig.add_trace(go.Scatter(x=karcher_mean[:, 0], y=karcher_mean[:, 1],
                             line=dict(color="#FF0000", width=2), name='Karcher mean'))
    if gap < (m*M22+l) and np.sum((np.array([t1, t2, t3, t4, M22]) - centers)**2/ pga.radius ** 2) <= 1:
        fig.add_trace(go.Scatter(x=shape[:, 0], y=shape[:, 1],
                             line=dict(color="#000000", width=3), name='Perturbation'))
                             # line=dict(color="#00CED1", width=3), name='Perturbation'))
    else:
        fig.add_trace(go.Scatter(x=shape[:, 0], y=shape[:, 1],
                                 line=dict(color="#FFA500", width=3), name='outside sampling space'))
    return fig


@app.callback(Output('scatterplot', 'figure'),
              [Input('t1-slider', 'value'),
               Input('t2-slider', 'value'),
               Input('t3-slider', 'value'),
               Input('t4-slider', 'value'),
               Input('M22-slider', 'value'),
               Input('gap-slider', 'value')])
def update_scatterplot(t1, t2, t3, t4, M22, gap):
    fig_scatterplot = go.Figure(
        data=go.Splom(dimensions=[dict(label='t1', values=df.t1), dict(label='t2', values=df.t2),
                                  dict(label='t3', values=df.t3), dict(label='t4', values=df.t4),
                                  dict(label='s2', values=df.s2), dict(label='gap', values=df.gap)],
                      showupperhalf=False, diagonal_visible=False,  # remove plots on diagonal
                      text=df['label'],
                      marker=dict(color=df['colors'], size=2)))

    fig_scatterplot.update_layout(
        width=1000,
        height=1000,
    )

    fig_scatterplot.add_trace(go.Splom(dimensions=[dict(label=r'$t_1$', values=[0]), dict(label='t2', values=[0]),
                                                   dict(label='t3', values=[0]), dict(label='t4', values=[0]),
                                                   dict(label=r'$s_2$', values=[pga.M_mean[1, 1]]),
                                                   dict(label='gap', values=[0.01])],
                                       showupperhalf=False, diagonal_visible=False,  # remove plots on diagonal
                                       marker=dict(color="#FF0000", size=10), name='Karcher mean'))

    if gap < (m*M22+l) and np.sum((np.array([t1, t2, t3, t4, M22]) - centers)**2/ pga.radius ** 2) <= 1:

        fig_scatterplot.add_trace(go.Splom(dimensions=[dict(label=r'$t_1$', values=[t1]), dict(label='t2', values=[t2]),
                                                       dict(label='t3', values=[t3]), dict(label='t4', values=[t4]),
                                                       dict(label=r'$s_2$', values=[M22]), dict(label='gap', values=[gap])],
                                           showupperhalf=False, diagonal_visible=False,  # remove plots on diagonal
                                           marker=dict(color="#000000", size=10), name='Perturbation'))
    else:
        fig_scatterplot.add_trace(go.Splom(dimensions=[dict(label=r'$t_1$', values=[t1]), dict(label='t2', values=[t2]),
                                                       dict(label='t3', values=[t3]), dict(label='t4', values=[t4]),
                                                       dict(label=r'$s_2$', values=[M22]),
                                                       dict(label='gap', values=[gap])],
                                           showupperhalf=False, diagonal_visible=False,  # remove plots on diagonal
                                           marker=dict(color="#FFA500", size=10), name='outside sampling space'))




    return fig_scatterplot


if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1')