import os
import numpy as np
import pandas as pd
from g2aero.PGA import Grassmann_PGAspace, SPD_TangentSpace
from g2aero.utils import position_airfoil, add_tailedge_gap, check_selfintersect

import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash.dependencies import Input, Output

examples_path = os.path.dirname(__file__)
root_path = os.path.abspath(os.path.join(examples_path, os.pardir))
space_folder = os.path.join(root_path, 'data', 'pga_space')

shapes_folder  = os.path.join(root_path, 'data', 'airfoils', )
airfoils = np.load(os.path.join(shapes_folder, 'CST_shapes_TE_gap.npz'))['classes']

colors_pallete = ['#FF00FF', '#00429d', '#2571b0', '#4a9fc3', '#6fced6', '#96ffea', '#008000', '#58ffd8',
                  '#ff9e7c', '#fa7779', '#e9546f', '#d13660', '#b51c4d']#, '#940638', '#720022']

dict_color = {airfoils[::1000][i]: colors_pallete[i] for i in range(len(airfoils[::1000]))}



pga = Grassmann_PGAspace.load_from_file(os.path.join(space_folder, 'CST_Gr_PGA.npz'))
t = pga.t

spd_tan = SPD_TangentSpace.load_from_file(os.path.join(space_folder, 'CST_SPD_tangent.npz'))


a_pga, b_pga = np.min(t, axis=0), np.max(t, axis=0)
c_pga = (b_pga + a_pga)/2
pga.radius = (b_pga-c_pga)*1.1
centers =[0, 0, 0, 0]
centers = np.array(centers)

df = pd.DataFrame(data=pga.t[:, :4], columns=['t1', 't2', 't3', 't4'])
df['label'] = airfoils
df['colors'] = [dict_color[airfoil] for airfoil in airfoils]

karcher_mean = pga.PGA2shape([0, 0, 0, 0])
karcher_mean = position_airfoil(karcher_mean)
q = 0.95
q_min, q_max = 0.5-q/2, 0.5+q/2
centers = (np.quantile(t, q_max, axis=0) + np.quantile(t, q_min, axis=0))/2
q = 0.99
q_min, q_max = 0.5-q/2, 0.5+q/2
pga.radius = (np.quantile(t, q_max, axis=0) - np.quantile(t, q_min, axis=0))/2
tmin = centers-pga.radius
tmax = centers+pga.radius
#######################################################################################
### LAYOUT
#######################################################################################
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


@app.callback(Output('airfoils', 'figure'),
              [Input('t1-slider', 'value'),
               Input('t2-slider', 'value'),
               Input('t3-slider', 'value'),
               Input('t4-slider', 'value')])
def calculate_shape(t1, t2, t3, t4):

    M = pga.M_mean.copy()
    # M[1, 1] = M22
    shape = pga.PGA2shape([t1, t2, t3, t4], M=M)
    shape = position_airfoil(shape)

    fig = go.Figure()
    fig.update_xaxes(range=[-0.01, 1.01], scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-0.3, 0.3], scaleanchor="x", scaleratio=1)
    fig.add_trace(go.Scatter(x=karcher_mean[:, 0], y=karcher_mean[:, 1],
                             line=dict(color="#FF0000", width=2), name='Karcher mean'))
    if not check_selfintersect(shape):
        fig.add_trace(go.Scatter(x=shape[:, 0], y=shape[:, 1],
                             line=dict(color="#000000", width=3), name='Perturbation'))
                             # line=dict(color="#00CED1", width=3), name='Perturbation'))
    else:
        fig.add_trace(go.Scatter(x=shape[:, 0], y=shape[:, 1],
                                 line=dict(color="#FFA500", width=3), name='shape selfintersects'))
    return fig


@app.callback(Output('scatterplot', 'figure'),
              [Input('t1-slider', 'value'),
               Input('t2-slider', 'value'),
               Input('t3-slider', 'value'),
               Input('t4-slider', 'value')])
def update_scatterplot(t1, t2, t3, t4):

    M = pga.M_mean.copy()
    # M[1, 1] = M22
    shape = pga.PGA2shape([t1, t2, t3, t4], M=M)
    shape = position_airfoil(shape)
    # shape = add_tailedge_gap(shape, gap)
    
    fig_scatterplot = go.Figure(
        data=go.Splom(dimensions=[dict(label='t1', values=df.t1), dict(label='t2', values=df.t2),
                                  dict(label='t3', values=df.t3), dict(label='t4', values=df.t4)],
                      showupperhalf=False, diagonal_visible=False,  # remove plots on diagonal
                      text=df['label'],
                      marker=dict(color=df['colors'], size=2)))

    fig_scatterplot.update_layout(
        width=1000,
        height=1000,
    )

    fig_scatterplot.add_trace(go.Splom(dimensions=[dict(label=r'$t_1$', values=[0]), dict(label='t2', values=[0]),
                                                   dict(label='t3', values=[0]), dict(label='t4', values=[0])],
                                       showupperhalf=False, diagonal_visible=False,  # remove plots on diagonal
                                       marker=dict(color="#FF0000", size=10), name='Karcher mean'))

    if not check_selfintersect(shape):
        fig_scatterplot.add_trace(go.Splom(dimensions=[dict(label=r'$t_1$', values=[t1]), dict(label='t2', values=[t2]),
                                                       dict(label='t3', values=[t3]), dict(label='t4', values=[t4])],
                                           showupperhalf=False, diagonal_visible=False,  # remove plots on diagonal
                                           marker=dict(color="#000000", size=10), name='Perturbation'))
    else:
        fig_scatterplot.add_trace(go.Splom(dimensions=[dict(label=r'$t_1$', values=[t1]), dict(label='t2', values=[t2]),
                                                       dict(label='t3', values=[t3]), dict(label='t4', values=[t4])],
                                           showupperhalf=False, diagonal_visible=False,  # remove plots on diagonal
                                           marker=dict(color="#FFA500", size=10), name='outside sampling space'))

    return fig_scatterplot


if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1')