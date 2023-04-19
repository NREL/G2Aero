import os
import numpy as np
import pandas as pd
from g2aero.PGA import Grassmann_PGAspace
from g2aero.utils import position_airfoil, add_tailedge_gap, check_selfintersect

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash.dependencies import Input, Output

examples_path = os.path.dirname(__file__)
root_path = os.path.abspath(os.path.join(examples_path, os.pardir))

space_folder = os.path.join(root_path, 'data', 'pga_space', 'full_space')
# shapes_folder = os.path.join(root_path, 'data', 'airfoils')

pga_dict = np.load(os.path.join(space_folder, 'PGA_space.npz'))
t = np.load(os.path.join(space_folder, 't.npz'))['t']
pga = Grassmann_PGAspace(pga_dict['Vh'], pga_dict['M_mean'], pga_dict['b_mean'], pga_dict['karcher_mean'], t)
M = np.load(os.path.join(space_folder, 'M_b.npz'))['M']
Mmin = np.min(M, axis=0).reshape(4,)
Mmax = np.max(M, axis=0).reshape(4,)
# b = np.load(os.path.join(space_folder, 'M_b.npz'))['b']
t = np.load(os.path.join(space_folder, 't.npz'))['t']
labels = np.load(os.path.join(space_folder, 't.npz'))['labels']
print(labels.shape)

q = 0.95
q_min, q_max = 0.5-q/2, 0.5+q/2
centers = (np.quantile(t, q_max, axis=0) + np.quantile(t, q_min, axis=0))/2
q = 0.99
q_min, q_max = 0.5-q/2, 0.5+q/2
pga.radius = (np.quantile(t, q_max, axis=0) - np.quantile(t, q_min, axis=0))/2

df = pd.DataFrame(data=t[:, :4], columns=['t1', 't2', 't3', 't4'])
df['label'] = labels
karcher_mean = pga.PGA2shape([0, 0, 0, 0])

# karcher_mean = position_airfoil(karcher_mean)
# karcher_mean = add_tailedge_gap(karcher_mean, 0.01)
tmin = centers-pga.radius
tmax = centers+pga.radius



app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.SKETCHY],
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}])

app.layout = html.Div([

    dbc.Row(dbc.Col(html.Div(html.H1(children='Database', style={'textAlign': 'center'})))),
    dbc.Row([
        dbc.Col(
            html.Div([
                html.H2(children='Airfoils from database', style={'textAlign': 'center'}),
                dcc.Dropdown(options=[f'{i}' for i in range(len(t))], value='0', id='dropdown-id'),
                html.Div(id='airfoil-name'),
                dcc.Graph(id='database-airfoils'),
            ]), width={"size": 8, "offset": 1 },  style={'textAlign': 'center'})
    ]),
    dbc.Row([
        dbc.Col(
            html.Div([
                html.H2(children='PGA space', style={'textAlign': 'center'}),
                dcc.Graph(id='scatterplot'),
            ]), width={"size": 5, "offset": 1}),
    ]),
    dbc.Row(dbc.Col(html.Div(html.H1(children='Airfoil perturbations', style={'textAlign': 'center'})))),
    dbc.Row([
        dbc.Col(
            html.Div([
                html.H2(children='PGA parameters',
                        style={'textAlign': 'center'}),

                html.Label(id='t1-update'),
                dcc.Slider(id='t1-slider', min=tmin[0], max=tmax[0], step=0.0001,
                           marks={i: f'{i:.2f}' for i in np.arange(np.round(tmin[0], 2), np.round(tmax[0], 2), 0.25)},
                           value=0.0),
                html.Br(),

                html.Label(id='t2-update'),
                dcc.Slider(id='t2-slider',  min=tmin[1], max=tmax[1], step=0.0001,
                           marks={i: f'{i:.2f}' for i in np.arange(np.round(tmin[0], 2), np.round(tmax[0], 2), 0.1)},
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

                html.Label(id='P11-update'),
                dcc.Slider(id='P11-slider',  min=Mmin[0], max=Mmax[0], step=0.001,
                           marks={i: f'{i:.2f}' for i in np.arange(np.round(Mmin[0], 2), np.round(Mmax[0], 2), 0.5)},
                           value=np.round(pga.M_mean[0, 0], 4)),
                html.Br(),

                html.Label(id='P12-update'),
                dcc.Slider(id='P12-slider',  min=Mmin[1], max=Mmax[1], step=0.001,
                           marks={i: f'{i:.2f}' for i in np.arange(np.round(Mmin[1], 2), np.round(Mmax[1], 2), 0.5)},
                           value=np.round(pga.M_mean[0, 0], 4)),
                html.Br(),

                html.Label(id='P22-update'),
                dcc.Slider(id='P22-slider',  min=Mmin[3], max=Mmax[3], step=0.001,
                           marks={i: f'{i:.2f}' for i in np.arange(np.round(Mmin[3], 2), np.round(Mmax[3], 2), 0.5)},
                           value=np.round(pga.M_mean[0, 0], 4)),
                html.Br(),



                # html.H2(children='Tailedge gap',
                #         style={'textAlign': 'center'}),
                # html.Label(id='gap-update'),
                # dcc.Slider(id='gap-slider',  min=0.002, max=np.round(np.max(gap), 3), step=0.0005,
                #            marks={i: f'{i:.3f}' for i in np.arange(0.002, np.round(np.max(gap), 3), 0.002)},
                #            value=0.01),
                # html.Br(),

            ]), width={"size": 4, "offset": 1}),
        dbc.Col(
            html.Div([
                dcc.Graph(id='airfoils'),
            ]), width=6, align="center")
    ])
])

@app.callback(Output('airfoil-name', 'children'), Input('dropdown-id', 'value'))
def update_output(value):
    return f'You have selected {value}th airfoil'

@app.callback(Output('database-airfoils', 'figure'), Input('dropdown-id', 'value'))
def plot_database_airfoil(i):
    i = int(i)
    shape = pga.PGA2shape(t[i], M=M[i])
    fig = go.Figure()
    fig.update_xaxes(range=[-0.01, 1.01], scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-0.3, 0.3], scaleanchor="x", scaleratio=1)

    fig.add_trace(go.Scatter(x=shape[:, 0], y=shape[:, 1], line=dict(color="#000000", width=3)))
    fig.update_layout(legend=dict(x=0.,y=1.))
    # fig.update_layout(
    #     width=600,
    #     height=400,
    # )
    return fig


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


@app.callback(Output('P11-update', 'children'), [Input('P11-slider', 'value')])
def display_value(value):
    return f'P11 = {value}'

@app.callback(Output('P12-update', 'children'), [Input('P12-slider', 'value')])
def display_value(value):
    return f'P12 = {value}'

@app.callback(Output('P22-update', 'children'), [Input('P22-slider', 'value')])
def display_value(value):
    return f'P22 = {value}'


# @app.callback(Output('gap-update', 'children'), [Input('gap-slider', 'value')])
# def display_value(value):
#     return f'gap = {value}'


@app.callback(Output('airfoils', 'figure'),
              [Input('t1-slider', 'value'),
               Input('t2-slider', 'value'),
               Input('t3-slider', 'value'),
               Input('t4-slider', 'value'),
            #    Input('M22-slider', 'value'),
            #    Input('gap-slider', 'value')
            ])
def calculate_shape(t1, t2, t3, t4):

    M = pga.M_mean.copy()
    # M[1, 1] = M22
    shape = pga.PGA2shape([t1, t2, t3, t4], M=M)
    # shape = position_airfoil(shape)
    # shape = add_tailedge_gap(shape, gap)

    fig = go.Figure()
    fig.update_xaxes(range=[-0.01, 1.01], scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-0.3, 0.3], scaleanchor="x", scaleratio=1)
    fig.add_trace(go.Scatter(x=karcher_mean[:, 0], y=karcher_mean[:, 1],
                             line=dict(color="#FF0000", width=2), name='Karcher mean'))
    # if gap < (m*M22+l) and np.sum((np.array([t1, t2, t3, t4, M22]) - centers)**2/ pga.radius ** 2) <= 1:
    if not check_selfintersect(shape):
        fig.add_trace(go.Scatter(x=shape[:, 0], y=shape[:, 1],
                             line=dict(color="#000000", width=3), name='Perturbation'))
                             # line=dict(color="#00CED1", width=3), name='Perturbation'))
    else:
        fig.add_trace(go.Scatter(x=shape[:, 0], y=shape[:, 1],
                                 line=dict(color="#FFA500", width=3), name='shape selfintersects'))
    fig.update_layout(legend=dict(x=0.,y=1.))
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
    # shape = position_airfoil(shape)
    # shape = add_tailedge_gap(shape, gap)
    
    fig_scatterplot = go.Figure(
        data=go.Splom(dimensions=[dict(label='t1', values=df.t1), dict(label='t2', values=df.t2),
                                  dict(label='t3', values=df.t3), dict(label='t4', values=df.t4)],
                      showupperhalf=False, diagonal_visible=False,  # remove plots on diagonal
                    #   text=f'{df.index}, {df.label}',
                    #   text=df.index,
                      text=df['label'],
                      showlegend=False,
                      marker=dict(size=5)))

    fig_scatterplot.update_layout(
        width=600,
        height=600,
    )

    fig_scatterplot.add_trace(go.Splom(dimensions=[dict(label='t1', values=[0]), dict(label='t2', values=[0]),
                                                   dict(label='t3', values=[0]), dict(label='t4', values=[0]),
                                                #    dict(label=r'$s_2$', values=[pga.M_mean[1, 1]]),
                                                #    dict(label='gap', values=[0.01])
                                                   ],
                                       showupperhalf=False, diagonal_visible=False,  # remove plots on diagonal
                                       marker=dict(color="#FF0000", size=10), name='Karcher mean'))

    # if gap < (m*M22+l) and np.sum((np.array([t1, t2, t3, t4, M22]) - centers)**2/ pga.radius ** 2) <= 1:
    if not check_selfintersect(shape):
        fig_scatterplot.add_trace(go.Splom(dimensions=[dict(label='t1', values=[t1]), dict(label='t2', values=[t2]),
                                                       dict(label='t3', values=[t3]), dict(label='t4', values=[t4])],
                                           showupperhalf=False, diagonal_visible=False,  # remove plots on diagonal
                                           marker=dict(color="#000000", size=10), name='Perturbation'))
    else:
        fig_scatterplot.add_trace(go.Splom(dimensions=[dict(label='t1', values=[t1]), dict(label='t2', values=[t2]),
                                                       dict(label='t3', values=[t3]), dict(label='t4', values=[t4])],
                                           showupperhalf=False, diagonal_visible=False,  # remove plots on diagonal
                                           marker=dict(color="#FFA500", size=10), name='shape selfintersects'))

    fig_scatterplot.update_layout(legend=dict(x=0.7,y=1.))


    return fig_scatterplot


if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1')