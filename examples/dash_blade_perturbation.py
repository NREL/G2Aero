import os
import numpy as np
import pandas as pd
from g2aero.Grassmann import landmark_affine_transform
from g2aero.perturbation import PGAspace
from g2aero.utils import remove_tailedge_gap
from g2aero import yaml_info
from g2aero.Grassmann_interpolation import GrassmannInterpolator
from g2aero.transform import TransformBlade

from plot_helpfunctions import plot_3d_blade_with_nominal

import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash.dependencies import Input, Output


############################################
# Load PGA_space
############################################
pga_dict = np.load(os.path.join(os.getcwd(), '../data', 'PGA_space', 'PGA_space.npz'))
pga = PGAspace(pga_dict['Vh'], pga_dict['M_mean'], pga_dict['b_mean'], pga_dict['karcher_mean'])
pga.M_mean = pga.M_mean.T
t = np.load(os.path.join(os.getcwd(), '../data', 'PGA_space', 't.npz'))['t']


r_min = np.abs(np.quantile(t, 0.0, axis=0))
r_max = np.abs(np.quantile(t, 0.0, axis=0))
pga.radius = np.max(np.array([r_min, r_max]), axis=0)
centers = np.array([0, 0, 0, 0])

airfoils = ['NACA64_A17', 'DU21_A17', 'DU25_A17', 'DU30_A17', 'DU35_A17', 'DU40_A17', 'DU00-W2-350', 'DU08-W-210',
            'FFA-W3-211',  'FFA-W3-241', 'FFA-W3-270blend', 'FFA-W3-301', 'FFA-W3-330blend', 'FFA-W3-360', 'SNL-FFA-W3-500']

n_airfoils = len(airfoils)
label_dict = dict(zip(airfoils, np.arange(len(airfoils))))

colors_pallete = ['#FF00FF', '#00429d', '#2571b0', '#4a9fc3', '#6fced6', '#96ffea', '#008000', '#58ffd8',
                  '#ff9e7c', '#fa7779', '#e9546f', '#d13660', '#b51c4d', '#940638', '#720022']
dict_color = {airfoils[i]: colors_pallete[i] for i in range(len(airfoils))}

df = pd.DataFrame(data=t, columns=['t1', 't2', 't3', 't4'])
df['label'] = [airfoil for airfoil in airfoils for i in range(1001)]
df['colors'] = [dict_color[airfoil] for airfoil in airfoils for i in range(1001)]

tmin = -pga.radius
tmax = pga.radius
############################################
# Load blade
############################################
blade_filename = 'IEA-15-240-RWT.yaml'
# blade_filename = 'nrel5mw_ofpolars.yaml'

shapes_path = os.path.join(os.getcwd(), "../data", 'blades_yamls', blade_filename)
n_cross_sections = 100

Blade = yaml_info.YamlInfo(shapes_path, n_landmarks=401)

nogap_shapes = np.empty_like(Blade.xy_landmarks)
for i, xy in enumerate(Blade.xy_landmarks):
    nogap_shapes[i] = remove_tailedge_gap(xy)
Blade.xy_nominal = Blade.shift_to_pitch_axis(Blade.eta_nominal)

Grassmann = GrassmannInterpolator(Blade.eta_nominal, Blade.xy_nominal)

eta_span = Grassmann.sample_eta(100, n_hub=10, n_tip=10, n_end=25)
Transform = TransformBlade(Blade.M_yaml_interpolator, Blade.b_yaml_interpolator, 
                           Blade.pitch_axis, Grassmann.interpolator_M, Grassmann.interpolator_b)

curve_gr_shapes, _, _ = landmark_affine_transform(nogap_shapes[2:-1])
t_blade = pga.gr_shapes2PGA(curve_gr_shapes)
#################################
# App
#################################
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SKETCHY])

app.layout = html.Div([
    dbc.Row(dbc.Col(html.Div(html.H1(children='Blade perturbations', style={'textAlign': 'center'})))),
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
                dcc.Graph(id='scatterplot'),
            ]), width=6, align="center")
    ]),
    dbc.Row(dbc.Col(html.Div([
                dcc.Graph(id='3d_blade'),
            ]), width=12, align="center")),
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


@app.callback(Output('scatterplot', 'figure'),
              [Input('t1-slider', 'value'),
               Input('t2-slider', 'value'),
               Input('t3-slider', 'value'),
               Input('t4-slider', 'value')])
def update_scatterplot(t1, t2, t3, t4):

    # Scatterplot of cst dataset
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
    # Add karcher mean as read dot
    fig_scatterplot.add_trace(go.Splom(dimensions=[dict(label=r'$t_1$', values=[0]), dict(label='t2', values=[0]),
                                                   dict(label='t3', values=[0]), dict(label='t4', values=[0])],
                                       showupperhalf=False, diagonal_visible=False,  # remove plots on diagonal
                                       marker=dict(color="#FF0000", size=10), name='Karcher mean'))

    # Add perturbed point
    fig_scatterplot.add_trace(go.Splom(dimensions=[dict(label=r'$t_1$', values=[t1]), dict(label='t2', values=[t2]),
                                                    dict(label='t3', values=[t3]), dict(label='t4', values=[t4])],
                                        showupperhalf=False, diagonal_visible=False,  # remove plots on diagonal
                                        marker=dict(color="#000000", size=10), name='Perturbation'))

    # Add baseline blade

    fig_scatterplot.add_trace(go.Splom(dimensions=[dict(label=r'$t_1$', values=t_blade[:, 0]), dict(label='t2', values=t_blade[:, 1]),
                                                    dict(label='t3', values=t_blade[:, 2]), dict(label='t4', values=t_blade[:, 3])],
                                        showupperhalf=False, diagonal_visible=False,  # remove plots on diagonal
                                         marker=dict(color="#000000", size=8), name='Baseline blade'))

    # fig_scatterplot.add_trace(go.Splom(dimensions=[dict(label=r'$t_1$', values=IEA15_airfoils[:, 0]), dict(label='t2', values=t_blade[:, 1]),
    #                                                 dict(label='t3', values=IEA15_airfoils[:, 2]), dict(label='t4', values=IEA15_airfoils[:, 3])],
    #                                     showupperhalf=False, diagonal_visible=False,  # remove plots on diagonal
    #                                     marker=dict(color="#FF0000", size=8), name='Perturbation'))


    perturbation, coef = pga.generate_perturbed_blade(curve_gr_shapes, coef=[t1, t2, t3, t4])
    t_perturbation = pga.gr_shapes2PGA(perturbation)
    fig_scatterplot.add_trace(go.Splom(dimensions=[dict(label=r'$t_1$', values=t_perturbation[:, 0]), dict(label='t2', values=t_perturbation[:, 1]),
                                                    dict(label='t3', values=t_perturbation[:, 2]), dict(label='t4', values=t_perturbation[:, 3])],
                                        showupperhalf=False, diagonal_visible=False,  # remove plots on diagonal
                                        marker=dict(color="#FF0000", size=8), name='Perturbed blade'))
    return fig_scatterplot


@app.callback(Output('3d_blade', 'figure'),
            [Input('t1-slider', 'value'),
            Input('t2-slider', 'value'),
            Input('t3-slider', 'value'),
            Input('t4-slider', 'value')])
def update_3d_blade(t1, t2, t3, t4):

    perturbation, _ = pga.generate_perturbed_blade(curve_gr_shapes, coef=[t1, t2, t3, t4])
    print(perturbation.shape)
    perturbation = np.vstack((perturbation, perturbation[-1:]))
    print(perturbation.shape)

    Grassmann.shapes_perturbation(perturbation, np.arange(2, len(perturbation)+2))
    _, new_blade = Grassmann(eta_span, grassmann=True)
    xyz_local = Transform.grassmann_to_phys(new_blade, eta_span)
    xyz_nominal = Transform.grassmann_to_phys(Grassmann.xy_grassmann, Grassmann.eta_nominal)

    fig_3d_blade = plot_3d_blade_with_nominal(xyz_local, xyz_nominal)
    return fig_3d_blade


if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1')