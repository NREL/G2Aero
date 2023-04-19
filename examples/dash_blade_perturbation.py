import os
import numpy as np
import pandas as pd
from g2aero.Grassmann import landmark_affine_transform
from g2aero.PGA import Grassmann_PGAspace, SPD_TangentSpace
from g2aero.utils import remove_tailedge_gap
from g2aero.yaml_info import YamlInfo
from g2aero.Grassmann_interpolation import GrassmannInterpolator
from g2aero.transform import TransformBlade
from g2aero.utils import position_airfoil, add_tailedge_gap, check_selfintersect
from g2aero.SPD import polar_decomposition

from plot_helpfunctions import plot_3d_blade_with_nominal

import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash.dependencies import Input, Output

examples_path = os.path.dirname(__file__)
root_path = os.path.abspath(os.path.join(examples_path, os.pardir))
# Load airfoils classes
shapes_folder = os.path.join(root_path, 'data', 'airfoils')
airfoils = np.load(os.path.join(shapes_folder, 'CST_shapes_TE_gap.npz'))['classes']
colors_pallete = ['#FF00FF', '#00429d', '#2571b0', '#4a9fc3', '#6fced6', '#96ffea', '#008000', '#58ffd8',
                  '#ff9e7c', '#fa7779', '#e9546f', '#d13660', '#b51c4d']#, '#940638', '#720022']
dict_color = {airfoils[::1000][i]: colors_pallete[i] for i in range(len(airfoils[::1000]))}
############################################
# Load PGA_space (dataset with TE gap)
############################################
space_folder = os.path.join(root_path, 'data', 'pga_space')
pga = Grassmann_PGAspace.load_from_file(os.path.join(space_folder, 'CST_Gr_PGA.npz'))
# spd_tan = SPD_TangentSpace.load_from_file(os.path.join(space_folder, 'CST_SPD_tangent.npz'))

# make dataframe
df = pd.DataFrame(data=pga.t[:, :4], columns=['t1', 't2', 't3', 't4'])
df['label'] = airfoils
df['colors'] = [dict_color[airfoil] for airfoil in airfoils]

# make Karcher mean airfoil
karcher_mean = pga.PGA2shape([0, 0, 0, 0])
karcher_mean = position_airfoil(karcher_mean)

# define center and radius to define ranges of parameters on a slidebar
q = 0.95
q_min, q_max = 0.5-q/2, 0.5+q/2
centers = (np.quantile(pga.t, q_max, axis=0) + np.quantile(pga.t, q_min, axis=0))/2
q = 0.99
q_min, q_max = 0.5-q/2, 0.5+q/2
pga.radius = (np.quantile(pga.t, q_max, axis=0) - np.quantile(pga.t, q_min, axis=0))/2
tmin = centers-pga.radius
tmax = centers+pga.radius

############################################
# Load blade
############################################
blade_filename = 'IEA-15-240-RWT.yaml'
# blade_filename = 'nrel5mw_ofpolars.yaml'
blade_path = os.path.join(root_path, "data", 'blades_yamls', blade_filename)
Blade = YamlInfo.init_from_yaml(blade_path, n_landmarks=401, landmark_method='cst')

shapes_bs = Blade.xy_landmarks[2:] # given (baseline) blade airfoils
shapes_gr_bs, M_bs, b_bs = polar_decomposition(shapes_bs)
t_blade = pga.gr_shapes2PGA(shapes_gr_bs)

# Interpolate baseline blade
n_cross_sections = 100
Grassmann = GrassmannInterpolator(Blade.eta_nominal, Blade.xy_nominal)
eta_span = Grassmann.sample_eta(n_cross_sections, n_hub=10, n_tip=10, n_end=25)
Transform = TransformBlade(Blade.M_yaml_interpolator, Blade.b_yaml_interpolator, 
                           Blade.pitch_axis, Grassmann.interpolator_M, Grassmann.interpolator_b)
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
                                         marker=dict(color="#FF0000", size=8), name='Baseline blade'))

    # fig_scatterplot.add_trace(go.Splom(dimensions=[dict(label=r'$t_1$', values=IEA15_airfoils[:, 0]), dict(label='t2', values=t_blade[:, 1]),
    #                                                 dict(label='t3', values=IEA15_airfoils[:, 2]), dict(label='t4', values=IEA15_airfoils[:, 3])],
    #                                     showupperhalf=False, diagonal_visible=False,  # remove plots on diagonal
    #                                     marker=dict(color="#FF0000", size=8), name='Perturbation'))

    perturbation, _ = pga.generate_perturbed_blade(shapes_gr_bs, coef=[t1, t2, t3, t4])
    t_perturbation = pga.gr_shapes2PGA(perturbation)
    fig_scatterplot.add_trace(go.Splom(dimensions=[dict(label=r'$t_1$', values=t_perturbation[:, 0]), dict(label='t2', values=t_perturbation[:, 1]),
                                                    dict(label='t3', values=t_perturbation[:, 2]), dict(label='t4', values=t_perturbation[:, 3])],
                                        showupperhalf=False, diagonal_visible=False,  # remove plots on diagonal
                                        marker=dict(color="#000000", size=8), name='Perturbed blade'))
    return fig_scatterplot


@app.callback(Output('3d_blade', 'figure'),
            [Input('t1-slider', 'value'),
            Input('t2-slider', 'value'),
            Input('t3-slider', 'value'),
            Input('t4-slider', 'value')])
def update_3d_blade(t1, t2, t3, t4):

    perturbation, _ = pga.generate_perturbed_blade(shapes_gr_bs, coef=[t1, t2, t3, t4])

    # Inverse affine transform
    new_blade_phys = np.empty_like(perturbation)
    for i, sh in enumerate(perturbation):
        new_blade_phys[i] = sh @ M_bs[i] + b_bs[i]

    # Add circles
    new_blade_full = np.vstack((Blade.xy_landmarks[:2], new_blade_phys))

    GrInterp = GrassmannInterpolator(Blade.eta_nominal, new_blade_full)
    eta_span = GrInterp.sample_eta(100, n_hub=10, n_tip=10, n_end=25)
    _, blade_gr = GrInterp(eta_span, grassmann=True)
    Transform = TransformBlade(Blade.M_yaml_interpolator, Blade.b_yaml_interpolator,
                            Blade.pitch_axis, GrInterp.interpolator_M, GrInterp.interpolator_b)
    xyz_local = Transform.grassmann_to_phys(blade_gr, eta_span)
    xyz_nominal = Transform.grassmann_to_phys(GrInterp.xy_grassmann, GrInterp.eta_nominal)
    fig_3d_blade = plot_3d_blade_with_nominal(xyz_local, xyz_nominal)
    return fig_3d_blade


if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1')