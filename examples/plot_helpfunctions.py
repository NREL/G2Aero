import plotly.graph_objects as go
import numpy as np
import pandas as pd
import seaborn as sns


def plot_3d_blade_with_nominal(shapes, shapes_nominal):
    N_eta_skp = 1
    N_arc_skp = 25
    line_b = go.scatter3d.Line(color="#1f77b4", width=1.)
    line_m = go.scatter3d.Line(color="#000000", width=4.)
    data = []

    # Plot Grassmann interpolator cross section refinements
    for i in range(0, int((shapes.shape[0])), N_eta_skp):
        data.append(go.Scatter3d(x=shapes[i, :, 2], y=shapes[i, :, 0], z=shapes[i, :, 1], mode='lines', line=line_b))

    # Plot span-wise landmarks
    for j in range(0, int((shapes.shape[1])), N_arc_skp):
        span_line = shapes[:, j]
        data.append(go.Scatter3d(x=span_line[:, 2], y=span_line[:, 0], z=span_line[:, 1],
                                 mode='lines', line=line_b))

    # Plot nominal cross sections
    for i in range(0, int(shapes_nominal.shape[0])):
        nominal_x = shapes_nominal[i, :, 0].flatten()
        nominal_y = shapes_nominal[i, :, 1].flatten()
        nominal_z = shapes_nominal[i, :, 2].flatten()
        data.append(go.Scatter3d(x=nominal_z, y=nominal_x, z=nominal_y, mode='lines', line=line_m))

    fig = go.Figure(data=data)
    fig.update_layout(scene_aspectmode='data')
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    return fig


def plot_3d_blade(shapes, nominal_shapes=False):
    N_eta_skp = 1
    N_arc_skp = 25
    line_b = go.scatter3d.Line(color="#1f77b4", width=1.)
    data = []

    # Plot Grassmann interpolator cross section refinements
    for i in range(0, int((shapes.shape[0])), N_eta_skp):
        data.append(go.Scatter3d(x=shapes[i, :, 2], y=shapes[i, :, 0], z=shapes[i, :, 1], 
                                 mode='lines', line=line_b, showlegend=False))

    # Plot span-wise landmarks
    for j in range(0, int((shapes.shape[1])), N_arc_skp):
        span_line = shapes[:, j]
        data.append(go.Scatter3d(x=span_line[:, 2], y=span_line[:, 0], z=span_line[:, 1],
                                 mode='lines', line=line_b, showlegend=False))

    if not nominal_shapes is None:
        line_m = go.scatter3d.Line(color="#000000", width=4.)
        for i in range(0, int((nominal_shapes.shape[0]))):
            data.append(go.Scatter3d(x=nominal_shapes[i, :, 2], y=nominal_shapes[i, :, 0], z=nominal_shapes[i, :, 1], 
                                     mode='lines', line=line_m, showlegend=False))

    fig = go.Figure(data=data)
    fig.update_layout(scene_aspectmode='data')
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    return fig


def scatterplot_with_blades(t, blades_t):
    coord_names = ['$t_1$', '$t_2$', '$t_3$', '$t_4$']
    df = pd.DataFrame(data=t, columns=coord_names)
    splot = sns.pairplot(df, x_vars=coord_names, y_vars=coord_names,
                     diag_kind='kde', corner=True, plot_kws=dict(alpha=.5, s=5))
    blade_line = ['-', 'dashed']
    for k, blade in enumerate(blades_t):
        splot.axes[1,0].plot(blade[:,0], blade[:,1], lw=2, color='k', ls=blade_line[k], marker="+")
        splot.axes[2,0].plot(blade[:,0], blade[:,2], lw=2, color='k', ls=blade_line[k], marker="+")
        splot.axes[3,0].plot(blade[:,0], blade[:,3], lw=2, color='k', ls=blade_line[k], marker="+")
        splot.axes[2,1].plot(blade[:,1], blade[:,2], lw=2, color='k', ls=blade_line[k], marker="+")
        splot.axes[3,1].plot(blade[:,1], blade[:,3], lw=2, color='k', ls=blade_line[k], marker="+")
        splot.axes[3,2].plot(blade[:,2], blade[:,3], lw=2, color='k', ls=blade_line[k], marker="+")
