import plotly.graph_objects as go
import numpy as np



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


def plot_3d_blade(shapes):
    N_eta_skp = 1
    N_arc_skp = 25
    line_b = go.scatter3d.Line(color="#1f77b4", width=1.)
    data = []

    # Plot Grassmann interpolator cross section refinements
    for i in range(0, int((shapes.shape[0])), N_eta_skp):
        data.append(go.Scatter3d(x=shapes[i, :, 2], y=shapes[i, :, 0], z=shapes[i, :, 1], mode='lines', line=line_b))

    # Plot span-wise landmarks
    for j in range(0, int((shapes.shape[1])), N_arc_skp):
        span_line = shapes[:, j]
        data.append(go.Scatter3d(x=span_line[:, 2], y=span_line[:, 0], z=span_line[:, 1],
                                 mode='lines', line=line_b))

    fig = go.Figure(data=data)
    fig.update_layout(scene_aspectmode='data')
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    return fig