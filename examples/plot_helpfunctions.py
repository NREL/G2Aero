import plotly.graph_objects as go
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.lines import Line2D
import matplotlib.animation as animation
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['animation.embed_limit'] = 2**128


def plot_3d_blade_grassmann(shapes, eta_span, shapes_nominal, eta_nominal, scale=1):
    N_eta_skp = 1
    N_arc_skp = 25
    line_b = go.scatter3d.Line(color="#1f77b4", width=1.)
    line_m = go.scatter3d.Line(color="#000000", width=4.)
    data = []

    # Plot Grassmann interpolator cross section refinements
    for i in range(0, shapes.shape[0], N_eta_skp):
        eta_span_z = np.repeat(eta_span[i], shapes.shape[1]) * scale
        data.append(go.Scatter3d(x=eta_span_z, y=shapes[i, :, 0], z=shapes[i, :, 1], 
                                 mode='lines', line=line_b, showlegend=False))

    # # Plot tip
    # i = shapes.shape[0] - 1
    # eta_span_z = np.repeat(eta_span[i], shapes.shape[1]) * scale
    # data.append(go.Scatter3d(x=eta_span_z, y=shapes[i, :, 0], z=shapes[i, :, 1],
    #                          mode='lines', line=line_b, showlegend=False))

    # Plot span-wise landmarks
    for j in range(0, int((shapes.shape[1])), N_arc_skp):
        span_line = shapes[:, j, :2]
        data.append(go.Scatter3d(x=eta_span * scale, y=span_line[:, 0], z=span_line[:, 1],
                                 mode='lines', line=line_b, showlegend=False))

    # Plot nominal cross sections
    for i in range(0, int(shapes_nominal.shape[0])):
        eta = np.repeat(eta_nominal[i], shapes_nominal.shape[1]) * scale
        nominal_x = shapes_nominal[i, :, 0].flatten()
        nominal_y = shapes_nominal[i, :, 1].flatten()
        data.append(go.Scatter3d(x=eta, y=nominal_x, z=nominal_y, 
                                 mode='lines', line=line_m, showlegend=False))

    fig = go.Figure(data=data)
    fig.update_layout(scene_aspectmode='data')
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    return fig


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




class SubplotAnimation(animation.TimedAnimation):
    def __init__(self, grassmann, eta_span_scaled, phys_nominal, gr_nominal, geodesic_gr, geodesic_phys, delay=150):

        font = {'family': 'times',
                'weight': 'bold',
                'size': 22}

        mpl.rc('font', **font)
        self.n = geodesic_gr.shape[0]
        self.n_shapes = phys_nominal.shape[0]
        self.geodesic_gr = geodesic_gr
        self.geodesic_phys = geodesic_phys
        self.x_phys = phys_nominal
        self.x_gr = gr_nominal
        self.eta_nominal = grassmann.eta_nominal_scaled
        self.eta_span = eta_span_scaled
        self.t_nominal = grassmann.t_nominal
        self.t_span = grassmann.interpolator_cdf(eta_span_scaled)

        gs_kw = dict(width_ratios=[1.5, 1, 1])
        fig, ax = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True, gridspec_kw=gs_kw)

        # Distances
        ax[0].axis('off')
        ax[0].set_aspect(0.25)
        ax[0].set_title('distance')
        ax[0].axis(ymin=-1, ymax=2)

        ax[0].add_artist(Line2D(np.linspace(self.eta_nominal[0], self.eta_nominal[-1], 1000), np.ones(1000), color='b'))
        ax[0].add_artist(Line2D(np.linspace(0, 1, 1000), np.zeros(1000), color='b'))
        ax[0].scatter(self.t_nominal, np.zeros(self.n_shapes), marker='o',
                      edgecolors='b', facecolor='None', s=50, linewidth=1)
        ax[0].scatter(self.eta_nominal, np.ones(self.n_shapes), marker='o',
                      edgecolors='b', facecolor='None', s=50, linewidth=1)
        ax[0].text(0.1, 1.2, 'physical')
        ax[0].text(0.1, 0.22, 'grassmann')

        # physical
        ax[1].axis('off')
        ax[1].axis('equal')
        ax[1].set_title('physical')
        ax[1].axis(xmin=-3.5, xmax=4.5, ymin=-5., ymax=4)
        # grassmann
        ax[2].axis('off')
        ax[2].axis('equal')
        ax[2].set_title('grassmann')
        ax[2].axis(xmin=-0.12, xmax=0.12, ymin=-0.12, ymax=0.12)

        self.k_line = Line2D([], [], color='k')
        self.moving_dot = Line2D([], [], color='k', marker='o', markeredgecolor='k')

        self.g_dot = Line2D([], [], color='g', marker='o', markeredgecolor='g')
        self.r_dot = Line2D([], [], color='r', marker='o', markeredgecolor='r')
        ax[0].add_line(self.g_dot)
        ax[0].add_line(self.r_dot)
        ax[0].add_line(self.k_line)
        ax[0].add_line(self.moving_dot)

        self.k_line1 = Line2D([], [], color='k')
        self.moving_dot1 = Line2D([], [], color='k', marker='o', markeredgecolor='k')
        self.g_dot1 = Line2D([], [], color='g', marker='o', markeredgecolor='g')
        self.r_dot1 = Line2D([], [], color='r', marker='o', markeredgecolor='r')
        ax[0].add_line(self.g_dot1)
        ax[0].add_line(self.r_dot1)
        ax[0].add_line(self.k_line1)
        ax[0].add_line(self.moving_dot1)

        self.m_dot = Line2D([], [], color='m', marker='o', markeredgecolor='m')
        self.phys_shape = Line2D([], [], color='g', linewidth=2)
        self.phys_shape_next = Line2D([], [], color='r', linewidth=2)
        self.phys_moving = Line2D([], [], color='k', linewidth=1)
        ax[1].add_line(self.m_dot)
        ax[1].add_line(self.phys_shape)
        ax[1].add_line(self.phys_shape_next)
        ax[1].add_line(self.phys_moving)

        self.gr_shape = Line2D([], [], color='g', linewidth=2)
        self.gr_shape_next = Line2D([], [], color='r', linewidth=2)
        self.gr_moving = Line2D([], [], color='k', linewidth=1)
        ax[2].add_line(self.gr_shape)
        ax[2].add_line(self.gr_shape_next)
        ax[2].add_line(self.gr_moving)

        animation.TimedAnimation.__init__(self, fig, interval=delay, blit=True)

    def _draw_frame(self, framedata):
        i = framedata
        j = np.where(self.eta_span[i] >= self.eta_nominal)[0][-1]
        j_prev = np.where(self.eta_span[i-1] >= self.eta_nominal)[0][-1]
        if i == 0:
            j_prev = -1
        if j > j_prev:
            self.g_dot.set_data(self.t_nominal[j], 0)
            self.g_dot1.set_data(self.eta_nominal[j], 1)
            self.phys_shape.set_data(self.x_phys[j, :, 0], self.x_phys[j, :, 1])
            self.gr_shape.set_data(self.x_gr[j, :, 0], self.x_gr[j, :, 1])
            if j < self.n_shapes - 1:
                self.r_dot.set_data(self.t_nominal[j + 1], 0)
                self.r_dot1.set_data(self.eta_nominal[j + 1], 1)
                self.phys_shape_next.set_data(self.x_phys[j+1, :, 0], self.x_phys[j+1, :, 1])
                self.gr_shape_next.set_data(self.x_gr[j+1, :, 0], self.x_gr[j+1, :, 1])

        # grassmann line

        self.k_line.set_data(self.t_span[:i], np.zeros(i))
        self.moving_dot.set_data(self.t_span[i], 0)
        # physical line
        self.k_line1.set_data(self.eta_span[:i], np.ones(i))
        self.moving_dot1.set_data(self.eta_span[i], 1)

        # circles
        self.m_dot.set_data(self.geodesic_phys[i, 0, 0], self.geodesic_phys[i, 0, 1])
        self.phys_moving.set_data(self.geodesic_phys[i, :, 0], self.geodesic_phys[i, :, 1])
        self.gr_moving.set_data(self.geodesic_gr[i, :, 0], self.geodesic_gr[i, :, 1])

        self._drawn_artists = [self.k_line, self.moving_dot, self.g_dot, self.r_dot,
                               self.k_line1, self.moving_dot1, self.g_dot1, self.r_dot1,
                               self.phys_shape, self.phys_shape_next, self.phys_moving,
                               self.gr_shape, self.gr_shape_next, self.gr_moving, self.m_dot]

    def new_frame_seq(self):
        return iter(range(self.n))

    def _init_draw(self):
        lines = [self.moving_dot, self.k_line, self.g_dot, self.r_dot,
                 self.moving_dot1, self.k_line1, self.g_dot1, self.r_dot1,
                 self.phys_shape, self.phys_shape_next, self.phys_moving,
                 self.gr_shape, self.gr_shape_next, self.gr_moving, self.m_dot]
        for l in lines:
            l.set_data([], [])