import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import plotly.graph_objects as go
mpl.rcParams['animation.embed_limit'] = 2**128


class SubplotAnimationPerturb(animation.TimedAnimation):
    def __init__(self, eta_span, t_span1, t_span2,
                 phys_shapes1, phys_shapes2,
                 gr_shapes1, gr_shapes2,
                 twist1, twist2, scale_x1, scale_x2, scale_y1, scale_y2, delay=150):

        font = {'family': 'times',
                'weight': 'bold',
                'size': 16}

        mpl.rc('font', **font)
        self.n = len(eta_span)
        self.geodesic_gr1 = gr_shapes1
        self.geodesic_gr2 = gr_shapes2
        self.geodesic_phys1 = phys_shapes1
        self.geodesic_phys2 = phys_shapes2
        self.eta_span = eta_span

        self.x = np.linspace(0, 1, self.n)
        self.eta_scaled = np.linspace(self.eta_span[0], self.eta_span[-1], self.n)
        self.t_span1 = t_span1(self.eta_scaled)
        self.t_span2 = t_span2(self.eta_scaled)
        self.t_span2 *= np.diff(self.t_span1)[2]/np.diff(self.t_span2)[2]
        self.twist1 = np.degrees(twist1(self.x))
        self.twist2 = np.degrees(twist2(self.x))
        self.scalex1 = scale_x1(self.x)
        self.scalex2 = scale_x2(self.x)
        self.scaley1 = scale_y1(self.x)
        self.scaley2 = scale_y2(self.x)

        gs_kw = dict(width_ratios=[1, 1, 1])
        fig, ax = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True, gridspec_kw=gs_kw)

        # physical
        ax[0, 0].axis('equal')
        ax[0, 0].set_title('physical shape')
        ax[0, 0].axis(xmin=-3.1, xmax=4.1, ymin=-3.6, ymax=3.6)

        # grassmann
        ax[0, 1].axis('equal')
        ax[0, 1].set_title('Grassmann shape')
        ax[0, 1].axis(xmin=-0.1, xmax=0.11, ymin=-0.1, ymax=0.11)

        # cdf
        ax[0, 2].set_title('Grassmann distance cdf')
        ax[0, 2].axis(ymin=0, ymax=self.t_span2[-1]+0.05, xmin=self.eta_scaled[0], xmax=self.eta_scaled[-1])
        ax[0, 2].add_artist(Line2D(self.eta_scaled, self.t_span1, ls='--', color='k', zorder=1))
        ax[0, 2].add_artist(Line2D(self.eta_scaled, self.t_span2, color='r',  zorder=0))
        ax[0, 2].set_xlabel(r'$\eta^{\prime}$')

        # twist
        ax[1, 0].set_title('twist')
        ax[1, 0].add_artist(Line2D(self.x, self.twist1, ls='--', color='k', zorder=1))
        ax[1, 0].add_artist(Line2D(self.x, self.twist2, color='r', zorder=0))
        ax[1, 0].axis(xmin=0.0, xmax=1.01, ymin=-5, ymax=17)
        ax[1, 0].set_xlabel(r'$\eta$')

        # scale_x
        ax[1, 1].set_title('scale x')
        ax[1, 1].add_artist(Line2D(self.x, self.scalex1, ls='--', color='k', zorder=1))
        ax[1, 1].add_artist(Line2D(self.x, self.scalex2, color='r',  zorder=0))
        ax[1, 1].axis(xmin=0.0, xmax=1.01, ymin=0, ymax=6)
        ax[1, 1].set_xlabel(r'$\eta$')

        # scale_y
        ax[1, 2].set_title('scale y')
        ax[1, 2].add_artist(Line2D(self.x, self.scaley1, ls='--', color='k', zorder=1))
        ax[1, 2].add_artist(Line2D(self.x, self.scaley2, color='r', zorder=0))
        ax[1, 2].axis(xmin=0.0, xmax=1.01, ymin=0, ymax=6)
        ax[1, 2].set_xlabel(r'$\eta$')

        ####################################################################################################
        #
        ####################################################################################################
        self.phys_moving1 = Line2D([], [], ls='--', color='k', zorder=1, linewidth=2)
        self.phys_moving2 = Line2D([], [], color='r', linewidth=1)
        ax[0, 0].add_line(self.phys_moving1)
        ax[0, 0].add_line(self.phys_moving2)

        self.gr_moving1 = Line2D([], [], ls='--', color='k', zorder=1, linewidth=2)
        self.gr_moving2 = Line2D([], [], color='r', linewidth=1)
        ax[0, 1].add_line(self.gr_moving1)
        ax[0, 1].add_line(self.gr_moving2)

        self.k_dot_cdf = Line2D([], [], color='k', marker='o', markeredgecolor='k')
        ax[0, 2].add_line(self.k_dot_cdf)
        self.r_dot_cdf = Line2D([], [], color='k', marker='o', markeredgecolor='k')
        ax[0, 2].add_line(self.r_dot_cdf)

        # self.k_line_twist = Line2D([], [], color='k')
        self.k_dot_twist = Line2D([], [], color='k', marker='o', markeredgecolor='k')
        # self.r_line_twist = Line2D([], [], color='k')
        self.r_dot_twist = Line2D([], [], color='k', marker='o', markeredgecolor='k')
        # ax[1, 0].add_line(self.k_line_twist)
        ax[1, 0].add_line(self.k_dot_twist)
        # ax[1, 0].add_line(self.r_line_twist)
        ax[1, 0].add_line(self.r_dot_twist)

        # self.k_line_scalex = Line2D([], [], color='k')
        self.k_dot_scalex = Line2D([], [], color='k', marker='o', markeredgecolor='k')
        # self.r_line_scalex = Line2D([], [], color='k')
        self.r_dot_scalex = Line2D([], [], color='k', marker='o', markeredgecolor='k')
        # ax[1, 1].add_line(self.k_line_scalex)
        ax[1, 1].add_line(self.k_dot_scalex)
        # ax[1, 1].add_line(self.r_line_scalex)
        ax[1, 1].add_line(self.r_dot_scalex)

        # self.k_line_scaley = Line2D([], [], color='k')
        self.k_dot_scaley = Line2D([], [], color='k', marker='o', markeredgecolor='k')
        # self.r_line_scaley = Line2D([], [], color='k')
        self.r_dot_scaley = Line2D([], [], color='k', marker='o', markeredgecolor='k')
        # ax[1, 2].add_line(self.k_line_scaley)
        ax[1, 2].add_line(self.k_dot_scaley)
        # ax[1, 2].add_line(self.r_line_scaley)
        ax[1, 2].add_line(self.r_dot_scaley)

        animation.TimedAnimation.__init__(self, fig, interval=delay, blit=True)

    def _draw_frame(self, framedata):
        i = framedata
        # cdf line
        self.k_dot_cdf.set_data(self.eta_span[i], self.t_span1[i])
        self.r_dot_cdf.set_data(self.eta_span[i], self.t_span2[i])

        # circles
        self.phys_moving1.set_data(self.geodesic_phys1[i, :, 0], self.geodesic_phys1[i, :, 1])
        self.phys_moving2.set_data(self.geodesic_phys2[i, :, 0], self.geodesic_phys2[i, :, 1])

        self.gr_moving1.set_data(self.geodesic_gr1[i, :, 0], self.geodesic_gr1[i, :, 1])
        self.gr_moving2.set_data(self.geodesic_gr2[i, :, 0], self.geodesic_gr2[i, :, 1])

        # plots
        self.k_dot_twist.set_data(self.x[i], self.twist1[i])
        self.k_dot_scalex.set_data(self.x[i], self.scalex1[i])
        self.k_dot_scaley.set_data(self.x[i], self.scaley1[i])
        self.r_dot_twist.set_data(self.x[i], self.twist2[i])
        self.r_dot_scalex.set_data(self.x[i], self.scalex2[i])
        self.r_dot_scaley.set_data(self.x[i], self.scaley2[i])
        self._drawn_artists = [self.k_dot_cdf, self.r_dot_cdf,
                               self.phys_moving1, self.phys_moving2,
                               self.gr_moving1, self.gr_moving2,
                               self.k_dot_twist, self.k_dot_scalex, self.k_dot_scaley]

    def new_frame_seq(self):
        return iter(range(self.n))

    def _init_draw(self):
        lines = [self.k_dot_cdf, self.r_dot_cdf,
                 self.phys_moving1, self.phys_moving2,
                 self.gr_moving1, self.gr_moving2,
                 self.k_dot_twist,  self.r_dot_twist,
                 self.k_dot_scalex,  self.r_dot_scalex,
                 self.k_dot_scaley, self.r_dot_scaley]
        for l in lines:
            l.set_data([], [])


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


def plot_subplots(X_all, X_Gr, cumsum_dist, eta_span, plot_folder):
    colors = ['g', 'r', 'y', 'b', 'orange', 'm', 'c']
    fig, ax = plt.subplots(1, 4, figsize=(14, 3))
    circle = plt.Circle((0, 0), 1, color='k', fill=False)
    ax[0].add_artist(circle)
    circle = plt.Circle((0, 0), 1, color='k', fill=False)
    ax[1].add_artist(circle)
    n_airfoils = len(X_all)
    for i in range(n_airfoils):
        X = X_all[i]
        X_grassmann = X_Gr[i]
        dist = cumsum_dist[i]

        # plot eta on circle
        scaled_eta = (eta_span - eta_span[0]) * 2 * np.pi / (eta_span[-1] - eta_span[0])
        ax[0].scatter(np.cos(scaled_eta), np.sin(scaled_eta), edgecolors=colors[i], marker='o',
                      facecolor='None', s=50, linewidth=1)
        ax[0].axis('equal')
        ax[0].set_title('Phys distance')
        ax[0].axis(xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2)

        # plot dist on circle
        scaled_dist = dist * 2 * np.pi / cumsum_dist[-1]
        ax[1].scatter(np.cos(scaled_dist), np.sin(scaled_dist), edgecolors=colors[i], marker='o',
                      facecolor='None', s=50, linewidth=1)
        ax[1].axis('equal')
        ax[1].set_title('Gr distance')
        ax[1].axis(xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2)

        # plot X
        ax[2].scatter(X[:, 0], X[:, 1], color=colors[i], marker='.', facecolor='None', s=20, linewidth=1)
        ax[2].axis('equal')
        ax[2].set_title('physical')
        ax[2].axis(xmin=-0.2, xmax=1.2, ymin=-0.6, ymax=0.6)

        # plot affine transformed
        ax[3].scatter(X_grassmann[:, 0], X_grassmann[:, 1], color=colors[i], marker='.', facecolor='None', s=20, linewidth=1)
        ax[3].axis('equal')
        ax[3].set_title('grassmann')
    plt.savefig(os.path.join(plot_folder, f'subplots.pdf'))


def plot_dist(eta_span, cumsum_dist, eta, t_span, plot_folder):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(eta_span, cumsum_dist)
    ax.plot(eta, t_span, linewidth=2)
    ax.axis('equal')
    ax.axis(xmin=0.0, ymin=-0.05, xmax=1, ymax=1)
    ax.set_xlabel(r'$\eta$ (physical)')
    ax.set_ylabel(r'$t$ (grassmannian)')
    ax.set_title('CDF of distances over the Grassmann')
    plt.savefig(os.path.join(plot_folder, f'dist.pdf'))


def plot_M_and_b(M, t_nominal, M_lin, t, plot_folder, name):
    colors = ['#00429d', '#64c0d0', '#b9ffe7', '#ff5589', '#db0052', '#93003a']
    labels = ['M_11', 'M_12', 'M_21', 'M_22', 'b_1', 'b_2']
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for i in range(6):
        # ax[0].scatter(t_nominal, M_flat[:, i], label=f'{i}')
        ax.plot(t, M_lin[:, i], '-', color=colors[i], label=labels[i])
        ax.plot(t_nominal, M[:, i], '--', color=colors[i])
    # ax.axis('equal')
    # ax.set_xlabel(r'$\eta$ (physical)')
    ax.set_xlabel(r'$t$ (grassmannian)')
    ax.set_title('M and b interpolation')
    ax.legend(loc=0)
    plt.savefig(os.path.join(plot_folder, f'{name}.pdf'))


def plot_M_and_b_t_eta(M, t_nominal, M_t, M_eta, t, plot_folder, name, xlabel=r'$t$ (grassmannian)'):
    colors = ['#00429d', '#64c0d0', '#b9ffe7', '#ff5589', '#db0052', '#93003a']
    labels = ['M_11', 'M_12', 'M_21', 'M_22', 'b_1', 'b_2']
    print(M.shape, M_t.shape)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for i in range(6):
        ax.scatter(t_nominal, M[:, i], color=colors[i])
        ax.plot(t, M_t[:, i], '-', color=colors[i], label=labels[i])
        ax.plot(t, M_eta[:, i], '-.', color=colors[i])
        ax.plot(t_nominal, M[:, i], '--', color=colors[i])
    ax.set_xlabel(xlabel)
    ax.set_title('M and b interpolation')
    ax.legend(loc=0)
    plt.savefig(os.path.join(plot_folder, f'{name}.pdf'))


def plot_3d_blade_nominal(shapes, span, plot_folder, name):

    scale = 10
    eta_span = np.repeat(span, shapes.shape[1])*scale
    shapes_x = shapes[:, :, 0].flatten()
    shapes_y = shapes[:, :, 1].flatten()
    fig = go.Figure(data=[go.Scatter3d(x=eta_span, y=shapes_x, z=shapes_y, mode='markers')])
    fig.update_layout(scene_aspectmode='data')
    fig.write_html(os.path.join(plot_folder, name))
    fig.show()
    return fig


# def plot_3d_blade(shapes, shapes_nominal=None, eta_nominal=None, name='blade.html'):
#
#     shapes_x = shapes[:, :, 0].flatten()
#     shapes_y = shapes[:, :, 1].flatten()
#     shapes_z = shapes[:, :, 2].flatten()
#     # eta_nominal = np.repeat(eta_nominal, shapes.shape[1])
#     # nominal_x = shapes_nominal[:, :, 0].flatten()
#     # nominal_y = shapes_nominal[:, :, 1].flatten()
#     fig = go.Figure(data=[go.Scatter3d(x=shapes_z, y=shapes_x, z=shapes_y, mode='lines')])
#                     # ,go.Scatter3d(x=eta_nominal, y=nominal_x, z=nominal_y, mode='markers')])
#     fig.update_layout(scene_aspectmode='data')
#     fig.write_html(name)
#     fig.show()

# def plot_3d_blade(shapes, shapes_nominal, plot_folder, name):
#
#     N_eta_skp = 1
#     N_arc_skp = 25
#     line_b = go.scatter3d.Line(color="#1f77b4", width=1.)
#     line_m = go.scatter3d.Line(color="#db0052", width=4.)
#     data = []
#
#     #     TE_line_u = shapes[:, 0, :]
#     #     TE_line_l = shapes[:, shapes.shape[1]-1, :]
#     #     for i, shape in enumerate(shapes):
#     #         eta_span_z = np.repeat(eta_span[i], len(shape))*scale
#     #         data.append(go.Scatter3d(x=eta_span_z, y=shape[:, 0], z=shape[:, 1], mode='lines', line=line))
#
#     #     data.append(go.Scatter3d(x=eta_span*scale, y=TE_line_u[:, 0], z=TE_line_u[:, 1], mode='lines', line=line))
#     #     data.append(go.Scatter3d(x=eta_span*scale, y=TE_line_l[:, 0], z=TE_line_l[:, 1], mode='lines', line=line))
#     #     data.append(go.Scatter3d(x=eta_span_z, y=shapes[:, 0, 0], z=shapes[:, 0, 1], mode='lines', line=line))
#
#     # Plot Grassmann interpolator cross section refinements
#     for i in range(0, int((shapes.shape[0])), N_eta_skp):
#         data.append(go.Scatter3d(x=shapes[i, :, 2], y=shapes[i, :, 0], z=shapes[i, :, 1], mode='lines', line=line_b))
#
#     # Plot tip
#     i = int((shapes.shape[0])) - 1
#     data.append(go.Scatter3d(x=shapes[i, :, 2], y=shapes[i, :, 0], z=shapes[i, :, 1],
#                              mode='lines', line=line_b))
#
#     # Plot span-wise landmarks
#     span_line = np.empty((shapes.shape[1], shapes.shape[0], 3))
#     for j in range(0, int((shapes.shape[1])), N_arc_skp):
#         span_line[j] = shapes[:, j, :]
#         data.append(go.Scatter3d(x=span_line[j, :, 2], y=span_line[j, :, 0], z=span_line[j, :, 1],
#                                  mode='lines', line=line_b))
#
#   # Plot nominal cross sections
    # for i in range(0, int(shapes_nominal.shape[0])):
    #     nominal_x = shapes_nominal[i, :, 0].flatten()
    #     nominal_y = shapes_nominal[i, :, 1].flatten()
    #     data.append(go.Scatter3d(x=shapes_nominal[i, :, 2].flatten(), y=nominal_x, z=nominal_y, mode='lines', line=line_m))
    #
    # fig = go.Figure(data=data)
    # fig.update_layout(scene_aspectmode='data')
    # fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    # fig.write_html(os.path.join(plot_folder, name))
    # fig.show()




def plot_3d_blade_with_nominal(shapes, eta_span, shapes_nominal, eta_nominal, plot_folder, name):
    scale = 130
    N_eta_skp = 1
    N_arc_skp = 25
    line_b = go.scatter3d.Line(color="#1f77b4", width=1.)
    line_m = go.scatter3d.Line(color="#db0052", width=4.)
    data = []

    #     TE_line_u = shapes[:, 0, :]
    #     TE_line_l = shapes[:, shapes.shape[1]-1, :]
    #     for i, shape in enumerate(shapes):
    #         eta_span_z = np.repeat(eta_span[i], len(shape))*scale
    #         data.append(go.Scatter3d(x=eta_span_z, y=shape[:, 0], z=shape[:, 1], mode='lines', line=line))

    #     data.append(go.Scatter3d(x=eta_span*scale, y=TE_line_u[:, 0], z=TE_line_u[:, 1], mode='lines', line=line))
    #     data.append(go.Scatter3d(x=eta_span*scale, y=TE_line_l[:, 0], z=TE_line_l[:, 1], mode='lines', line=line))
    #     data.append(go.Scatter3d(x=eta_span_z, y=shapes[:, 0, 0], z=shapes[:, 0, 1], mode='lines', line=line))

    # Plot Grassmann interpolator cross section refinements
    for i in range(0, int((shapes.shape[0])), N_eta_skp):
        eta_span_z = np.repeat(eta_span[i], shapes.shape[1]) * scale
        data.append(go.Scatter3d(x=eta_span_z, y=shapes[i, :, 0], z=shapes[i, :, 1], mode='lines', line=line_b))

    # Plot tip
    i = int((shapes.shape[0])) - 1
    eta_span_z = np.repeat(eta_span[i], shapes.shape[1]) * scale
    data.append(go.Scatter3d(x=eta_span_z, y=shapes[i, :, 0], z=shapes[i, :, 1],
                             mode='lines', line=line_b))

    # Plot span-wise landmarks
    span_line = np.empty((shapes.shape[1], shapes.shape[0], 2))
    for j in range(0, int((shapes.shape[1])), N_arc_skp):
        span_line[j] = shapes[:, j, :2]
        data.append(go.Scatter3d(x=eta_span * scale, y=span_line[j, :, 0], z=span_line[j, :, 1],
                                 mode='lines', line=line_b))

    # Plot nominal cross sections
    for i in range(0, int(shapes_nominal.shape[0])):
        eta = np.repeat(eta_nominal[i], shapes_nominal.shape[1]) * scale
        nominal_x = shapes_nominal[i, :, 0].flatten()
        nominal_y = shapes_nominal[i, :, 1].flatten()
        data.append(go.Scatter3d(x=eta, y=nominal_x, z=nominal_y, mode='lines', line=line_m))

    fig = go.Figure(data=data)
    fig.update_layout(scene_aspectmode='data')
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    fig.write_html(os.path.join(plot_folder, name))
    fig.show()


def plot_3d_blade(shapes,name='blade.html'):

    N_eta_skp = 1
    N_arc_skp = 25
    line_b = go.scatter3d.Line(color="#1f77b4", width=1.)
    line_m = go.scatter3d.Line(color="#db0052", width=4.)
    data = []

    #     TE_line_u = shapes[:, 0, :]
    #     TE_line_l = shapes[:, shapes.shape[1]-1, :]
    #     for i, shape in enumerate(shapes):
    #         eta_span_z = np.repeat(eta_span[i], len(shape))*scale
    #         data.append(go.Scatter3d(x=eta_span_z, y=shape[:, 0], z=shape[:, 1], mode='lines', line=line))

    #     data.append(go.Scatter3d(x=eta_span*scale, y=TE_line_u[:, 0], z=TE_line_u[:, 1], mode='lines', line=line))
    #     data.append(go.Scatter3d(x=eta_span*scale, y=TE_line_l[:, 0], z=TE_line_l[:, 1], mode='lines', line=line))
    #     data.append(go.Scatter3d(x=eta_span_z, y=shapes[:, 0, 0], z=shapes[:, 0, 1], mode='lines', line=line))

    # Plot Grassmann interpolator cross section refinements
    for i in range(0, int((shapes.shape[0])), N_eta_skp):
        data.append(go.Scatter3d(x=shapes[i, :, 2], y=shapes[i, :, 0], z=shapes[i, :, 1], mode='lines', line=line_b))

    # # Plot tip
    # i = int((shapes.shape[0])) - 1
    # eta_span_z = np.repeat(eta_span[i], shapes.shape[1]) * scale
    # data.append(go.Scatter3d(x=eta_span_z, y=shapes[i, :, 0], z=shapes[i, :, 1],
    #                          mode='lines', line=line_b))

    # Plot span-wise landmarks
    span_line = np.empty((shapes.shape[1], shapes.shape[0], 3))
    for j in range(0, int((shapes.shape[1])), N_arc_skp):
        span_line[j] = shapes[:, j]
        data.append(go.Scatter3d(x=span_line[j, :, 2], y=span_line[j, :, 0], z=span_line[j, :, 1],
                                 mode='lines', line=line_b))

    # # Plot nominal cross sections
    # for i in range(0, int(shapes_nominal.shape[0])):
    #     eta = np.repeat(eta_nominal[i], shapes_nominal.shape[1]) * scale
    #     nominal_x = shapes_nominal[i, :, 0].flatten()
    #     nominal_y = shapes_nominal[i, :, 1].flatten()
    #     data.append(go.Scatter3d(x=eta, y=nominal_x, z=nominal_y, mode='lines', line=line_m))

    fig = go.Figure(data=data)
    fig.update_layout(scene_aspectmode='data')
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    fig.write_html(name)
    fig.show()
