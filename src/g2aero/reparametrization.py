import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import lsq_linear
from scipy.special import comb

from g2aero.Grassmann import landmark_affine_transform
from g2aero.utils import add_tailedge_gap


def get_landmarks(xy, n_landmarks=401, method='polar', add_gap=False, **kwargs):
    """

    :param xy:
    :param n_landmarks:
    :param method:
    :param add_gap:
    :param kwargs:
    :return:
    """
    xy = np.asarray(xy)
    le_ind = np.argmin(xy[:, 0])  # Leading edge index
    y1_avg = np.average(xy[:le_ind, 1])  # Determine orientation of the airfoil shape
    if y1_avg > 0:
        xy = xy[::-1]  # Flip such that the pressure side is always first

    # Normalize coordinates (x from 0 to 1) to rid of the rounding error
    x_min, x_max = np.min(xy[:, 0]), np.max(xy[:, 0])
    xy[:, 0] = (xy[:, 0] - x_min) / (x_max - x_min)
    xy[:, 1] = xy[:, 1] / (x_max - x_min)
    if not np.allclose(x_min, 0) or not np.allclose(x_max, 1):
        print('WARNING!: Airfoil shape is not normalized properly', x_min, x_max, (x_max - x_min))

    if method == 'cst':
        xy_landmarks, _ = cst_reparametrization(xy, n_landmarks, **kwargs)
    elif method == 'polar':
        xy_landmarks = polar_reparametrization(xy, n_landmarks, **kwargs)

    # make tailedge gap
    if add_gap:
        xy_landmarks = add_tailedge_gap(xy_landmarks, 0.002)

    return xy_landmarks


def polar_reparametrization(xy, n_landmarks=401, sampling='uniform_gr'):

    def arc_distance(xy):
        dist = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        t = np.cumsum(dist) / np.sum(dist)
        return np.hstack(([0.0], t))

    xy_gr, M, b = landmark_affine_transform(xy)
    # arc length
    t = arc_distance(xy_gr)
    # angles and alpha
    theta_i = np.unwrap(np.arctan2(xy_gr[:, 1], xy_gr[:, 0]))
    alpha_i = xy_gr[:, 0] * np.cos(theta_i) + xy_gr[:, 1] * np.sin(theta_i)
    theta = PchipInterpolator(t, theta_i)
    alpha = PchipInterpolator(t, alpha_i)

    # distribute landmarks uniformly along the arc length on Grassmann
    if sampling == 'uniform_gr':
        t_new = np.linspace(0, 1, n_landmarks)
    # distribute landmarks uniformly along the arc length in physical space
    if sampling == 'uniform_phys':
        t_tmp = np.linspace(0, 1, 10000)
        landmarks = np.vstack((alpha(t_tmp) * np.cos(theta(t_tmp)), alpha(t_tmp) * np.sin(theta(t_tmp)))).T
        landmarks = landmarks @ M + b
        t_phys = arc_distance(landmarks)
        t_new = PchipInterpolator(t_phys, t_tmp)(np.linspace(0, 1, n_landmarks))

    #TODO: fix sampling with curvature (doesn't work properly right now)
    # distribute landmarks according to curvature of shape in physical space
    elif sampling == 'curvature_polar':
        th, d_th, dd_th = theta(t), theta(t, 1), theta(t, 2)
        a, d_a, dd_a = alpha(t), alpha(t, 1), alpha(t, 2)

        n = np.vstack((np.cos(th), np.sin(th)))
        d_n = np.vstack((-np.sin(th), np.cos(th)))
        dd_n = np.vstack((-np.cos(th), -np.sin(th)))
        d_s = d_a * n + a * d_n * d_th
        dd_s = dd_a * n + 2 * d_a * d_n * d_th + a * (dd_n * d_th ** 2 + d_n * dd_th)

        curvature_i = np.abs(d_s[0] * dd_s[1] - d_s[1] * dd_s[0]) / (d_s[0] ** 2 + d_s[1] ** 2) ** 1.5
        curvature_cdf_i = np.cumsum(curvature_i) - curvature_i[0]
        curvature_cdf_i /= curvature_cdf_i[-1]

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 2, figsize=(10, 12))
        ax[0, 0].plot(t, th, '.')
        ax[0, 0].set_ylabel('theta(t)')
        ax[0, 1].plot(t, a, '.')
        ax[0, 1].set_ylabel('a(t)')
        ax[1, 0].plot(t, d_a, '.')
        ax[1, 0].set_ylabel("a'(t)")
        ax[1, 1].plot(t, dd_a, '.')
        ax[1, 1].set_ylabel("a''(t)")
        ax[2, 0].semilogy(t, curvature_i, '.')
        ax[2, 0].set_ylabel("curvature(t)")
        ax[2, 1].plot(t, curvature_cdf_i, '.')
        ax[2, 1].set_ylabel("cdf_curvature(t)")

        cdf = PchipInterpolator(t, curvature_cdf_i)
        t_tmp = np.linspace(0, 1, 10000)
        interpolator_cdf = PchipInterpolator(cdf(t_tmp), t_tmp)
        t_new = interpolator_cdf(np.linspace(0, 1, n_landmarks))

    elif sampling == 'curvature_planar':
        t_phys = arc_distance(xy)
        # s1 = CubicSpline(t_phys, xy[:, 0], bc_type=((2, 0), (2, 0)))
        # s2 = CubicSpline(t_phys, xy[:, 1], bc_type=((2, 0), (2, 0)))
        s1 = CubicSpline(t_phys, xy[:, 0])
        s2 = CubicSpline(t_phys, xy[:, 1])
        # s1 = PchipInterpolator(t_phys, xy[:, 0])
        # s2 = PchipInterpolator(t_phys, xy[:, 1])

        ds1, ds2 = s1(t_phys, 1), s2(t_phys, 1)
        dds1, dds2 = s1(t_phys, 2), s2(t_phys, 2)

        curvature_i = np.abs(ds1 * dds2 - ds2 * dds1) / (ds1 ** 2 + ds2 ** 2) ** 1.5
        curvature_cdf_i = np.cumsum(curvature_i) - curvature_i[0]
        curvature_cdf_i /= curvature_cdf_i[-1]

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(4, 2, figsize=(10, 12))
        ax[0, 0].plot(t_phys, xy[:, 0], '.')
        ax[0, 0].set_ylabel('s1(t)')
        ax[0, 1].plot(t_phys, xy[:, 1], '.')
        ax[0, 1].set_ylabel('s2(t))')
        ax[1, 0].plot(t_phys, ds1, '.')
        ax[1, 0].set_ylabel("s1'(t)")
        ax[1, 1].plot(t_phys, ds2, '.')
        ax[1, 1].set_ylabel("s2'(t)")
        ax[2, 0].plot(t_phys, dds1, '.')
        ax[2, 0].set_ylabel("s1''(t)")
        ax[2, 1].plot(t_phys, dds2, '.')
        ax[2, 1].set_ylabel("s2''(t)")
        ax[3, 0].semilogy(t_phys, curvature_i, '.')
        ax[3, 1].plot(t_phys, curvature_cdf_i, '.')

        cdf = PchipInterpolator(t_phys, curvature_cdf_i)
        t_tmp = np.linspace(0, 1, 10000)
        interpolator_cdf = PchipInterpolator(cdf(t_tmp), t_tmp)
        t_new = interpolator_cdf(np.linspace(0, 1, n_landmarks))

    landmarks = np.vstack((alpha(t_new) * np.cos(theta(t_new)), alpha(t_new) * np.sin(theta(t_new)))).T
    landmarks = landmarks @ M + b
    # if len(xy) < len(landmarks):
    return landmarks


def cst_reparametrization(xy, n_landmarks=401, name='', cst_order=8):
    xy = np.asarray(xy)
    if name in ['circular', 'Cylinder', 'Cylinder1', 'Cylinder2']:
        n1, n2 = 0.5, 0.5
    else:
        n1, n2 = 0.5, 1.0

    le_ind = np.argmin(xy[:, 0])  # Leading edge index

    # # tailedge gap
    # te_lower, te_upper = xy[0, 1], xy[-1, 1]
    # split int upper and lower parts
    xy_upper, xy_lower = xy[le_ind:], xy[:le_ind]

    # calculate cst coefficients
    cst_upper = calc_cst_param(xy_upper[:, 0], xy_upper[:, 1], n1, n2, cst_order)
    cst_lower = calc_cst_param(xy_lower[:, 0], xy_lower[:, 1], n1, n2, cst_order)
    cst = np.r_[cst_lower, cst_upper]

    n_half = int(n_landmarks / 2)
    x_c = -np.cos(np.linspace(0, np.pi, n_half + 1)) * 0.5 + 0.5
    xy_landmarks = from_cst_parameters(x_c, cst_lower, cst_upper, n1, n2)

    return xy_landmarks, cst


def calc_cst_param(x, y, n1, n2, order=8):
    """
    Solve the least squares problem for a given shape
    :param x: (np.ndarray): (x/c) coordinates locations
    :param y: (np.ndarray): (y/c) coordinate locations
    :param n1: normal coord
    :param n2: normal coord
    :param order:
    :return: ``(BP+1)`` CST parameters
    """
    amat = cst_matrix(x, n1, n2, order)
    amat = np.hstack((amat, x.reshape(-1, 1)))
    bvec = y
    out = lsq_linear(amat, bvec)
    return out.x


def cst_matrix(x, n1, n2, order):
    x = np.asarray(x)
    class_function = np.power(x, n1) * np.power((1.0 - x), n2)

    K = comb(order, range(order + 1))
    shape_function = np.empty((order + 1, x.shape[0]))
    for i in range(order + 1):
        shape_function[i, :] = K[i] * np.power(x, i) * np.power((1.0 - x), (order - i))

    return (class_function * shape_function).T


def from_cst_parameters(xinp, cst_lower, cst_upper, n1, n2):
    """ Compute landmark coordinates for the airfoil
    :param xinp: (np.ndarray): Non-dimensional x-coordinate locations
    :param cst_lower: (np.ndarray): cst parameters for lower part
    :param cst_upper: (np.ndarray): cst parameters for upper part
    :param n1: (double): normal coord
    :param n2: (double): normal coord
    :return: Numpy arrays for landmark coordinates
    """
    x = np.asarray(xinp)
    order = np.size(cst_lower) - 2
    amat = cst_matrix(xinp, n1, n2, order)
    amat = np.hstack((amat, x.reshape(-1, 1)))

    y_lower = np.dot(amat, cst_lower)
    y_upper = np.dot(amat, cst_upper)

    x = np.hstack((x[::-1], x[1:])).reshape(-1, 1)
    y = np.hstack((y_lower[::-1], y_upper[1:])).reshape(-1, 1)

    return np.hstack((x, y))