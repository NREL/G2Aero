import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import lsq_linear
from scipy.special import comb

from .Grassmann import landmark_affine_transform
from .utils import add_tailedge_gap, arc_distance


def get_landmarks(xy, n_landmarks=401, method='planar', add_gap=False, **kwargs):
    """Provides landmarks after reparametrization.
    :param xy: (n, 2) given coordinated defining the shape
    :param n_landmarks: scalar number of landmarks in returned shape
    :param method: method of reparametrization: 'cst', 'polar' or 'planar'
    :param add_gap: size of the tail gap added to the shape
    :param kwargs: additional parameters for reparametrization (e.g. sampling strategy for 'polar' and 'planar' method)
    :return: (n_landmarks, 2) array of landmarks after reparametrization
    """
    xy = np.asarray(xy)

    # check for epsilon close arc-lengths and discard landmarks which are too close (considered a "duplicate landmark")
    ind = np.where(np.diff(arc_distance(xy), axis=0) <= 1e-7)[0]
    xy = np.delete(xy, ind, axis=0)

    le_ind = np.argmin(xy[:, 0])  # Leading edge index
    y1_avg = np.average(xy[:le_ind, 1])  # Determine orientation of the airfoil shape
    if y1_avg > 0:
        xy = xy[::-1]  # Flip such that the pressure side is always first
        
    if method == 'cst':
        xy_landmarks, _ = cst_reparametrization(xy, n_landmarks, **kwargs)
    elif method == 'polar':
        xy_landmarks = polar_reparametrization(xy, n_landmarks, **kwargs)
    elif method == 'planar':
        xy_landmarks = planar_reparametrization(xy, n_landmarks, **kwargs)

    # make tailedge gap
    if add_gap:
        xy_landmarks = add_tailedge_gap(xy_landmarks, 0.002)

    return xy_landmarks


def planar_reparametrization(xy, n_landmarks, sampling='uniform', **kwargs):

    t_phys = arc_distance(xy)
    
    s1 = CubicSpline(t_phys, xy[:, 0], bc_type='natural')
    s2 = CubicSpline(t_phys, xy[:, 1], bc_type='natural')

    # check if the shape is closed (0 tailgap)
    # if not closed reduce the number of landmarks by one (so after closing we end up with the same number of landmarks)
    closed_flag = True
    if np.linalg.norm(xy[0] - xy[-1]) > 10e-7:
        n_landmarks -= 1
        closed_flag = False

    # if sampling == 'chebyshev_nodes':
    #     t_new = np.polynomial.chebyshev.Chebyshev.roots()

    if sampling == 'uniform_gr':
        xy_gr, _, _ = landmark_affine_transform(xy)
        t_gr = arc_distance(xy_gr)
        interpolator = PchipInterpolator(t_gr, t_phys)
        t_new = interpolator(np.linspace(0, 1, n_landmarks))

    if sampling == 'uniform_phys' or sampling == 'uniform':
        t_new = np.linspace(0, 1, n_landmarks)

    if sampling == 'curvature':
        t_tmp = np.linspace(0, 1, 100000)
        curvature_i = curvature_planar(t_tmp, s1, s2)
        curvature_cdf_i = np.cumsum(curvature_i) - curvature_i[0]
        curvature_cdf_i /= curvature_cdf_i[-1]

        t_new = PchipInterpolator(curvature_cdf_i, t_tmp)(np.linspace(0, 1, n_landmarks))

    landmarks = np.vstack((s1(t_new), s2(t_new))).T

    # close the shape by adding first landmark to the end
    if not closed_flag:
        landmarks = np.vstack((landmarks, landmarks[0]))

    return landmarks


def polar_reparametrization(xy, n_landmarks=401, sampling='uniform_gr', **kwargs):

    xy_gr, M, b = landmark_affine_transform(xy)
    t = arc_distance(xy_gr)     # arc length
    # angles and alpha
    theta_i = np.unwrap(np.arctan2(xy_gr[:, 1], xy_gr[:, 0]))
    alpha_i = xy_gr[:, 0] * np.cos(theta_i) + xy_gr[:, 1] * np.sin(theta_i)
    theta = CubicSpline(t, theta_i)
    alpha = CubicSpline(t, alpha_i, bc_type='natural')
    
    # distribute landmarks uniformly along the arc length on Grassmann
    if sampling == 'uniform_gr':
        t_new = np.linspace(0, 1, n_landmarks)
    # distribute landmarks uniformly along the arc length in physical space
    elif sampling == 'uniform_phys' or sampling == 'uniform':
        t_tmp = np.linspace(0, 1, 100000)
        landmarks = np.vstack((alpha(t_tmp) * np.cos(theta(t_tmp)), alpha(t_tmp) * np.sin(theta(t_tmp)))).T
        landmarks = landmarks @ M + b
        t_phys = arc_distance(landmarks)
        t_new = PchipInterpolator(t_phys, t_tmp)(np.linspace(0, 1, n_landmarks))
    # distribute landmarks according to curvature of shape in physical space
    elif sampling == 'curvature':
        t_tmp = np.linspace(0, 1, 100000)
        curvature_i = curvature_polar(t_tmp, alpha, theta, M)
        curvature_cdf_i = np.cumsum(curvature_i) - curvature_i[0]
        curvature_cdf_i /= curvature_cdf_i[-1]
        t_new = PchipInterpolator(curvature_cdf_i, t_tmp)(np.linspace(0, 1, n_landmarks))

    landmarks = np.vstack((alpha(t_new) * np.cos(theta(t_new)), alpha(t_new) * np.sin(theta(t_new)))).T @ M + b
    return landmarks


def cst_reparametrization(xy, n_landmarks=401, original_landmarks=False, name='', cst_order=8, **kwargs):
    xy = np.asarray(xy)
    if name in ['circular', 'Cylinder', 'Cylinder1', 'Cylinder2']:
        n1, n2 = 0.5, 0.5
    else:
        n1, n2 = 0.5, 1.0

    le_ind = np.argmin(xy[:, 0])  # Leading edge index
    # split int upper and lower parts
    xy_upper, xy_lower = xy[le_ind:], xy[:le_ind]

    # tailedge gap
    te_lower, te_upper = xy[0, 1], xy[-1, 1]

    # calculate cst coefficients
    cst_upper = calc_cst_param(xy_upper[:, 0], xy_upper[:, 1], n1, n2, te_upper, cst_order)
    cst_lower = calc_cst_param(xy_lower[:, 0], xy_lower[:, 1], n1, n2, te_lower, cst_order)

    if original_landmarks:
        upper = halfsurface_from_cst_parameters(xy_upper[:, 0], cst_upper, n1, n2, te_upper)
        lower = halfsurface_from_cst_parameters(xy_lower[:, 0], cst_lower, n1, n2, te_lower)
        xy_landmarks = np.vstack((lower, upper))
    else:
        n_half = int(n_landmarks / 2)
        x_c = -np.cos(np.linspace(0, np.pi, n_half + 1)) * 0.5 + 0.5
        xy_landmarks = from_cst_parameters(x_c, cst_lower, cst_upper, n1, n2, te_lower, te_upper)

    cst = np.r_[cst_lower, cst_upper, te_lower, te_upper]

    return xy_landmarks, cst


def calc_cst_param(x, y, n1, n2, y_tailedge=0.0, order=8):
    """
    Solve the least squares problem for a given shape
    :param x: (np.ndarray): (x/c) coordinates locations
    :param y: (np.ndarray): (y/c) coordinate locations
    :param n1: normal coord
    :param n2: normal coord
    :param order: order of the polynomial used
    :return: CST parameters
    """
    amat = cst_matrix(x, n1, n2, order)
    bvec = y - x * y_tailedge
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


def from_cst_parameters(x, cst_lower, cst_upper, n1=0.5, n2=1.0, te_lower=0, te_upper=0):
    """ Compute landmark coordinates for the airfoil given x-coordinate locations
    :param x: (np.ndarray): Non-dimensional x-coordinate locations
    :param cst_lower: (np.ndarray): cst parameters for lower part
    :param cst_upper: (np.ndarray): cst parameters for upper part
    :param n1: (double): normal coord
    :param n2: (double): normal coord
    :param te_lower: (double): tail edge size for lower part
    :param te_upper: (double): tail edge size for upper part
    :return: Numpy arrays for landmark coordinates
    """
    x = np.asarray(x)
    order = np.size(cst_lower) - 1
    amat = cst_matrix(x, n1, n2, order)

    y_lower = np.dot(amat, cst_lower) + te_lower * x
    y_upper = np.dot(amat, cst_upper) + te_upper * x

    x = np.hstack((x[::-1], x[1:])).reshape(-1, 1)
    y = np.hstack((y_lower[::-1], y_upper[1:])).reshape(-1, 1)

    return np.hstack((x, y))

def halfsurface_from_cst_parameters(x, cst, n1=0.5, n2=1.0, te=0):
    """ Compute landmark coordinates for the upper or lower surface of airfoil
    :param x: (np.ndarray): Non-dimensional x-coordinate locations
    :param cst: (np.ndarray): cst parameters for upper or lower part
    :param n1: (double): normal coord
    :param n2: (double): normal coord
    :param te: (double): tail edge size for uppper or lower part
    :return: Numpy arrays for landmark coordinates
    """
    x = np.asarray(x)
    order = np.size(cst) - 1
    amat = cst_matrix(x, n1, n2, order)
    y = np.dot(amat, cst) + te * x

    return np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))


def curvature_polar(t, alpha, theta, M):
    th, d_th, dd_th = theta(t), theta(t, 1), theta(t, 2)
    a, d_a, dd_a = alpha(t), alpha(t, 1), alpha(t, 2)

    n = np.vstack((np.cos(th), np.sin(th)))
    d_n = np.vstack((-np.sin(th), np.cos(th)))
    dd_n = np.vstack((-np.cos(th), -np.sin(th)))
    d_s = M.T @ (d_a * n + a * d_n * d_th)
    dd_s = M.T @ (dd_a * n + 2 * d_a * d_n * d_th + a * (dd_n * d_th ** 2 + d_n * dd_th))

    return np.abs(d_s[0] * dd_s[1] - d_s[1] * dd_s[0]) / (d_s[0] ** 2 + d_s[1] ** 2) ** 1.5


def curvature_planar(t_phys, s1, s2):
    ds1, ds2 = s1(t_phys, 1), s2(t_phys, 1)
    dds1, dds2 = s1(t_phys, 2), s2(t_phys, 2)

    return np.abs(ds1 * dds2 - ds2 * dds1) / (ds1 ** 2 + ds2 ** 2) ** 1.5