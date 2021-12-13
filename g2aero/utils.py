import numpy as np
from scipy.optimize import lsq_linear
from scipy.special import comb


def get_landmarks(xy, name, n_landmarks=401, cst_order=8, add_gap=False):

    if name in ['circular', 'Cylinder', 'Cylinder1', 'Cylinder2']:
        n1, n2 = 0.5, 0.5
    else:
        n1, n2 = 0.5, 1.0

    # Normalize coordinates (x from 0 to 1) to rid of the rounding error
    x_min, x_max = np.min(xy[:, 0]), np.max(xy[:, 0])
    xy[:, 0] = (xy[:, 0] - x_min) / (x_max - x_min)
    xy[:, 1] = xy[:, 1] / (x_max - x_min)
    if not np.allclose(x_min, 0) or not np.allclose(x_max, 1):
        print('WARNING!: Airfoil shape is not normalized properly', x_min, x_max, (x_max - x_min))

    le_ind = np.argmin(xy[:, 0])  # Leading edge index
    y1_avg = np.average(xy[:le_ind, 1])  # Determine orientation of the airfoil shape
    if y1_avg > 0:
        xy = xy[::-1]  # Flip such that the pressure side is always first

    # split into upper and lower parts
    xy_upper, xy_lower = xy[le_ind:], xy[:le_ind]
    # make tailedge gap
    if add_gap:
        te_lower_add = np.maximum(np.abs(xy[0, 1]), add_gap/2) - np.abs(xy[0, 1])
        te_upper_add = np.maximum(xy[-1, 1], 0.002) - xy[-1, 1]
        xy_upper[:, 1] += xy_upper[:, 0] * te_upper_add
        xy_lower[:, 1] -= xy_lower[:, 0] * te_lower_add

    # calculate cst coefficients
    cst_upper = calc_cst_param(xy_upper[:, 0], xy_upper[:, 1], n1, n2, cst_order)
    cst_lower = calc_cst_param(xy_lower[:, 0], xy_lower[:, 1], n1, n2, cst_order)
    cst = np.r_[cst_lower, cst_upper]

    n_half = int(n_landmarks / 2)
    x_c = -np.cos(np.linspace(0, np.pi, n_half + 1)) * 0.5 + 0.5
    xy_landmarks = from_cst_parameters(x_c, cst_lower, cst_upper, n1, n2)

    return xy_landmarks, cst


def from_cst_parameters(xinp, cst_lower, cst_upper, n1, n2):
    """ Compute landmark coordinates for the airfoil

    :param xinp: (np.ndarray): Non-dimensional x-coordinate locations
    :param cst_lower: (np.ndarray): cst parameters for lower part
    :param cst_upper: (np.ndarray): cst parameters for upper part
    :param n1: (double): normal coord
    :param n2: (double): normal coord
    :param te_lower: (double): Trailing edge thickness above camber line
    :param te_upper: (double): Trailing edge thickness below camber line
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




def add_tailedge_gap(xy, gap):
    # split into upper and lower parts
    le_ind = np.argmin(xy[:, 0])  # Leading edge index
    xy_upper, xy_lower = xy[le_ind:], xy[:le_ind]
    xy_upper[:, 1] += xy_upper[:, 0] * gap/2
    xy_lower[:, 1] -= xy_lower[:, 0] * gap/2
    return np.r_[xy_lower, xy_upper]


def position_airfoil(shape):
    y_shift = shape[0, 1]
    shape[:, 1] -= y_shift
    x_shift = shape[0, 0]

    shape[:, 0] -= x_shift
    min_angle = 0
    min_x = np.min(shape[:, 0])
    for angle in np.linspace(-np.pi / 18, np.pi / 18, 1000):
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        shape_tmp = shape @ R.T
        if np.min(shape_tmp[:, 0]) < min_x:
            min_x = np.min(shape_tmp[:, 0])
            min_angle = angle
    R = np.array([[np.cos(min_angle), -np.sin(min_angle)], [np.sin(min_angle), np.cos(min_angle)]])
    shape = shape @ R.T
    shape[:, 0] = (shape[:, 0] - min_x) / -min_x

    return shape