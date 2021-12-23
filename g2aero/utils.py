import numpy as np


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


def global_blade_coordinates(xyz_local):

    n_shapes, n_landmarks, _ = xyz_local.shape
    xyz_global = np.empty((n_shapes, n_landmarks, 3))
    for i, xyz in enumerate(xyz_local):
        xyz_global[i] = np.c_[xyz[:, 1], -xyz[:, 0], xyz[:, 2]]
    return