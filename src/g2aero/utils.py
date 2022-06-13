import numpy as np


def add_tailedge_gap(xy, gap):
    # split into upper and lower parts
    le_ind = np.argmin(xy[:, 0])  # Leading edge index
    xy_upper, xy_lower = xy[le_ind:], xy[:le_ind]
    xy_upper[:, 1] += xy_upper[:, 0] * gap/2
    xy_lower[:, 1] -= xy_lower[:, 0] * gap/2
    return np.r_[xy_lower, xy_upper]


def remove_tailedge_gap(xy):
    le_ind = np.argmin(xy[:, 0])  # Leading edge index
    # split int upper and lower parts
    xy_upper, xy_lower = xy[le_ind:], xy[:le_ind]
    te_lower, te_upper = xy[0, 1], xy[-1, 1]
    xy_upper[:, 1] -=  xy_upper[:, 0] * te_upper
    xy_lower[:, 1] -=  xy_lower[:, 0] * te_lower
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


def find_selfintersect(shape):
    """Find all self intersection of shape. 

    :param X: (n_landmarks, 2) discrete planar curve with n landmarks representing a shape
    :return: bool  = if intersection exist
            (n, 2) array with index p and q of intersecting intervals (n=number of intersections)
            (n, 2) array of intersection locations (n=number of intersections)
    """

    # The last point must be duplicated as the first point for closed curves
    if False in shape[-1] == shape[0]:
        shape = np.append(shape, shape[0])

    n_landmarks, _ = shape.shape
    
    diffX = np.diff(shape, axis=0)
    ind, Xint = [], []
    for p in range(n_landmarks-2):
        a, c = diffX[p]
        for q in range(p+1, n_landmarks-1):
            b, d = -diffX[q]
            det = a*d-b*c
            bb = shape[q] - shape[p]
            t1 = (d*bb[0] - b*bb[1])/det
            t2 = (-c*bb[0]+ a*bb[1])/det
            if  10e-8 < t1 < 1-10e-8:
                if 10e-8 < t2 < 1-10e-8:
                    ind.append([p, q])
                    Xint.append(t1*shape[p+1] + (1-t1)*shape[p])
    return len(ind) > 0, np.asarray(ind), np.asarray(Xint)


def check_selfintersect(shape):
    """Check if self intersection in a shape exist. 

    :param X: (n_landmarks, 2) discrete planar curve with n landmarks representing a shape
    :return: bool  = if intersection exist
    """
    if False in shape[-1] == shape[0]:
        shape = np.append(shape, shape[0])

    n_landmarks, _ = shape.shape
    
    diffX = np.diff(shape, axis=0)
    for p in range(n_landmarks-2):
        a, c = diffX[p]
        for q in reversed(range(p+1, n_landmarks-1)):
            b, d = -diffX[q]
            det = a*d-b*c
            bb = shape[q] - shape[p]
            t1 = (d*bb[0] - b*bb[1])/det
            t2 = (-c*bb[0]+ a*bb[1])/det
            if  10e-8 < t1 < 1-10e-8:
                if 10e-8 < t2 < 1-10e-8:
                    return True
    return False
