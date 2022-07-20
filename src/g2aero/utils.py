import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator


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
    """Find all self intersections of shape. 

    :param X: (n_landmarks, 2) discrete planar curve with n landmarks representing a shape
    :return: bool  = if intersection exist
            (n, 2) array with index p and q of intersecting intervals (n=number of intersections)
            (n, 2) array of intersection locations (n=number of intersections)
    """

    n_landmarks, _ = shape.shape
    
    diffX = np.diff(shape, axis=0)
    ind, Xint = [], []
    for p in range(n_landmarks-2):
        a, c = diffX[p]
        for q in range(p+1, n_landmarks-1):
            b, d = -diffX[q]
            det = a*d-b*c
            if det!=0:
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
    n_landmarks, _ = shape.shape
    
    diffX = np.diff(shape, axis=0)
    for p in range(n_landmarks-2):
        a, c = diffX[p]
        for q in reversed(range(p+1, n_landmarks-1)):
            b, d = -diffX[q]
            det = a*d-b*c
            if det!=0:
                bb = shape[q] - shape[p]
                t1 = (d*bb[0] - b*bb[1])/det
                t2 = (-c*bb[0]+ a*bb[1])/det
                if  10e-8 < t1 < 1-10e-8:
                    if 10e-8 < t2 < 1-10e-8:
                        return True
    return False


def arc_distance(xy):
    dist = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    t = np.cumsum(dist) / np.sum(dist)
    return np.hstack(([0.0], t))

########################################################################################
# Functions to calculate some characteristics of a shape
########################################################################################
def calc_curvature(xy):
    t_phys = arc_distance(xy)
    
    s1 = CubicSpline(t_phys, xy[:, 0])
    s2 = CubicSpline(t_phys, xy[:, 1])
    t_tmp = np.linspace(0, 1, 100000)
    
    ds1, ds2 = s1(t_tmp, 1), s2(t_tmp, 1)
    dds1, dds2 = s1(t_tmp, 2), s2(t_tmp, 2)
    return ds1 * dds2 - ds2 * dds1 / (ds1 ** 2 + ds2 ** 2) ** 1.5


# def calc_camber(xy):
#     t_phys = arc_distance(xy)
    
#     s1 = CubicSpline(t_phys, xy[:, 0])
#     s2 = CubicSpline(t_phys, xy[:, 1])
    
#     t_new = np.linspace(0, 1, 100000)
#     xy = np.vstack((s1(t_new), s2(t_new))).T
    
#     le_ind = np.argmin(xy[:, 0])  # Leading edge index
#     xy_upper, xy_lower = xy[le_ind:], xy[:le_ind+1][::-1]  # split int upper and lower parts
    
#     upper = CubicSpline(xy_upper[:, 0], xy_upper[:, 1], )
#     lower = CubicSpline(xy_lower[:, 0], xy_lower[:, 1])
    
#     x_min = max(xy_upper[-1, 0], xy_lower[0, 0])
#     x_camber = np.linspace(x_min, 1, 10000)
#     camber = (lower(x_camber) + upper(x_camber))/2
    
#     return x_camber, camber

# def d2_camber(xy):

#     # increase number of landmarks
#     t_phys = arc_distance(xy)
    
#     s1 = CubicSpline(t_phys, xy[:, 0])
#     s2 = CubicSpline(t_phys, xy[:, 1])
    
#     t_new = np.linspace(0, 1, 100000)
#     xy = np.vstack((s1(t_new), s2(t_new))).T
    
#     le_ind = np.argmin(xy[:, 0])  # Leading edge index
#     xy_upper, xy_lower = xy[le_ind:], xy[:le_ind+1][::-1]  # split int upper and lower parts
    
#     # d2_upper = CubicSpline(xy_upper[:, 0], xy_upper[:, 1]).derivative(2)
#     # d2_lower = CubicSpline(xy_lower[:, 0], xy_lower[:, 1]).derivative(2)
#     d2_upper = PchipInterpolator(xy_upper[:, 0], xy_upper[:, 1]).derivative(2)
#     d2_lower = PchipInterpolator(xy_lower[:, 0], xy_lower[:, 1]).derivative(2)
    
#     x_camber = np.linspace(0.002, 1, 10000)
#     d2_camber = (d2_lower(x_camber) + d2_upper(x_camber))/2
    
#     return d2_camber


def calc_area(xy, a=0, b=1):
    """ Function calculates area inside of an airfoil shape xy
    (x coordinates are normalized by chord length) from left 
    boundary a to right boundary b using discrete representation 
    of a shape and trapesoidal rule. 
        E.g. to calculate area inside of an airfoil from 25% chord 
    to 45% chord one can use `calc_area(xy, a=0.25, b=0.45)`

    :param xy: (n, 2) array of given shape
    :param a: left boundary
    :param b: right boundary
    :return: scalar area
    """
    le_ind = np.argmin(xy[:, 0])  # Leading edge index
    xy_upper, xy_lower = xy[le_ind:], xy[:le_ind+1][::-1]  # split int upper and lower parts
    ind_upper = np.where((xy_upper[:, 0]>a) & (xy_upper[:, 0]<b))[0]
    ind_lower = np.where((xy_lower[:, 0]>a) & (xy_lower[:, 0]<b))[0]
    area = np.trapz(xy_upper[ind_upper, 1], xy_upper[ind_upper, 0]) - np.trapz(xy_lower[ind_lower, 1], xy_lower[ind_lower, 0])
    return area