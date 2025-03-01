from re import I
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

def tailedge_gap(xy):
    """Calculate tail edge gap size

    :param xy (n, 2) shape coordinates
    :return: TE gap size
    """
    return xy[-1, 1] - xy[0, 1]

def add_tailedge_gap(xy, gap):
    """Adding tail edge gap. 
    Before using need to position airfoil first (chord on x-axis from 0 to 1)

    :param xy: (n, 2) shape coordinate
    :param gap: gap size to be added
    :return: array of shape coordinates with added gap
    """
    # need to position airfoil first
    # split into upper and lower parts
    le_ind = np.argmin(xy[:, 0])  # Leading edge index
    xy_upper, xy_lower = xy[le_ind:], xy[:le_ind]
    xy_upper[:, 1] += xy_upper[:, 0] * gap/2
    xy_lower[:, 1] -= xy_lower[:, 0] * gap/2
    return np.r_[xy_lower, xy_upper]

def add_minimum_gap(xy):
    """Adding minimum TE gap of 2mm.
    Before using need to position airfoil first (chord on x-axis from 0 to 1)

    :param xy: (n, 2)shape coordinates
    :return: array of shape coordinates with added gap
    """
    return add_tailedge_gap(xy, 0.001)


def remove_tailedge_gap(xy):
    """Remove tailedge gap.

    :param xy: (n, 2) shape coordinates
    :return: array of shape coordinates without gap
    """
    le_ind = np.argmin(xy[:, 0])  # Leading edge index
    # split int upper and lower parts
    xy_upper, xy_lower = xy[le_ind:], xy[:le_ind]
    te_lower, te_upper = xy[0, 1], xy[-1, 1]
    xy_upper[:, 1] -=  xy_upper[:, 0] * te_upper
    xy_lower[:, 1] -=  xy_lower[:, 0] * te_lower
    xy_upper[-1] = [1, 0]
    xy_lower[0] = [1, 0]
    return np.r_[xy_lower, xy_upper]


def position_airfoil(shape_inp, rotate=True, LE_cst=False, return_LEind=False):
    """Normalize, rotates and translates given airfoil coordinates 
    so that the leading edge is at (0, 0) and tail edge is at (1, 0).

    :param shape_inp: (n, 2) shape coordinates
    :param rotate: (bool) rotate airfoil to set LE at (0,0), defaults to True
    :param LE_cst: (bool) set true if shape_inp is from CST parametrization to use x=0 point as LE, defaults to False
    :param return_LEind: (bool) return index of leading edge coordinate, defaults to False
    :return: array of shape coordinates 
    """

    shape = np.array(shape_inp)

    # position tail at (0, 0)
    TE = (shape[-1] + shape[0]) / 2
    shape -= TE

    # rotate to position LE at y = 0 axis
    if rotate or return_LEind:
        if LE_cst:
            LE_ind = int(len(shape)/2)
            LE = shape[LE_ind]
            chord = np.linalg.norm(shape[LE_ind])
        else:  # search for LE
            # Calculate distance from TE to every other point
            dist = np.linalg.norm(shape, axis=1)
            LE_ind = np.argmax(dist)
            chord = np.max(dist)

        LE = shape[LE_ind]
        if rotate:
            cos, sin = tuple(LE / chord)
            R = np.array([[-cos, -sin], [sin, -cos]])
            shape = shape @ R.T
    else:
        LE_ind = np.argmin(shape[:, 0])
        LE = shape[LE_ind]
        chord = np.linalg.norm(shape[LE_ind])
        # t_phys = arc_distance(shape)
        # s1 = CubicSpline(t_phys, shape[:, 0], bc_type='natural')
        # t = CubicSpline(shape[:, y], t_phys,  bc_type='natural')
        # t_0 = t()
        # TODO:

    # scale to chord = 1
    shape /= chord 
    shape[:, 0] += 1

    # to avoid numerical difficulties seting LE to (0, 0)
    shape -= shape[LE_ind]

    if return_LEind:
        return shape, LE_ind
    else:
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
    """Calculate distance along the arc of points defining airfoil shape. 
    t in [0, 1]

    :param xy: (n_landmarks, 2) discrete planar curve with n landmarks representing a shape
    :return: (n_landmarks, ) distance along the arc of points defining airfoil shape
    """
    dist = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    t = np.cumsum(dist) / np.sum(dist)
    return np.hstack(([0.0], t))

########################################################################################
# Functions to calculate some characteristics of a shape
########################################################################################
def calc_unit_normal(shape):
    t_phys = arc_distance(shape)
    # tangent vector
    dx = CubicSpline(t_phys, shape[:, 0], bc_type='natural').derivative(nu=1)(t_phys)
    dy = CubicSpline(t_phys, shape[:, 1], bc_type='natural').derivative(nu=1)(t_phys)
    # Rotate this vector 90 degrees
    dx_n = -dy
    dy_n = dx 
    # Scale it to magnitude equal 1
    scale = np.sqrt(dx_n**2 + dy_n**2)
    dx_n /= scale
    dy_n /= scale
    unit_normal = np.array([dx_n, dy_n]).T
    return unit_normal



def calc_curvature(xy):
    """Calculate curvature based on planar respresentation.

    :param xy: (n_landmarks, 2) discrete planar curve with n landmarks representing a shape
    :return: curvature values
    """
    t_phys = arc_distance(xy)
    
    s1 = CubicSpline(t_phys, xy[:, 0])
    s2 = CubicSpline(t_phys, xy[:, 1])
    t_tmp = np.linspace(0, 1, 100000)
    
    ds1, ds2 = s1(t_tmp, 1), s2(t_tmp, 1)
    dds1, dds2 = s1(t_tmp, 2), s2(t_tmp, 2)

    return ds1 * dds2 - ds2 * dds1 / (ds1 ** 2 + ds2 ** 2) ** 1.5

def calc_curvature_at_x(xy):
    """Calculate curvature based on planar respresentation.

    :param xy: (n_landmarks, 2) discrete planar curve with n landmarks representing a shape
    :return: curvature values
    """
    t_phys = arc_distance(xy)
    
    s1 = CubicSpline(t_phys, xy[:, 0])
    s2 = CubicSpline(t_phys, xy[:, 1])
    t_tmp = np.linspace(0, 1, 100000)
    
    ds1, ds2 = s1(t_tmp, 1), s2(t_tmp, 1)
    dds1, dds2 = s1(t_tmp, 2), s2(t_tmp, 2)
    curvature = CubicSpline(t_tmp, ds1 * dds2 - ds2 * dds1 / (ds1 ** 2 + ds2 ** 2) ** 1.5)

    return curvature(t_phys)


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