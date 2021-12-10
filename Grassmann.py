# Intrinsic maps and routines for the Grassmannian
import numpy as np


def procrustes(X, Y):
    """
    Procrustes problem
    :param X:
    :param Y:
    :return:
    """
    U, s, Vh = np.linalg.svd(X.T @ Y)
    R = U @ Vh
    return R


def landmark_affine_transform(xy_array):
    """
    Shift and scale all shapes using Landmark-Affine standardization (Bryner, 2D affine and projective spaces)
    :param xy_array: array of (n, 2) arrays of physical coordinates of shapes
    :return: xy_grassmann, M, b
    """
    n_shapes, n_landmarks, _ = xy_array.shape
    xy_grassmann = np.empty_like(xy_array)
    Minv = np.empty((n_shapes, 2, 2))
    b = np.empty((n_shapes, 2))

    for i, xy in enumerate(xy_array):
        center_mass = np.mean(xy, axis=0)
        U, D, _ = np.linalg.svd(1 / np.sqrt(n_landmarks - 1) * (xy - center_mass).T)
        Minv[i] = np.diag(1. / D) @ U.T
        b[i] = center_mass
        xy_transformed = (xy - center_mass) @ Minv[i].T
        xy_grassmann[i] = 1 / np.sqrt(n_landmarks - 1) * xy_transformed

    # Procrustes problem
    for i in reversed(range(1, n_shapes)):
        M_pr = xy_grassmann[i - 1].T @ xy_grassmann[i]
        U, s, Vh = np.linalg.svd(M_pr)
        R = U @ Vh
        Minv[i - 1] = R.T @ Minv[i - 1]
        xy_grassmann[i - 1] = xy_grassmann[i - 1] @ R

    return xy_grassmann, np.sqrt(n_landmarks - 1) * np.linalg.inv(Minv), b


def exp(t, X, log_map):
    """
    Exponential mapping (Grassmannian geodesic)
    :param t: parameter that controls the location on geodesic (scalar from [0, 1])
    :param X: starting point of geodesic in Grassmannian
    :param log_map: direction (tangent vector \Delta)
    :return:
    """
    U, S, Vh = np.linalg.svd(log_map, full_matrices=False)
    exp_map = np.hstack((X @ Vh.T, U)) @ np.vstack((np.diag(np.cos(t * S)), np.diag(np.sin(t * S)))) @ Vh
    return exp_map


def log(X, Y):
    """
    Calculate logarithmic map log_X(Y) (inverse mapping of exponential map).
    Calculates direction(tangent vector \Delta) from X to Y in tangent subspace.
    :param X: starting point of geodesic on Gr
    :param Y: end point of geodesic on Gr
    :return: direction (tangent vector \Delta)
    """
    X, Y = np.asarray(X), np.asarray(Y)
    ortho_projection = np.eye(len(X)) - X @ X.T
    Delta = ortho_projection @ Y @ np.linalg.inv(X.T @ Y)
    U, S, Vh = np.linalg.svd(Delta, full_matrices=False)
    log_map = U @ np.diag(np.arctan(S)) @ Vh
    return log_map


def distance(X, Y):
    """
    Geodesic distance on Grassmannian is defined by the principal angles
    between the subspaces spanned by the columns of X and Y, denoted by
    span(X) and span(Y). The cosines of the principal angles theta1 and
    theta2 between span(X) and snap(Y) are the singular values of X.T@Y.
    That is, X.T@Y = U D V.T,  where D = diag(cos(theta1), cos(theta2)).
    The distance between two shapes is then defined as dist = sqrt(theta1**2+theta2**2)
    :param X: first shape
    :param Y: second shape
    :return: distance between two shapes on Grassmannian
    """
    X, Y = np.asarray(X), np.asarray(Y)
    dim = X.shape[1]  # get dimensions(must be identical for Y)
    if X.shape != Y.shape:
        raise ValueError('Input matrices must have the same number of columns')
    if not np.allclose(X.T @ X, np.eye(dim)):
        raise ValueError(f'First input does not constitute an element of the Grassmannian: X.T @ X = {X.T @ X}')
    if not np.allclose(Y.T @ Y, np.eye(dim)):
        raise ValueError('Second input does not constitute an element of the Grassmannian')

    D = np.linalg.svd(X.T @ Y, compute_uv=False)  # compute singular values

    # to avoid nan if value is close to 1.0
    # (keeps D <= 1.0 for arccos and makes distance = 0 between really similar shapes)
    ind_of_ones = np.array([i for i in range(dim) if np.isclose(D[i], 1.0)], dtype=int)
    D[ind_of_ones] = 1.0

    theta = np.arccos(D)  # compute principal angles
    dist = np.sqrt(np.sum(np.real(theta) ** 2))

    return dist


def Karcher(shapes, max_steps=20):
    """
    Calculated Karcher mean for given elements on Grassmann
    :param shapes: Given elements
    :param max_steps: maximum number of iterations to converge
    :return: Karcher mean (element on Grassmann)
    """
    shapes = np.asarray(shapes)
    log_directions = np.zeros_like(shapes)
    mu_karcher = shapes[0]
    print('Karcher mean convergence:')
    for j in range(max_steps):
        for i, shape in enumerate(shapes):
            if not (i == 0 and j == 0):
                log_directions[i] = log(mu_karcher, shape)
        V = np.mean(log_directions, axis=0)
        mu_karcher = exp(1, mu_karcher, V)
        print(f'||V||_F = {np.linalg.norm(V, ord="fro")}')
        if np.linalg.norm(V, ord='fro') <= 1e-8:
            return mu_karcher
    print('WARNING: Maximum count reached...')
    return mu_karcher


def PGA(mu, shapes_gr, sub=None):
    """
    Principal Geodesic Analysis
    :param mu: Karcher mean
    :param shapes_gr: elements on Grassmann
    :param sub: return subset of directions (first sub directions)
    :return: U is principal basis, t are given elements in principal coordinates,
             S is corresponding singular values, PGA directions are basis vectors scaled by singular value
    """
    shapes_gr, mu = np.asarray(shapes_gr), np.asarray(mu)
    n_shapes, n_landmarks, dim = shapes_gr.shape
    # get tangent directions from mu to each point (each direction is set of (n_landmark, dim)-dimensional vectors)
    # flatten each (n_landmark, dim) vector into (n_landmark*dim) vector for later svd decomposition
    H = np.zeros((n_shapes, n_landmarks * dim))
    for i, shape in enumerate(shapes_gr):
        H[i] = log(mu, shape).flatten()
    # Principal Geodesic Analysis (PGA)
    # # columns of V are principal directions/axis
    U, S, Vh = np.linalg.svd(H, full_matrices=False)
    # projection of the data on principal axis (H@V = U@S@Vh@V = U@S))
    t = U*S  # shape(n_shapes, n_landmark*dim)
    if sub:
        return Vh[:sub, :], S[:sub], t[:, :sub]
    return Vh, S, t


def PGA_modes(PGA_directions, mu, scale=1, sub=10):
    """
    Moves given element on Grassmann in each of sub given directions.
    :param PGA_directions: directions to perturb element mu
    :param mu: element on grassmann
    :param sub: subset of directions (first sub directions)
    :return: elements on Grassmann (perturbed from mu in PGA_directions)
    """
    PGA_fwd = np.zeros_like(PGA_directions)
    for i in range(min(sub, len(PGA_directions))):
        PGA_fwd[i] = exp(scale, mu, PGA_directions[i])
    return PGA_fwd


def get_PGA_coordinates(shapes_gr, mu, V):
    shapes_gr, mu = np.asarray(shapes_gr), np.asarray(mu)
    n_shapes, n_landmarks, dim = shapes_gr.shape
    # get tangent directions from mu to each point (each direction is set of (n_landmark, dim)-dimensional vectors)
    # flatten each (n_landmark, dim) vector into (n_landmark*dim) vector
    H = np.zeros((n_shapes, n_landmarks * dim))
    for i, shape in enumerate(shapes_gr):
        H[i] = log(mu, shape).flatten()
    coords = H@V
    return coords


def perturb_gr_shape(Vh, mu, perturbation):
    """
    Given element on Grassmann, perturbs it in given direction by a given amount.
    :param Vh: principal basis transposed (shape (n_landmarks*2)x(n_landmarks*2))
    :param mu: mean element
    :param perturbation: amount of perturbations in pga coordinates
    :param scale: scale perturbations
    :return: perturbed element on Grassmann
    """
    direction = perturbation@Vh
    direction = direction.reshape(-1, 2)
    perturbed_shape = exp(1, mu, direction)
    return perturbed_shape


def parallel_translate(start, end_direction, vector, t=1):
    """ Parallel translation of a vector along geodesic from start point to end point
        Edelman et al. (Theorem 2.4, pp. 321)
    :param start:
    :param end_direction:
    :param vector:
    :param t:
    :return:
    """
    n_landmarks = end_direction.shape[0]
    U, S, Vh = np.linalg.svd(end_direction, full_matrices=False)
    exp_map = np.hstack((start @ Vh.T, U)) @ np.vstack((-np.diag(np.sin(t * S)), np.diag(np.cos(t * S)))) @ U.T \
              + np.eye(n_landmarks) - U@U.T
    return exp_map @ vector
