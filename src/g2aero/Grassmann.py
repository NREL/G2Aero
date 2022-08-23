# Intrinsic maps and routines for the Grassmannian
import numpy as np


def procrustes(X, Y):
    """Procrustes clustering match the shapes via Procrustes analysis (Gower 1975).

    This function calculates rotation that can be applied to shapes for matching
    (rotation does not fundamentally modify the elements in the Grassmannian).

    :param X: (n_landmarks, 2) array defining shape 1
    :param Y: (n_landmarks, 2) array defining shape 2
    :return: (2, 2) array of rotation matrix
    """
    X = np.asarray(X)
    U, _, Vh = np.linalg.svd(X.T @ Y)
    R = U @ Vh
    return R


def landmark_affine_transform(X_phys):
    """Shift and scale all shapes using Landmark-Affine standardization (Bryner, 2D affine and projective spaces).

    LA-standardization  normalizes  the  shape  such that  it  has  zero  mean  (translation  invariance)  and
    sample covariance proportional to I2 over the n discrete boundary landmarks defining the shape.

    :param X_phys:(n_shapes, n_landmarks, 2) array of physical coordinates defining shapes
    :return: X_grassmann, M, b, such that X_phys = X_grassmann @ M + b.
    """
    X_phys = np.asarray(X_phys)
    if len(X_phys.shape) < 3:
        X_phys = np.expand_dims(X_phys, axis=0)

    n_shapes, _, _ = X_phys.shape
    X_grassmann = np.empty_like(X_phys)
    M = np.empty((n_shapes, 2, 2))
    b = np.empty((n_shapes, 2))

    for i, xy in enumerate(X_phys):
        b[i] = np.mean(xy, axis=0) # center of mass
        U, D, Vh = np.linalg.svd((xy - b[i]).T, full_matrices=False)
        M[i] = np.diag(D)@U.T
        X_grassmann[i] = Vh.T
    # Procrustes problem
    if n_shapes > 1:
        for i in reversed(range(1, n_shapes)):
            R = procrustes(X_grassmann[i - 1], X_grassmann[i])
            M[i - 1] =  R.T @ M[i - 1] 
            X_grassmann[i - 1] = X_grassmann[i - 1] @ R

    if n_shapes == 1:
        return X_grassmann.squeeze(axis=0), M.squeeze(axis=0), b.squeeze(axis=0)
    return X_grassmann, M, b

def exp(t, X, direction):
    """Exponential mapping (Grassmannian geodesic).
    
    :param t: scalar > 0, how far in given direction to move (if t=0, exp(X, log_map) = X)
    :param X: (n_landmarks, 2) array defining starting point of geodesic on Grassmann
    :param direction: (n_landmarks, 2) array defining direction in tangent space (tangent vector \Delta)
    :return: (n_landmarks, 2) array defining end point on Grassmann
    """
    U, S, Vh = np.linalg.svd(direction, full_matrices=False)
    exp_map = np.hstack((X @ Vh.T, U)) @ np.vstack((np.diag(np.cos(t *
                                                                   S)), np.diag(np.sin(t * S)))) @ Vh
    return exp_map


def log(X, Y):
    """Logarithmic mapping (inverse mapping of exponential map).
    Algorithm 11 (Zimmermann, 2019)
    Calculate logarithmic map log_X(Y) (inverse mapping of exponential map).
    Calculates direction(tangent vector \Delta) from X to Y in tangent subspace.

    :param X: (n_landmarks, 2) array defining start point of geodesic on Grassmann
    :param Y: (n_landmarks, 2) array defining end point of geodesic on Grassmann
    :return: (n_landmarks, 2) array defining direction in tangent space (tangent vector \Delta)
    """
    X, Y = np.asarray(X), np.asarray(Y)
    # Procrutes match
    R = procrustes(Y, X)
    Y = Y @ R
    # Projection to horizontal space
    ortho_projection = np.eye(len(X)) - X @ X.T
    L = ortho_projection @ Y 
    Q, Sigma, Vh = np.linalg.svd(L, full_matrices=False)
    log_map = (Q * np.arcsin(Sigma)) @ Vh
    return log_map


def distance(X, Y):
    """Geodesic distance on Grassmannian.

    Geodesic distance on Grassmannian is defined by the principal angles
    between the subspaces spanned by the columns of X and Y, denoted by
    span(X) and span(Y). The cosines of the principal angles theta1 and
    theta2 between span(X) and snap(Y) are the singular values of X.T@Y.
    That is, X.T@Y = U D V.T,  where D = diag(cos(theta1), cos(theta2)).
    The distance between two shapes is then defined as dist = sqrt(theta1**2+theta2**2).

    :param X: (n_landmarks, 2) array defining first shape
    :param Y: (n_landmarks, 2) array defining second shape
    :return: distance between two shapes on Grassmannian
    """
    X, Y = np.asarray(X), np.asarray(Y)
    dim = X.shape[1]  # get dimensions(must be identical for Y)
    if X.shape != Y.shape:
        raise ValueError('Input matrices must have the same number of columns')
    check_orthogonality(X)
    check_orthogonality(Y)

    D = np.linalg.svd(X.T @ Y, compute_uv=False)  # compute singular values

    # to avoid nan if value is close to 1.0
    # (keeps D <= 1.0 for arccos and makes distance = 0 between really similar shapes)
    ind_of_ones = np.array([i for i in range(dim) if np.isclose(D[i], 1.0)], dtype=int)
    D[ind_of_ones] = 1.0

    theta = np.arccos(D)  # compute principal angles
    dist = np.sqrt(np.sum(np.real(theta) ** 2))

    return dist


def Karcher(shapes, eps=1e-8, max_steps=20):
    """Karcher mean for given shapes.

    Calculated Karcher mean for given shapes (elements on Grassmann) by
    minimizing the sum of squared (Riemannian) distances to all shapes in the data 
    (Fletcher, Lu, and Joshi 2003)

    :param shapes: (n_shapes, n_landmarks, 2) array of given shapes (Grassmann elements)
    :param max_steps: maximum number of iterations to converge
    :return: (n_landmarks, 2) array defining Karcher mean (element on Grassmann)
    """
    shapes = np.asarray(shapes)
    # check for orthogonality (for Grassmann elements)
    check_orthogonality(shapes[0])
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
        if np.linalg.norm(V, ord='fro') <= eps:
            return mu_karcher
    print('WARNING: Maximum count reached...')
    return mu_karcher


def PGA(mu, shapes_gr, n_coord=None):
    """Principal Geodesic Analysis (PGA).

    Principal Geodesic Analysis (PGA), a generalization of Principal Component Analysis (PCA) over Riemannian manifolds.
    PGA is a data-driven approach that determines principal components as elements in a central tangent space,
    given a data set represented as elements in asmooth manifold.

    :param mu: (n_landmarks, 2) array defining Karcher mean (element on Grassmann)
    :param shapes_gr: (n_shapes, n_landmarks, 2) given shapes (elements on Grassmann)
    :param n_coord: dimension of resulting PGA space (if None n_coord=n_landmarks)
    :return: Vh is principal basis transposed ((n_coord*2)x(n_coord*2)),
             t are given elements in principal coordinates,
             S is corresponding singular values,
    """
    shapes_gr, mu = np.asarray(shapes_gr), np.asarray(mu)
    check_orthogonality(shapes_gr[0])
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
    if n_coord is None or n_coord > 2*n_landmarks:
        n_coord = 2*n_landmarks
    return Vh[:n_coord, :], S[:n_coord], t[:, :n_coord]


def PGA_modes(PGA_directions, mu, scale=1, sub=10):
    """
    Moves given element on Grassmann in each of given directions.

    :param PGA_directions: directions to perturb element mu
    :param mu: element on grassmann
    :param sub: subset of directions (first sub directions)
    :return: elements on Grassmann (perturbed from mu in PGA_directions)
    """
    PGA_directions = PGA_directions.reshape(len(PGA_directions), -1, 2)
    PGA_fwd = np.zeros_like(PGA_directions)
    for i in range(min(sub, len(PGA_directions))):
        PGA_fwd[i] = exp(scale, mu, PGA_directions[i])
    return PGA_fwd


def get_PGA_coordinates(shapes_gr, mu, V, n_modes=None):
    """
    Get PGA coordinates of given standardized shapes (elements of Grassmann) and
    PGA space defined by Karcher mean and basis vectors.

    :param shapes_gr: (n_shapes, n_landmarks, 2) array of given standardized shapes (element of Grassmann)
    :param mu: (n_landmarks, 2) array of Karcher mean (origin of PGA space)
    :param V: (n_coord*2, n_coord*2) array of PGA basis vectors
    :return: (n_shapes, n_coord) array of PGA coordinates for given shapes
    """
    shapes_gr, mu = np.asarray(shapes_gr), np.asarray(mu)
    if len(shapes_gr.shape) < 3:
        shapes_gr = np.expand_dims(shapes_gr, axis=0)
    n_shapes, n_landmarks, dim = shapes_gr.shape
    if n_modes is None or n_modes > 2*n_landmarks:
        n_modes = 2*n_landmarks
    # get tangent directions from mu to each point (each direction is set of (n_landmark, dim)-dimensional vectors)
    # flatten each (n_landmark, dim) vector into (n_landmark*dim) vector
    H = np.zeros((n_shapes, n_landmarks * dim))
    for i, shape in enumerate(shapes_gr):
        H[i] = log(mu, shape).flatten()
    coords = H@V[:, :n_modes]
    return coords


def perturb_gr_shape(Vh, mu, perturbation):
    """Given element Karcher mean, perturbs it in given direction by a given amount.

    :param Vh: (n_landmarks*2, n_landmarks*2) array of PGA basis vectors transposed
    :param mu: (n_landmarks, 2) array of Karcher mean (elenemt on Grassmann)
    :param perturbation: (n_modes,) array of amount of perturbations in pga coordinates
    :return: (n_landmarks, 2) array of perturbed element on Grassmann
    """
    perturbation = perturbation.reshape(1, -1)
    n_modes = perturbation.shape[1]
    direction = perturbation@Vh[:n_modes]
    direction = direction.reshape(-1, 2)
    perturbed_shape = exp(1, mu, direction)
    return perturbed_shape


def parallel_translate(start, end_direction, vector):
    """ Parallel translation of a vector along geodesic from start point to end point
    Edelman et al. (Theorem 2.4, pp. 321).

    :param start: (n_landmarks, 2) array defining start point element on Grassmann
    :param end_direction: (n_landmarks, 2) array defining direction to the end point element on Grassmann (log map)
    :param vector: (n_landmarks, 2) array defining vector to be translated
    :return: (n_landmarks, 2) array defining translated vector
    """
    n_landmarks = end_direction.shape[0]
    U, S, Vh = np.linalg.svd(end_direction, full_matrices=False)
    exp_map = np.hstack((start @ Vh.T, U)) @ np.vstack((-np.diag(np.sin(S)), np.diag(np.cos(S)))) @ U.T \
              + np.eye(n_landmarks) - U@U.T
    return exp_map @ vector

def check_orthogonality(X):
    if not np.allclose(X.T @ X, np.eye(X.shape[1])):
        raise ValueError(f'Input does not constitute an element of the Grassmannian: X.T @ X = {X.T @ X}')
