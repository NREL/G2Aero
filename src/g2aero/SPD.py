# Intrinsic maps and routines for the symmetric positive definite (SPD) matrices

import numpy as np

def polar_decomposition(X_phys):
    """

    :param X_phys:(n_shapes, n_landmarks, 2) array of physical coordinates defining shapes
    :return: X_grassmann, P, b, such that X_phys = X_grassmann @ P + b.
    """
    X_phys = np.asarray(X_phys)
    if len(X_phys.shape) < 3:
        X_phys = np.expand_dims(X_phys, axis=0)

    n_shapes, n_landmarks, _ = X_phys.shape
    X_grassmann = np.empty_like(X_phys)
    P = np.empty((n_shapes, 2, 2))
    b = np.empty((n_shapes, 2))

    for i, xy in enumerate(X_phys):
        center_mass = np.mean(xy, axis=0)
        U, D, _ = np.linalg.svd((xy - center_mass).T, full_matrices=False)
        P[i] = (U * D) @ U.T
        Pinv = (U / D) @ U.T
        b[i] = center_mass
        X_grassmann[i] = (xy - center_mass) @ Pinv
    
    if n_shapes == 1:
        return X_grassmann.squeeze(axis=0), P.squeeze(axis=0), b.squeeze(axis=0)
    return X_grassmann, P, b


def exp(t, P, S):
    """SPD Exponential. (Fletcher, P. T., & Joshi, S. 2004)
    
    :param t: scalar > 0, how far in given direction to move (if t=0, exp(P, log_map) = P)
    :param P: (2, 2) array = starting point \in S_++^2
    :param S: (2, 2) array = direction in tangent space \in T_P S_++^2
    :return: (2, 2) array  = end point 
    """
    Lambda, u = np.linalg.eig(P)
    g = u * np.sqrt(Lambda)  # g = u @ np.diag(np.sqrt(Lambda))
    ginv = np.linalg.inv(g)
    Y = ginv @ S @ ginv.T
    Sigma, v = np.linalg.eig(Y)
    gv = g@v
    return (gv * np.exp(t*Sigma)) @ gv.T

def log(P, D):
    """SPD Logarithmic mapping (inverse mapping of exponential map).
    (Fletcher, P. T., & Joshi, S. 2004)

    Calculates direction S (tangent vector \Delta) from P to D in tangent subspace.

    :param P: (2, 2) array = start point  \in S_++^2
    :param D: (2, 2) array = end point \in S_++^2
    :return: (2, 2) array = direction in tangent space (tangent vector \Delta) \in T_P S_++^2
    """
    Lambda, u = np.linalg.eig(P)
    g = u * np.sqrt(Lambda)  # g = u @ np.diag(np.sqrt(Lambda))
    ginv = np.linalg.inv(g)
    Y = ginv @ D @ ginv.T
    Sigma, v = np.linalg.eig(Y)
    gv = g@v
    return (gv * np.log(Sigma)) @ gv.T


def Karcher(data, eps=1e-8, max_steps=20):
    """Karcher mean for given shapes.

    Calculated Karcher mean for given data by minimizing the sum 
    of squared (Riemannian) distances to all points in the data 
    (Fletcher, Lu, and Joshi 2003)

    :param data: (n_elements, 2, 2) array of given data
    :param max_steps: maximum number of iterations to converge
    :return: (2, 2) array defining Karcher mean
    """
    data = np.asarray(data)
    log_directions = np.zeros_like(data)
    mu_karcher = data[0]
    print('Karcher mean convergence:')
    for j in range(max_steps):
        for i, point in enumerate(data):
            if not (i == 0 and j == 0):
                log_directions[i] = log(mu_karcher, point)
        V = np.mean(log_directions, axis=0)
        mu_karcher = exp(1, mu_karcher, V)
        print(f'||V||_F = {np.linalg.norm(V, ord="fro")}')
        if np.linalg.norm(V, ord='fro') <= eps:
            return mu_karcher
    print('WARNING: Maximum count reached...')
    return mu_karcher

def PGA(mu, data, n_coord=None):
    """Principal Geodesic Analysis (PGA).

    Principal Geodesic Analysis (PGA), a generalization of Principal Component Analysis (PCA) over Riemannian manifolds.
    PGA is a data-driven approach that determines principal components as elements in a central tangent space,
    given a data set represented as elements in asmooth manifold.

    :param mu: (2, 2) array defining Karcher mean
    :param data: (n_points, 2, 2) given data
    :param n_coord: dimension of resulting PGA space (if None n_coord=4)
    :return: Vh is principal basis transposed ((n_coord*2)x(n_coord*2)),
             t are given elements in principal coordinates,
             S is corresponding singular values,
    """
    data, mu = np.asarray(data), np.asarray(mu)
    n_points, dim, _ = data.shape
    # get tangent directions from mu to each point (each direction is set of (2, 2)-dimensional vectors)
    # flatten each (2, 2) vector into (n_landmark*dim) vector for later svd decomposition
    H = np.zeros((n_points, dim * dim))
    for i, point in enumerate(data):
        H[i] = log(mu, point).flatten()
    # Principal Geodesic Analysis (PGA)
    # # columns of V are principal directions/axis
    U, S, Vh = np.linalg.svd(H, full_matrices=False)
    # projection of the data on principal axis (H@V = U@S@Vh@V = U@S))
    t = U*S  # shape(n_shapes, 2*2)
    if n_coord is None or n_coord > 2*2:
        n_coord = 4
    return Vh[:n_coord, :], S[:n_coord], t[:, :n_coord]

# def PGA(muX, XD):
#     m = XD.shape[0]
#     n = XD.shape[1]
#     N = XD.shape[2]
#     H = np.zeros((m*n, N))
#     for i in range(N):
#         Hi = Log(muX, XD[:, :, i])
#         H[:, i] = Hi.reshape((m*n,))
    
#     # Principal Geodesic Analysis (PGA)
#     U, S, V = np.linalg.svd(1/np.sqrt(N) * H, full_matrices=False)
#     # compute normal coordiantes
#     t = H.T @ U
#     return U, t, S