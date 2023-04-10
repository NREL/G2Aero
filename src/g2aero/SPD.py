# Intrinsic maps and routines for the symmetric positive definite (SPD) matrices

import numpy as np

def vec(P):
    """
    Return vector of the upper-triangle elements
    :param P:(n, n) symmetric matrix
    :return: (n*(n+1)/2,1) vectorization of unique entries
    """
    return P[np.triu_indices_from(P)]

def vecinv(p):
    """
    Return symmetric matrix from vectorized form
    :param p:(n*(n+1)/2,1) vector of symmetric matrix entries returned by consistent vectorization
    :return: (n,n) corresponding symmetric matrix
    """
    # compute matrix dimension (solve quadratic equation n**2+n-2*len(p)=0) 
    p = np.array(p).flatten()
    discrimenant = 1+4*2*len(p)
    root = (-1 + np.sqrt(discrimenant))/2
    n = int(root)
    assert(root%n == 0), "Vector with such length can't be converted to symmetric square matrix"

    P = np.zeros((n, n))
    inds = np.triu_indices(n)
    P[inds] = p
    P_upper = np.triu(P, 1)
    out = P + P_upper.T
    return out


# def vec(P):
#     """
#     :param P:(n, n) symmetric matrix
#     :return: (n*(n+1)/2,1) vectorization of unique entries
#     """
#     n = P.shape[1]
#     vec = np.zeros((int(n*(n+1)/2),1))
#     k = 0
#     for i in range(1,n+1):
#         vec[int(k):int(i+k)] = P[0:int(i),int(i-1)].reshape((i,1))
#         k += i
#     return vec


# def vecinv(p):
#     """

#     :param p:(n*(n+1)/2,1) vector of symmetric matrix entries returned by consistent vectorization
#     :return: (n,n) corresponding symmetric matrix
#     """
#     # compute matrix dimension
#     roots = [(-0.5 + np.sqrt(0.25 + 2*len(p))), \
#              (-0.5 - np.sqrt(0.25 + 2*len(p)))]
#     n = int(np.max(roots))
#     # assign diagonal entries
#     i_d = np.array(vec(np.eye(4)),dtype=bool)
#     D = np.diag(p[i_d])
#     # collect the remaining off-diagonal entries
#     OD = p[~i_d]
    
#     # assign off-diagonal entries
#     P = np.zeros((n,n))
#     k = 0
#     for i in range(1,n):
#         for j in range(0,i):
#                 P[j,i] = OD[k]
#                 k+=1
#     return P + P.T + D


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
    """SPD Logarithmic mapping (inverse mapping of exponential map). (Fletcher, P. T., & Joshi, S. 2004)

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
    """Karcher mean for given shapes. (Fletcher, Lu, and Joshi 2003)

    Calculated Karcher mean for given data by minimizing the sum 
    of squared (Riemannian) distances to all points in the data  

    :param data: (n_elements, 2, 2) array of given data
    :param eps: float number: convergence criterion 
    :param max_steps: maximum number of iterations to converge
    :return: (2, 2) array defining Karcher mean
    """
    data = np.asarray(data)
    log_directions = np.zeros_like(data)
    mu_karcher = data[0]
    print('SPD manifold')
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


def tangent_space(mu, data):
    data, mu = np.asarray(data), np.asarray(mu)
    n_points, n, _ = data.shape
    # get tangent directions from mu to each point
    # vectorize matrices for later svd decomposition
    H = np.zeros((n_points, int(n*(n+1)/2)))
    for i, element in enumerate(data):
        H[i] = vec(log(mu, element)).flatten()
    return H


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
    # get tangent vectors from mu to each point
    H = tangent_space(mu, data)
    # Principal Geodesic Analysis (PGA)
    # # columns of V are principal directions/axis
    U, S, Vh = np.linalg.svd(H, full_matrices=False)
    # projection of the data on principal axis (H@V = U@S@Vh@V = U@S))
    t = U*S
    if n_coord is None or n_coord > H.shape[1]:
        n_coord = H.shape[1]
    return Vh[:n_coord, :], S[:n_coord], t[:, :n_coord]


def perturb_mu(Vh, mu, perturbation):
    """Given element Karcher mean, perturbs it in given direction by a given amount.

    :param Vh: (n_landmarks*2 - 4 , n_landmarks*2 - 4) array of PGA basis vectors transposed
    :param mu: (n_landmarks, 2) array of Karcher mean (elenemt on Grassmann)
    :param perturbation: (n_modes,) array of amount of perturbations in pga coordinates
    :return: (n_landmarks, 2) array of perturbed element on Grassmann
    """
    perturbation = np.asarray(perturbation).reshape(1, -1)
    direction = vecinv(perturbation@Vh)
    spd_matrix = exp(1, mu, direction)
    return spd_matrix 
