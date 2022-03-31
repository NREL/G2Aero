# Intrinsic maps and routines for the symmetric positive definite matrices

import numpy as np
from scipy.linalg import expm, logm

def Exp(p, X):
    Lambda, u = np.linalg.eig(p)
    g = u @ np.sqrt(np.diag(Lambda))
    ginv = np.linalg.inv(g)
    Y = ginv @ X @ ginv.T
    Sigma, v = np.linalg.eig(Y)
    
    return (g@v) @ expm(np.diag(Sigma)) @ ((g@v).T)

def Log(p, x):
    Lambda, u = np.linalg.eig(p)
    g = u @ np.sqrt(np.diag(Lambda))
    ginv = np.linalg.inv(g)
    Y = ginv @ x @ ginv.T
    Sigma, v = np.linalg.eig(Y)
    
    return (g@v) @ logm(np.diag(Sigma)) @ ((g@v).T)

def Karcher(XD):
    # infer dimensionality from data
    m = XD.shape[0]
    n = XD.shape[1]
    N = XD.shape[2]
    
    # initialize
    count = 0
    V = np.ones((m, n))
    muX = XD[:,:,int(N/2)].reshape((m, n))
    L = np.zeros((m, n, N))
    
    print('Karcher mean convergence:')
    while np.linalg.norm(V, ord='fro') >= 1e-8:
        for i in range(N):
            if (count == 0) & (i == 0):
                L[:,:,0] = np.zeros((m,n))
            else:
                L[:,:,i] = Log(muX, XD[:,:,i].reshape((m,n)))
        V = 1/N*np.sum(L, axis=2)
        muX = Exp(muX, V)
        count += 1
        print('||V||_F = ', np.linalg.norm(V, ord='fro'))
        if count > 20:
            print('WARNING: Maximum count reached...')
            break
    return muX

def PGA(muX, XD):
    m = XD.shape[0]
    n = XD.shape[1]
    N = XD.shape[2]
    H = np.zeros((m*n, N))
    for i in range(N):
        Hi = Log(muX, XD[:, :, i])
        H[:, i] = Hi.reshape((m*n,))
    
    # Principal Geodesic Analysis (PGA)
    U, S, V = np.linalg.svd(1/np.sqrt(N) * H, full_matrices=False)
    # compute normal coordiantes
    t = H.T @ U
    return U, t, S