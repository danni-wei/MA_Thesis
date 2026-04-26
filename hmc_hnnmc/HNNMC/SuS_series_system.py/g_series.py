# file: g_series.py
import numpy as np

def g_fun(X):
    """
    X: array-like, shape (n_samples, 2) 或 (2,)
    返回: G(x) = min{g1,g2,g3,g4}，逐样本
    """
    X = np.atleast_2d(X)
    x1 = X[:, 0]
    x2 = X[:, 1]
    u = x1 - x2
    v = (x1 + x2)/np.sqrt(2.0)

    g1 = 0.1*(u**2) - v + 3.0
    g2 = 0.1*(u**2) + v + 3.0
    g3 = u + 7.0/np.sqrt(2.0)
    g4 = -u + 7.0/np.sqrt(2.0)

    G = np.minimum.reduce([g1, g2, g3, g4])
    return G if len(G) > 1 else G[0]