import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def smce(Y, lambda_val=10, KMax=None, dim=2, n=1, gtruth=[], verbose=True):
    """
    This function performs SMCE clustering and embedding
    Parameters:
    -----------
    Y: 
        DxN matrix of N data points in the D-dimensional space
    lambda_val, KMax: 
        see smce_optimization
    gtruth: 
        ground truth cluster labels, if available. For computing the misclassification rate.
        
    Return:
    -----------
    Yc: 
        a dictionary where key is the cluster id and the value is the corresponding 
        dxN matrix of data points in this cluster in the reduced dimensional space
    Yj: 
        dxN cluster embemdding
    clusters: 
        a list of cluster labels
    missrate: 
        misclassification rate if gtruth is provided 
    """
    # solve the sparse optimization program
    W = smce_optimization(Y, lambda_val, KMax, verbose)

    W = processC(W, 0.95)

    # symmetrize the adjacency matrices
    Wsym = np.maximum(np.abs(W), np.abs(W.T))

    # perform clustering
    Yj, clusters, missrate = smce_clustering(Wsym, n, dim, gtruth)

    # perform embedding
    Yc, ind = smce_embedding(Wsym, clusters, dim)
    
    return Yc, Yj, clusters, missrate

def smce_optimization(X, lambda_val=10, KMax=None, verbose=True):
    """
    This function solves the optimization function of SMCE for the given
    data points
    Parameters:
    -----------
    X: 
        DxN matrix of N data points in the D-dimensional space
    lambda: 
        regularization parameter of the SMCE optimization program
    KMax:
        maximum neighborhood size to select the sparse neighbors from
    verbose: 
        ture if want to see the optimization information, else false

    Return:
    -----------
    W: 
        NxN sparse matrix of weights obtained by the SMCE algorithm 
    """
    N = X.shape[1]

    KMax=N-1 if (KMax is None or KMax > N - 1 or KMax < 1) else KMax
    Dist = cdist(X.T, X.T, 'euclidean')

    W = np.zeros((N, N))  # weight matrix used for clustering and dimension reduction

    for i in range(N):
        ids = np.argsort(Dist[:, i])
        ids = ids[:KMax]

        Y = X[:, ids[1:]] - X[:, ids[0]].reshape(-1, 1)
        v = Dist[ids[1:], i].reshape(-1, 1)

        for j in range(KMax - 1):
            Y[:, j] = Y[:, j] / v[j]

        if verbose:
            print(f'Point {i + 1}, ', end='')
        

        c = admm_vec_func(Y, v / np.sum(v), lambda_val, verbose)

        W[ids[1:KMax], i] = (np.abs(c / v) / np.sum(np.abs(c / v))).reshape(-1)

    return W

def processC(C, ro=1):
    if ro < 1:
        m, N = C.shape
        Cp = np.zeros((m, N))
        S, Ind = np.sort(np.abs(C), axis=0)[::-1], np.argsort(np.abs(C), axis=0)[::-1]
        for i in range(N):
            cL1 = np.sum(S[:, i])
            stop = False
            cSum = 0
            t = 0
            while not stop:
                cSum = cSum + S[t, i]      
                if cSum >= ro * cL1:
                    stop = True
                    Cp[Ind[:t+1, i], i] = C[Ind[:t+1, i], i]
                t += 1
    else:
        Cp = C

    return Cp

def smce_clustering(W, n, dim, gtruth):
    if n == 1:
        gtruth = np.ones((1, W.shape[0]))

    MAXiter = 1000
    REPlic = 20
    N = W.shape[0]
    n_clusters = len(np.unique(gtruth))

    # cluster the data using the normalized symmetric Laplacian
    D = np.diag(1 / np.sqrt(np.sum(W, axis=0) + np.finfo(float).eps))
    L = np.eye(N) - np.matmul(np.matmul(D, W), D)
    _, _, V = np.linalg.svd(L)
    V = V.T
    Yn = V[:, ::-1][:, :n_clusters]
    for i in range(N):
        Yn[i, :] = Yn[i, :n_clusters] / np.linalg.norm(Yn[i, :n_clusters] + np.finfo(float).eps)

    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters, max_iter=MAXiter, n_init=REPlic).fit(Yn[:, :n_clusters])
        grp = kmeans.labels_
    else:
        grp = np.zeros(N)

    Y = V[:, -2:-2-dim:-1]
    Y = np.matmul(D, Y).T

    # compute the misclassification rate if n > 1
    missrate = np.sum(grp != gtruth) / len(gtruth)

    return Y, grp, missrate

def smce_embedding(W, grp=None, dim=None):
    if grp is None:
        grp = np.zeros(W.shape[0])
    n = len(np.unique(grp))

    if dim is None:
        dim = 3 * np.zeros(W.shape[0])

    Yg = {}
    indg = {}
    for i in range(n+1):
        indg[i] = np.where(grp == i)[0]

        Ng = len(indg[i])
        Wg = W[indg[i]][:, indg[i]]
        Yg[i], _ = SpectralEmbedding(Wg, dim)
        Yg[i] = np.sqrt(Ng) * Yg[i].T
    return Yg, indg

def admm_vec_func(Y, q, lambda_val=10, verbose=True, mu=10, thr=[1e-6, 1e-6, 1e-5], maxIter=10000):
    
    N = Y.shape[1]

    A = np.linalg.inv((np.matmul(Y.T, Y)) + mu * np.eye(N) + mu)
    C = np.zeros((N, 1))
    Z1 = np.zeros((N, 1))
    Lambda = np.zeros((N, 1))
    gamma = 0
    err1 = 10 * thr[0]
    err2 = 10 * thr[1]
    err3 = 10 * thr[2]
    i = 1

    while (err1 > thr[0] or err2 > thr[1] or err3 > thr[2]) and i < maxIter:
        Z2 = np.matmul(A, (mu * C - Lambda + gamma))
        C = np.maximum(0, (np.abs(mu * Z2 + Lambda) - lambda_val * q)) * np.sign(mu * Z2 + Lambda) / mu
        Lambda = Lambda + mu * (Z2 - C)
        gamma = gamma + mu * (1 - np.sum(Z2, axis=0))

        err1 = errorCoef(Z2, C)
        err2 = errorCoef(np.sum(Z2, axis=0), np.ones((1, N)))
        err3 = errorCoef(Z1, Z2)

        Z1 = Z2
        i += 1

    if verbose:
        print(f'errors = [{err1:.1e} {err2:.1e} {err3:.1e}], iter: {i}')

    return C

def SpectralEmbedding(W, d):
    N = W.shape[0]
    if d > N-1:
        d = N-1
        print(f'Warning: d is set to {d}')
    eps = np.finfo(float).eps
    D = np.diag(1 / np.sqrt(np.sum(W, axis=0) + eps))
    L = np.eye(N) - np.matmul(np.matmul(D,W), D)
    _, S, V = np.linalg.svd(L)
    V = V.T
    Y = np.matmul(D, V[:, -d-1:-1][:,::-1])
    Eval = np.diag(S[-d-1:-1][::-1])

    return Y, Eval



def errorCoef(Z, C):
    # assert Z.shape == C.shape
    err = np.sum(np.abs(Z - C)) / np.prod(C.shape)
    return err
