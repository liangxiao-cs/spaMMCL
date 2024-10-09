from sklearn.cluster import KMeans
import ot
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
import numba


@numba.njit("f4(f4[:], f4[:])")
def euclid_dist(t1, t2):
    sum = 0
    for i in range(t1.shape[0]):
        sum += (t1[i] - t2[i]) ** 2
    return np.sqrt(sum)

# 1003
@numba.njit("f4[:,:](f4[:,:])", parallel=True, nogil=True)
def pairwise_distance(X):
    n = X.shape[0]
    adj = np.empty((n, n), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(n):
            adj[i][j] = euclid_dist(X[i], X[j])
    return adj


def calculate_adj_matrix(x, y):
    X = np.array([x, y]).T.astype(np.float32)
    return pairwise_distance(X)


def Moran_I(genes_exp, x, y, k=5, knn=True):
    XYmap = pd.DataFrame({"x": x, "y": y})
    if knn:
        XYnbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean').fit(XYmap)
        XYdistances, XYindices = XYnbrs.kneighbors(XYmap)
        W = np.zeros((genes_exp.shape[0], genes_exp.shape[0]))
        for i in range(0, genes_exp.shape[0]):
            W[i, XYindices[i, :]] = 1
        for i in range(0, genes_exp.shape[0]):
            W[i, i] = 0
    else:
        W = calculate_adj_matrix(x=x, y=y, histology=False)
    I = pd.Series(index=genes_exp.columns, dtype="float64")
    for k in genes_exp.columns:
        X_minus_mean = np.array(genes_exp[k] - np.mean(genes_exp[k]))
        X_minus_mean = np.reshape(X_minus_mean, (len(X_minus_mean), 1))
        Nom = np.sum(np.multiply(W, np.matmul(X_minus_mean, X_minus_mean.T)))
        Den = np.sum(np.multiply(X_minus_mean, X_minus_mean))
        I[k] = (len(genes_exp[k]) / np.sum(W)) * (Nom / Den)
    return I


def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
    n_cell = distance.shape[0]
    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
    new_type = [str(i) for i in list(new_type)]
    return new_type


def clustering(adata_clu, n_clusters=7, radius=50, refinement=True):
    embedding = adata_clu.obsm['emb']
    kk = KMeans(n_clusters=n_clusters, random_state=0).fit(embedding)
    raw_preds = kk.labels_
    adata_clu.obs['domain'] = raw_preds
    if refinement:
        new_type = refine_label(adata_clu, radius, key='domain')
        adata_clu.obs['domain'] = new_type