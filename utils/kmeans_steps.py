from __future__ import annotations
import numpy as np

def init_representatives(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(len(X), size=k, replace=False)
    return X[idx].copy()

def assign_clusters(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    # distancias cuadradas: (n,k)
    d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1)

def update_centroids(X: np.ndarray, labels: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    C_new = np.zeros((k, X.shape[1]), dtype=float)
    for j in range(k):
        pts = X[labels == j]
        if len(pts) == 0:
            # cluster vacío: reubicar al azar
            C_new[j] = X[rng.integers(0, len(X))]
        else:
            C_new[j] = pts.mean(axis=0)
    return C_new

def inertia_sse(X: np.ndarray, C: np.ndarray, labels: np.ndarray) -> float:
    sse = 0.0
    for j in range(C.shape[0]):
        pts = X[labels == j]
        if len(pts) > 0:
            sse += ((pts - C[j]) ** 2).sum()
    return float(sse)
