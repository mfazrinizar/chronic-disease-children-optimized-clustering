import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

try:
    import skfuzzy as fuzz
    HAS_SKFUZZY = True
except Exception:
    HAS_SKFUZZY = False


def manual_fcm(X, c, m=2, error=0.005, maxiter=1000, seed=42):
    n_samples, n_features = X.shape
    np.random.seed(seed)
    u = np.random.dirichlet(np.ones(c), size=n_samples).T
    for i in range(maxiter):
        u_old = u.copy()
        centers = []
        for j in range(c):
            num = np.sum((u[j, :] ** m)[:, np.newaxis] * X, axis=0)
            den = np.sum(u[j, :] ** m)
            centers.append(num/den)
        centers = np.array(centers)
        dist = np.linalg.norm(X[:, np.newaxis] - centers, axis=2).T
        dist = np.fmax(dist, 1e-10)
        u = np.zeros_like(u)
        for j in range(c):
            for k_idx in range(c):
                u[j, :] += (dist[j, :] / dist[k_idx, :]) ** (2 / (m - 1))
        u = 1.0 / u
        if np.linalg.norm(u - u_old) < error:
            break
    fpc = np.sum(u ** 2) / n_samples
    return centers, u, fpc


def compute_cluster_metrics(X, k=3):
    results = {}

    # KMeans
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lab_km = km.fit_predict(X)
    results['KMeans'] = {
        'SI': silhouette_score(X, lab_km),
        'DBI': davies_bouldin_score(X, lab_km),
        'labels': lab_km
    }

    # Agglomerative
    hc = AgglomerativeClustering(n_clusters=k, linkage='average')
    lab_hc = hc.fit_predict(X)
    results['Agglomerative'] = {
        'SI': silhouette_score(X, lab_hc),
        'DBI': davies_bouldin_score(X, lab_hc),
        'labels': lab_hc
    }

    # Fuzzy C-Means
    try:
        if HAS_SKFUZZY:
            _, u, _, _, _, _, fpc = fuzz.cluster.cmeans(X.T, c=k, m=2.0, error=0.005, maxiter=1000, init=None)
        else:
            _, u, fpc = manual_fcm(X, c=k)
            if not HAS_SKFUZZY:
                u = u.T

        if u.shape[0] == k:
            u = u.T

        lab_fcm = np.argmax(u, axis=1)
        results['Fuzzy C-Means'] = {
            'SI': silhouette_score(X, lab_fcm),
            'DBI': davies_bouldin_score(X, lab_fcm),
            'labels': lab_fcm,
            'membership': u,
            'FPC': fpc
        }
    except Exception as e:
        # Return structure but with None where unavailable
        results['Fuzzy C-Means'] = {
            'SI': None,
            'DBI': None,
            'labels': None,
            'membership': None,
            'FPC': None
        }

    return results
