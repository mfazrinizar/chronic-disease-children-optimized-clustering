import os
import json
import numpy as np
import pandas as pd
from sklearn.manifold import MDS, TSNE, SpectralEmbedding, Isomap
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances
from sklearn.preprocessing import StandardScaler

from src.util.config import (
    CONFIG_FILE_PATH, PROJECTION_FILE_PATH, DATA_DIR, SNAPSHOTS_DIR, RUN_ID
)
from src.model.scalers import get_scalers_map
from src.model.clustering import evaluate_all_models
from src.preprocess.visualization import save_snapshot_workflow


def hyper_grid_search(X_raw, provinces, df_stats, k=3):
    if os.path.exists(CONFIG_FILE_PATH):
        with open(CONFIG_FILE_PATH, 'r') as f:
            data = json.load(f)
        best_config = data.get('hyperparams', {})

        if os.path.exists(PROJECTION_FILE_PATH):
            df_proj = pd.read_csv(PROJECTION_FILE_PATH, index_col=0)
            if 'Cluster_KM' in df_proj.columns:
                X_final = df_proj.drop(columns=['Cluster_KM']).values
            else:
                X_final = df_proj.values
            try:
                evals = evaluate_all_models(X_final, k)
                print("    [CACHE EVAL] Cluster evaluation on cached projection:")
                for name, m in evals.items():
                    si = m.get('SI', float('nan'))
                    dbi = m.get('DBI', float('nan'))
                    print(f"        - {name}: SI={si:.4f}, DBI={dbi:.4f}")
            except Exception as e:
                print(f"    [CACHE EVAL] Evaluation failed: {e}")

            save_snapshot_workflow(X_final, best_config, list(provinces), k, iteration=0, current_history=[], df_stats=df_stats)

            return best_config, X_final

        # reconstruct projection if projection missing
        try:
            scaler = get_scalers_map().get(best_config.get('Scaler', 'Standard'), StandardScaler())
            X_scaled = scaler.fit_transform(X_raw)

            transformer_type = best_config.get('Transformer')
            metric = best_config.get('Metric', 'euclidean')
            seed = int(best_config.get('Seed', 42))

            X_proj = None
            if transformer_type == 'MDS':
                dist = pairwise_distances(X_scaled, metric=metric)
                model = MDS(n_components=2, dissimilarity='precomputed', random_state=seed, normalized_stress='auto')
                X_proj = model.fit_transform(dist)
            elif transformer_type == 'Spectral':
                aff = 'nearest_neighbors' if metric != 'rbf' else 'rbf'
                model = SpectralEmbedding(n_components=2, random_state=seed, affinity=aff)
                X_proj = model.fit_transform(X_scaled)
            elif transformer_type == 'Isomap':
                n_neighbors = int(best_config.get('n_neighbors', 5))
                model = Isomap(n_components=2, n_neighbors=n_neighbors, metric=metric)
                X_proj = model.fit_transform(X_scaled)
            elif transformer_type == 't-SNE':
                perp = float(best_config.get('perplexity', 30))
                model = TSNE(n_components=2, metric=metric, perplexity=perp, random_state=seed, init='random', method='exact')
                X_proj = model.fit_transform(X_scaled)
            elif transformer_type == 'KernelPCA':
                kernel = best_config.get('kernel', 'rbf')
                model = KernelPCA(n_components=2, kernel=kernel, random_state=seed)
                X_proj = model.fit_transform(X_scaled)

            if X_proj is None:
                raise RuntimeError("Unknown transformer in saved config; cannot reconstruct projection.")

            save_snapshot_workflow(X_proj, best_config, list(provinces), k, iteration=0, current_history=[], df_stats=df_stats)

            last_snapshot = sorted([f for f in os.listdir(SNAPSHOTS_DIR) if 'projection.csv' in f])[-1]
            df_best = pd.read_csv(os.path.join(SNAPSHOTS_DIR, last_snapshot), index_col=0)
            df_best.to_csv(PROJECTION_FILE_PATH)

            X_final = df_best.drop(columns=['Cluster_KM']).values if 'Cluster_KM' in df_best.columns else df_best.values
            try:
                evals = evaluate_all_models(X_final, k)
                print("    [CACHE EVAL] Cluster evaluation on cached projection:")
                for name, m in evals.items():
                    si = m.get('SI', float('nan'))
                    dbi = m.get('DBI', float('nan'))
                    print(f"        - {name}: SI={si:.4f}, DBI={dbi:.4f}")
            except Exception as e:
                print(f"    [CACHE EVAL] Evaluation failed: {e}")

            return best_config, X_final

        except Exception as e:
            print(f"    [WARN] Failed to reconstruct projection from config: {e}")
            print("    [INFO] Falling back to full search.")

    # full search
    scalers = get_scalers_map()
    metrics_list = ['correlation', 'cosine', 'euclidean', 'manhattan']
    transformers = ['MDS', 'Spectral', 'Isomap', 't-SNE', 'KernelPCA']

    best_score = -float('inf')
    best_config = {}
    tuning_history = []
    MAX_ITER = 10

    for i in range(MAX_ITER):
        s_name = np.random.choice(list(scalers.keys()))
        metric = np.random.choice(metrics_list)
        transformer_type = np.random.choice(transformers)
        seed = np.random.randint(0, 100000)

        try:
            scaler = scalers[s_name]
            X_scaled = scaler.fit_transform(X_raw)

            X_proj = None
            if transformer_type == 'MDS':
                dist = pairwise_distances(X_scaled, metric=metric)
                model = MDS(n_components=2, dissimilarity='precomputed', random_state=seed, normalized_stress='auto')
                X_proj = model.fit_transform(dist)
            elif transformer_type == 'Spectral':
                aff = 'nearest_neighbors' if metric != 'rbf' else 'rbf'
                model = SpectralEmbedding(n_components=2, random_state=seed, affinity=aff)
                X_proj = model.fit_transform(X_scaled)
            elif transformer_type == 'Isomap':
                n_neighbors = np.random.randint(3, 6)
                model = Isomap(n_components=2, n_neighbors=n_neighbors, metric=metric)
                X_proj = model.fit_transform(X_scaled)
            elif transformer_type == 't-SNE':
                perp = np.random.choice([2, 3, 4])
                model = TSNE(n_components=2, metric=metric, perplexity=perp, random_state=seed, init='random', method='exact')
                X_proj = model.fit_transform(X_scaled)
            elif transformer_type == 'KernelPCA':
                kernel = np.random.choice(['rbf', 'poly', 'cosine'])
                model = KernelPCA(n_components=2, kernel=kernel, random_state=seed)
                X_proj = model.fit_transform(X_scaled)

            if X_proj is None:
                continue

            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_proj)
            si = silhouette_score(X_proj, labels)
            dbi = davies_bouldin_score(X_proj, labels)
            score = si - (0.5 * dbi)

            tuning_history.append({
                'Iteration': i, 'Scaler': s_name, 'Transformer': transformer_type,
                'Metric': metric, 'Seed': seed, 'SI': si, 'DBI': dbi, 'Score': score
            })

            if score > best_score:
                best_score = score
                best_config = {
                    'Scaler': s_name, 'Transformer': transformer_type,
                    'Metric': metric, 'Seed': seed, 'SI': si, 'DBI': dbi
                }
                save_snapshot_workflow(X_proj, best_config, provinces, k, i, tuning_history, df_stats)

            if si > 0.99 and dbi < 0.01:
                break

        except Exception:
            continue

    if tuning_history:
        pd.DataFrame(tuning_history).to_csv(os.path.join(DATA_DIR, "full_tuning_history.csv"), index=False)

    with open(CONFIG_FILE_PATH, 'w') as f:
        json.dump({'run_id': RUN_ID, 'hyperparams': best_config, 'metrics': {'SI': best_config.get('SI', 0.0), 'DBI': best_config.get('DBI', 0.0)}}, f, indent=4)

    last_snapshot = sorted([f for f in os.listdir(SNAPSHOTS_DIR) if 'projection.csv' in f])[-1]
    df_best = pd.read_csv(os.path.join(SNAPSHOTS_DIR, last_snapshot), index_col=0)
    df_best.to_csv(PROJECTION_FILE_PATH)

    return best_config, df_best.drop(columns=['Cluster_KM']).values
