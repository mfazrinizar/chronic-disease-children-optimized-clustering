import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, AgglomerativeClustering

from src.util.config import IMG_DIR, DATA_DIR, TIMESTAMP, SNAPSHOTS_DIR
from src.model.scalers import get_scalers_map
from src.model.clustering import HAS_SKFUZZY, manual_fcm

if HAS_SKFUZZY:
    import skfuzzy as fuzz


def visualize_dataset_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"    [WARN] Cannot read CSV {csv_path}: {e}")
        return

    if 'Year' in df.columns:
        df['Year'] = df['Year'].astype(int)

    metrics = [c for c in df.columns if c not in ['Province', 'Year']]
    raw_copy = os.path.join(DATA_DIR, "raw_dataset_copy.csv")
    df.to_csv(raw_copy, index=False)

    for m in metrics:
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=df, x='Year', y=m, hue='Province', marker='o')
        plt.title(f"{m} - trend by Province")
        plt.tight_layout()
        path = os.path.join(IMG_DIR, f"raw_trend_{m}.png")
        plt.savefig(path)
        plt.close()

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, m in zip(axes, metrics):
        sns.lineplot(data=df, x='Year', y=m, hue='Province', marker='o', ax=ax, legend=False)
        ax.set_title(m)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.95))
    fig.tight_layout(rect=[0, 0, 0.95, 1])
    combined_path = os.path.join(IMG_DIR, "raw_trends_combined.png")
    fig.savefig(combined_path)
    plt.close(fig)

    agg = df.groupby('Province')[metrics].mean()
    agg_path = os.path.join(DATA_DIR, "raw_dataset_province_mean_stats.csv")
    agg.to_csv(agg_path)

    print(f"    [DATA] Raw CSV copy: {raw_copy}")
    for m in metrics:
        print(f"    [IMG] raw_trend_{m}.png -> {os.path.join(IMG_DIR, f'raw_trend_{m}.png')}")
    print(f"    [IMG] combined overview -> {combined_path}")
    print(f"    [DATA] Province mean stats -> {agg_path}")


def visualize_preprocessing(X_raw, feature_names=None):
    scalers = get_scalers_map()
    fig, axes = plt.subplots(1, len(scalers) + 1, figsize=(20, 4))
    cols = list(feature_names) if feature_names is not None else [f"F{i}" for i in range(X_raw.shape[1])]
    df_raw = pd.DataFrame(X_raw, columns=cols)
    sns.boxplot(data=df_raw, ax=axes[0], color='lightgray')
    axes[0].set_title("Raw Data")
    stats_map = {'Raw': df_raw.describe()}
    for i, (name, scaler) in enumerate(scalers.items()):
        try:
            X_sc = scaler.fit_transform(X_raw)
            df_sc = pd.DataFrame(X_sc, columns=cols)
            sns.boxplot(data=df_sc, ax=axes[i + 1], color='skyblue')
            axes[i + 1].set_title(name)
            stats_map[name] = df_sc.describe()
        except Exception:
            pass
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "preprocessing_comparison.png"))
    plt.close()

    stats_combined = pd.concat(stats_map, axis=0)
    stats_csv_path = os.path.join(DATA_DIR, f"preprocessing_comparison_stats_{TIMESTAMP}.csv")
    stats_combined.to_csv(stats_csv_path)

    stats_txt_path = os.path.join(DATA_DIR, f"preprocessing_comparison_text_{TIMESTAMP}.txt")
    with open(stats_txt_path, "w", encoding="utf-8") as f:
        f.write("PREPROCESSING COMPARISON (FULL STATS)\n")
        f.write("=" * 80 + "\n\n")
        for k, df in stats_map.items():
            f.write(f"--- {k} ---\n")
            f.write(df.to_string())
            f.write("\n\n")

    print("    [INFO] Preprocessing visualization saved.")
    print(f"    [DATA] Full preprocessing stats saved to:\n        - {stats_csv_path}\n        - {stats_txt_path}")


def visualize_normalized_data(X_norm, provinces, feature_names):
    try:
        df_norm = pd.DataFrame(X_norm, index=provinces, columns=feature_names)
    except Exception:
        df_norm = pd.DataFrame(X_norm)
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df_norm, palette='Set2')
    plt.title("Normalized Data - Feature Distributions (boxplot)")
    path_box = os.path.join(IMG_DIR, "normalized_data_boxplot.png")
    plt.tight_layout()
    plt.savefig(path_box)
    plt.close()

    try:
        corr = df_norm.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Normalized Data - Feature Correlation")
        path_corr = os.path.join(IMG_DIR, "normalized_data_correlation.png")
        plt.tight_layout()
        plt.savefig(path_corr)
        plt.close()
    except Exception:
        path_corr = None

    print(f"    [IMG] Normalized data visuals saved:\n        - {path_box}")
    if path_corr:
        print(f"        - {path_corr}")


def visualize_projection(df_proj, provinces=None, k=3):
    if isinstance(df_proj, str):
        if not os.path.exists(df_proj):
            print(f"    [WARN] Projection file not found: {df_proj}")
            return
        df = pd.read_csv(df_proj, index_col=0)
    else:
        df = df_proj.copy()
    if not {'Dim1', 'Dim2'}.issubset(set(df.columns)):
        print("    [WARN] Projection DataFrame missing Dim1/Dim2 columns.")
        return

    clusters = df['Cluster_KM'] if 'Cluster_KM' in df.columns else None
    plt.figure(figsize=(8, 6))
    if clusters is not None:
        sns.scatterplot(x=df['Dim1'], y=df['Dim2'], hue=clusters, palette='tab10', s=200)
    else:
        sns.scatterplot(x=df['Dim1'], y=df['Dim2'], s=200)
    if provinces is None:
        try:
            provinces = df.index.tolist()
        except Exception:
            provinces = None
    if provinces is not None and len(provinces) == len(df):
        for i, txt in enumerate(provinces):
            plt.annotate(txt, (df['Dim1'].iloc[i], df['Dim2'].iloc[i] + 0.02), fontsize=8, ha='center')
    plt.title("2D Projection")
    proj_path = os.path.join(IMG_DIR, "projection_scatter.png")
    plt.tight_layout()
    plt.savefig(proj_path)
    plt.close()
    print(f"    [IMG] Projection scatter saved -> {proj_path}")


def visualize_tuning_history(history_csv_path):
    if not os.path.exists(history_csv_path):
        print(f"    [WARN] Tuning history not found: {history_csv_path}")
        return
    try:
        df = pd.read_csv(history_csv_path)
    except Exception as e:
        print(f"    [WARN] Cannot read tuning history: {e}")
        return

    plt.figure(figsize=(10, 5))
    if 'Iteration' in df.columns:
        x = df['Iteration']
    else:
        x = np.arange(len(df))
    if 'SI' in df.columns:
        plt.plot(x, df['SI'], label='SI', marker='o')
    if 'DBI' in df.columns:
        plt.plot(x, df['DBI'], label='DBI', marker='o')
    if 'Score' in df.columns:
        plt.plot(x, df['Score'], label='Score', marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Metric")
    plt.title("Tuning History: SI / DBI / Score")
    plt.legend()
    hist_path = os.path.join(IMG_DIR, "tuning_history_metrics.png")
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()

    print(f"    [IMG] Tuning history metrics plot saved -> {hist_path}")


def save_snapshot_workflow(X_proj, config, provinces, k, iteration, current_history, df_stats):
    prefix = f"iter_{iteration:04d}_SI{config['SI']:.3f}"
    print(f"    [SNAPSHOT SAVED] {prefix}")

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lab_km = km.fit_predict(X_proj)

    hc = AgglomerativeClustering(n_clusters=k, linkage='average')
    lab_hc = hc.fit_predict(X_proj)
    Z_hc = linkage(X_proj, method='average')

    if HAS_SKFUZZY:
        _, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
            X_proj.T, c=k, m=2.0, error=0.005, maxiter=1000, init=None
        )
    else:
        _, u, fpc = manual_fcm(X_proj, c=k)
        u = u.T
    if u.shape[0] == k:
        u = u.T

    df_u = pd.DataFrame(u, index=provinces, columns=[f"Membership_C{j}" for j in range(u.shape[1])])

    df_stats_w_label = df_stats.copy()
    df_stats_w_label['Cluster'] = lab_km
    cluster_summary = df_stats_w_label.groupby('Cluster').mean()

    explanation_text = [
        f"SNAPSHOT REPORT: Iteration {iteration}",
        f"Metrics: SI={config['SI']:.4f}, DBI={config['DBI']:.4f}",
        f"Config: {config}\n",
        "=" * 40,
        "CLUSTER PROFILES (MEAN STATISTICS)",
        "=" * 40
    ]

    for c_id in sorted(df_stats_w_label['Cluster'].unique()):
        members = df_stats_w_label[df_stats_w_label['Cluster'] == c_id].index.tolist()
        stats = cluster_summary.loc[c_id]

        info = f"CLUSTER {c_id} ({', '.join(members)})"
        stat_detail = f"  -> Asthma: {stats['Asthma_Prevalence_pct']:.2f}% | Pneumonia: {stats['Pneumonia_Prevalence_pct']:.2f}% | Anemia: {stats['Anemia_Prevalence_pct']:.2f}%"
        explanation_text.append(info)
        explanation_text.append(stat_detail)
        explanation_text.append("-" * 20)

    snap_txt = os.path.join(SNAPSHOTS_DIR, f"{prefix}_explanation.txt")
    with open(snap_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(explanation_text))

    data_txt = os.path.join(DATA_DIR, f"{prefix}_explanation.txt")
    with open(data_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(explanation_text))

    fig = plt.figure(figsize=(20, 6))
    plt.suptitle(f"SNAPSHOT (Iter {iteration}) | SI: {config['SI']:.4f}", fontsize=14, fontweight='bold')

    ax1 = plt.subplot(1, 3, 1)
    sns.scatterplot(x=X_proj[:, 0], y=X_proj[:, 1], hue=lab_km, palette='viridis', s=600, style=provinces, ax=ax1)
    for i, txt in enumerate(provinces):
        ax1.annotate(txt, (X_proj[i, 0], X_proj[i, 1] + 0.05), ha='center', fontsize=8)
    ax1.set_title("K-Means")

    ax2 = plt.subplot(1, 3, 2)
    dendrogram(Z_hc, labels=provinces, leaf_rotation=45, ax=ax2)
    ax2.set_title("Hierarchical")

    ax3 = plt.subplot(1, 3, 3)
    sns.heatmap(df_u, annot=True, cmap='Blues', fmt='.2f', yticklabels=provinces, xticklabels=df_u.columns, ax=ax3)
    ax3.set_title("Fuzzy Membership")

    plt.tight_layout()
    plt.savefig(os.path.join(SNAPSHOTS_DIR, f"{prefix}_dashboard.png"))
    plt.close()

    df_proj_out = pd.DataFrame(X_proj, index=provinces, columns=['Dim1', 'Dim2'])
    df_proj_out['Cluster_KM'] = lab_km
    df_proj_out.to_csv(os.path.join(SNAPSHOTS_DIR, f"{prefix}_projection.csv"))

    df_proj_out.to_csv(os.path.join(DATA_DIR, f"{prefix}_projection.csv"))
    cluster_summary_csv = os.path.join(DATA_DIR, f"{prefix}_cluster_summary.csv")
    cluster_summary.to_csv(cluster_summary_csv)

    df_u.to_csv(os.path.join(DATA_DIR, f"{prefix}_fuzzy_membership.csv"))

    if current_history:
        pd.DataFrame(current_history).to_csv(os.path.join(SNAPSHOTS_DIR, f"{prefix}_history_so_far.csv"), index=False)
        pd.DataFrame(current_history).to_csv(os.path.join(DATA_DIR, f"{prefix}_history_so_far.csv"), index=False)

    print(f"    [DATA] Saved snapshot text & numeric files to {DATA_DIR}:")
    print(f"        - {data_txt}")
    print(f"        - {cluster_summary_csv}")
    print(f"        - {os.path.join(DATA_DIR, f'{prefix}_fuzzy_membership.csv')}")
