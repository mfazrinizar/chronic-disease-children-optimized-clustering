import pandas as pd
from src.util.config import DATASET_PATH
from src.util.config import MODEL_DATA_DIR
from src.util.config import MODEL_DATA_DATA_DIR
from src.util.config import MODEL_DATA_CONFIGS_DIR


def load_and_transform(csv_path=None):
    if csv_path is None:
        csv_path = DATASET_PATH
    try:
        df = pd.read_csv(csv_path)
        print("    [INFO] CSV Found & Loaded.")
    except Exception:
        print("    [WARN] CSV Not Found! Using Manual Fallback Data.")
        data = {
            'Province': ['Jawa Barat'] * 9 + ['Jawa Tengah'] * 9 + ['Jawa Timur'] * 9 +
                        ['Sumatera Selatan'] * 9 + ['DKI Jakarta'] * 9 + ['Bali'] * 9,
            'Year': list(range(2015, 2024)) * 6,
            'Asthma_Prevalence_pct': [
                11.92, 10.56, 10.9, 10.75, 9.58, 10.87, 10.01, 9.96, 10.74,
                12.37, 10.09, 11.74, 10.8, 11.84, 10.91, 11.5, 10.5, 10.02,
                7.42, 6.19, 7.74, 6.2, 6.96, 6.23, 8.15, 7.31, 7.59,
                7.28, 7.29, 8.26, 7.18, 6.18, 7.01, 7.85, 8.85, 9.42,
                7.73, 6.99, 7.2, 8.59, 7.11, 8.3, 7.18, 5.61, 7.72,
                5.28, 6.88, 7.1, 5.48, 8.62, 8.01, 6.92, 10.41, 6.67
            ],
            'Pneumonia_Prevalence_pct': [
                2.79, 2.94, 3.08, 4.36, 3.63, 2.91, 3.95, 2.44, 2.84,
                2.55, 4.74, 4.54, 5.97, 4.35, 4.25, 3.3, 5.58, 4.52,
                3.11, 3.21, 4.06, 3.11, 3.47, 2.86, 3.83, 3.63, 2.55,
                6.02, 8.62, 8.15, 5.8, 6.34, 5.96, 7.46, 6.21, 8.38,
                10.55, 8.41, 10.09, 8.85, 11.2, 9.79, 10.59, 10.49, 8.61,
                5.59, 7.49, 6.4, 6.85, 7.25, 6.47, 7.62, 6.73, 6.78
            ],
            'Anemia_Prevalence_pct': [
                24.12, 32.21, 25.76, 28.24, 27.52, 26.86, 26.88, 28.52, 26.2,
                29.94, 30.62, 35.5, 24.98, 25.75, 30.81, 31.68, 29.24, 29.85,
                23.01, 25.14, 22.28, 22.51, 22.73, 24.42, 24.59, 23.65, 23.83,
                24.16, 25.57, 25.07, 23.86, 27.83, 26.84, 20.08, 25.64, 22.13,
                19.13, 23.22, 21.65, 21.53, 18.46, 20.73, 20.66, 22.21, 18.19,
                29.63, 28.24, 34.55, 31.11, 30.72, 33.89, 32.77, 32.66, 26.49
            ]
        }
        df = pd.DataFrame(data)

    val_vars = [c for c in df.columns if c not in ['Province', 'Year']]
    df_pivot = df.pivot(index='Province', columns='Year', values=val_vars)
    df_pivot = df_pivot.sort_index(axis=1)
    df_pivot.columns = [f'{col[0]}_{col[1]}' for col in df_pivot.columns]

    df_stats = df.groupby('Province')[val_vars].mean()

    return df_pivot.values, df_pivot.index, df_pivot.columns, df_stats
