import os
import subprocess
import argparse

from src.util import print_workflow_diagram, print_step
from src.util.config import DATASET_PATH, DATA_DIR, PROJECTION_FILE_PATH, RUN_ID, BASE_DIR
from src.preprocess.data_loader import load_and_transform
from src.preprocess.visualization import (
    visualize_dataset_csv,
    visualize_preprocessing,
    visualize_normalized_data,
    visualize_projection,
    visualize_tuning_history
)
from src.model.search_engine import hyper_grid_search
from src.model.scalers import get_scalers_map


def run_preprocess_flow():
    print_workflow_diagram()

    X_raw, provinces, feature_names, df_stats = load_and_transform()

    if X_raw is not None:
        visualize_dataset_csv(DATASET_PATH)
        visualize_preprocessing(X_raw, feature_names)
        best_config, X_final = hyper_grid_search(X_raw, provinces, df_stats, k=3)

        print_step(4, "FINAL EXPORT", "Saving best normalized data...")
        scaler = get_scalers_map()[best_config['Scaler']]
        X_norm = scaler.fit_transform(X_raw)
        import pandas as pd
        df_norm = pd.DataFrame(X_norm, index=provinces, columns=feature_names)
        df_norm.to_csv(os.path.join(DATA_DIR, "best_normalized_data.csv"))
        print("    [INFO] 'best_normalized_data.csv' saved.")

        try:
            visualize_normalized_data(X_norm, provinces, feature_names)
        except Exception as e:
            print(f"    [WARN] visualize_normalized_data failed: {e}")

        try:
            if os.path.exists(PROJECTION_FILE_PATH):
                df_proj_vis = __import__('pandas').read_csv(PROJECTION_FILE_PATH, index_col=0)
                visualize_projection(df_proj_vis, provinces, k=3)
        except Exception as e:
            print(f"    [WARN] visualize_projection failed: {e}")

        history_csv = os.path.join(DATA_DIR, "full_tuning_history.csv")
        try:
            visualize_tuning_history(history_csv)
        except Exception as e:
            print(f"    [WARN] visualize_tuning_history failed: {e}")

        print(f"\n[COMPLETE] ALL DONE. Run ID: {RUN_ID}")
        print(f"   Check folder: {BASE_DIR}")


def run_dashboard():
    cmd = ["streamlit", "run", "src/app/dashboard.py"]
    try:
        print("Launching Streamlit dashboard...")
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"Failed to launch dashboard: {e}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run project flows: dashboard or preprocess")
    parser.add_argument('--mode', choices=['dashboard', 'preprocess'], help='Mode to run')
    args = parser.parse_args(argv)

    if args.mode == 'dashboard':
        run_dashboard()
    elif args.mode == 'preprocess':
        run_preprocess_flow()
    else:
        # interactive chooser
        print("Select mode to run:\n 1) Dashboard (interactive UI)\n 2) Preprocess -> Search -> Export")
        choice = input("Enter 1 or 2: ")
        if choice.strip() == '1':
            run_dashboard()
        else:
            run_preprocess_flow()


if __name__ == '__main__':
    main()
