import os
import json
import re
import pandas as pd
from src.util.config import MODEL_DATA_DIR

CONFIGS_DIR = os.path.join(MODEL_DATA_DIR, "configs")
DATA_DIR = os.path.join(MODEL_DATA_DIR, "data")
PROJECTIONS_DIR = os.path.join(MODEL_DATA_DIR, "projections")


def load_history_data():
    path = os.path.join(DATA_DIR, "history_data.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_normalized_data():
    path = os.path.join(DATA_DIR, "best_normalized_data.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None


def list_configs():
    if not os.path.exists(CONFIGS_DIR):
        return []
    files = [f for f in os.listdir(CONFIGS_DIR) if f.endswith('.json')]
    configs = []
    for f in files:
        match = re.match(r'iter_(\d+)_SI([\d.]+)_config\.json', f)
        if match:
            iteration = int(match.group(1))
            si = float(match.group(2))
            configs.append({'file': f, 'iteration': iteration, 'SI': si})
    configs = sorted(configs, key=lambda x: x['SI'], reverse=True)
    return configs


def load_config(filename):
    path = os.path.join(CONFIGS_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def load_projection(config_file):
    proj_file = config_file.replace('_config.json', '_projection.csv')
    path = os.path.join(PROJECTIONS_DIR, proj_file)
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None
