import os
import uuid
from datetime import datetime

USE_FIXED_ID = True
FIXED_ID_NAME = "cluster-search-flow-v1"

if USE_FIXED_ID:
    RUN_ID = FIXED_ID_NAME
else:
    RUN_ID = f"cluster-search-flow-{str(uuid.uuid4())[:6]}"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

BASE_DIR = f"./experiments/experiment_results_{RUN_ID}"
CONFIG_DIR = os.path.join(BASE_DIR, "config")
IMG_DIR = os.path.join(BASE_DIR, "images")
DATA_DIR = os.path.join(BASE_DIR, "data")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "snapshots_evolution")

CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, "best_config.json")
PROJECTION_FILE_PATH = os.path.join(DATA_DIR, "best_projection.csv")

MODEL_DATA_DIR = "./model_data"
MODEL_DATA_CONFIGS_DIR = os.path.join(MODEL_DATA_DIR, "configs")
MODEL_DATA_DATA_DIR = os.path.join(MODEL_DATA_DIR, "data")
MODEL_DATA_PROJECTIONS_DIR = os.path.join(MODEL_DATA_DIR, "projections")

DATASET_PATH = "./dataset/chronic_disease_children_trend.csv"

for d in [BASE_DIR, CONFIG_DIR, IMG_DIR, DATA_DIR, SNAPSHOTS_DIR]:
    os.makedirs(d, exist_ok=True)
