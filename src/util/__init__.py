from .config import *


def print_step(step_num, title, description=""):
    print(f"\n{'=' * 60}")
    print(f"[{step_num}] {title.upper()}")
    if description:
        print(f"    -> {description}")
    print(f"{'=' * 60}")


def print_workflow_diagram():
    diagram = """
    =======================================================
			WORKFLOW MODELLING ARCHITECTURE
    =======================================================

    [RAW DATASET: CSV]
	    |
	    V
    [STEP 1: DATA INGESTION]
	    |-- Pivot Time-Series (Rows -> Columns)
	    |-- Aggregate Statistics (For Explanation)
	    |
	    V
    [STEP 2: PREPROCESSING VISUALIZATION]
	    |-- Boxplot Distribution Analysis
	    |-- Outlier Inspection
	    |
	    V
    [STEP 3: HYPER-EVOLUTIONARY SEARCH ENGINE]
	    |  <LOOP: 100000 Iterations (IF SI < 0.99 OR DBI > 0.01)>
	    |-- Randomize: Scaler -> Metric -> Manifold -> Seed
	    |-- Project: High Dim -> 2D Space
	    |-- Evaluate: K-Means (k=3), AgglomerativeClustering (n=3), Fuzzy C-Means (c=3) -> SI & DBI Scores
	    |-- TRIGGER: New Best Score Found?
	    |      |-- YES: [SNAPSHOT MODULE]
	    |      |    |-- Save Projection (CSV)
	    |      |    |-- Save Config (JSON)
	    |      |    |-- Generate Dashboard (PNG)
	    |      |    |-- Generate Explanation (TXT)
	    |      |    |-- Log History (CSV)
	    |
	    V
    [STEP 4: FINAL MODEL EXPORT]
	    |-- Export Best Normalized Data
	    |-- Finalize Best Configuration

    =======================================================
    """
    print(diagram)

