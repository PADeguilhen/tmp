import pandas as pd
import numpy as np

PIXEL_TO_MM = 0.86

GROUPS = {
    "body": ["shoulder", "elbow", "wrist"],
    "fingers": ["thumb", "index", "middle", "ring", "pinky"],
    "eyes": ["eye"]
}

def rmse(a, b):
    return np.sqrt(np.nanmean((a - b) ** 2))

def compute_metrics(reference_csv, model_csv):
    ref = pd.read_csv(reference_csv)
    pred = pd.read_csv(model_csv)

    results = {}

    # Toutes les colonnes x,y
    coord_cols = [c for c in ref.columns if c.endswith("_x") or c.endswith("_y")]

    # --- Erreur globale ---
    diff = (ref[coord_cols] - pred[coord_cols]) * PIXEL_TO_MM
    results["rmse_global_mm"] = rmse(diff.values, 0)

    # --- Par groupe ---
    for group, keywords in GROUPS.items():
        cols = [
            c for c in coord_cols
            if any(k in c.lower() for k in keywords)
        ]
        if cols:
            diff_group = (ref[cols] - pred[cols]) * PIXEL_TO_MM
            results[f"rmse_{group}_mm"] = rmse(diff_group.values, 0)

    # --- Données manquantes ---
    missing = pred[coord_cols].isna().sum().sum()
    total = pred[coord_cols].shape[0] * pred[coord_cols].shape[1]
    results["missing_percentage"] = 100 * missing / total

    return results


if __name__ == "__main__":
    metrics = compute_metrics(
        "csv/reference.csv",
        "csv/mediapipe_demo_500k.csv"
    )
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")
