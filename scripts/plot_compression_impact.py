import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PIXEL_TO_MM = 0.86

FILES = {
    "MediaPipe Original": "csv/mediapipe_original_200k.csv",
    "MediaPipe Anonymized": "csv/mediapipe_anonymized_200k.csv",
    "RTMW Original": "csv/rtmw_original_200k.csv",
    "RTMW Anonymized": "csv/rtmw_anonymized_200k.csv"
}

REFERENCE = "csv/reference.csv"

GROUPS = {
    "Global": None,
    "Body": ["shoulder", "elbow", "wrist"],
    "Fingers": ["thumb", "index", "middle", "ring", "little"],
    "Eyes": ["eye"]
}

# =========================
# METRICS
# =========================

def rmse_mm(ref, pred):
    diff = (ref - pred) * PIXEL_TO_MM
    return np.sqrt(np.nanmean(diff ** 2))

def missing_percentage(df):
    return 100 * df.isna().sum().sum() / df.size

# =========================
# LOAD REFERENCE
# =========================

ref = pd.read_csv(REFERENCE, sep=";")
coord_cols = [c for c in ref.columns if c.endswith("_x") or c.endswith("_y")]

# =========================
# COMPUTE METRICS
# =========================

rmse_results = {g: [] for g in GROUPS}
missing_results = []

labels = []

for label, path in FILES.items():
    pred = pd.read_csv(path)
    labels.append(label)

    missing_results.append(missing_percentage(pred[coord_cols]))

    for group, keywords in GROUPS.items():
        if keywords is None:
            cols = coord_cols
        else:
            cols = [c for c in coord_cols if any(k in c.lower() for k in keywords)]

        rmse_results[group].append(
            rmse_mm(ref[cols], pred[cols])
        )

# =========================
# PLOTS
# =========================

x = np.arange(len(labels))
width = 0.2

# --- RMSE ---
plt.figure(figsize=(10, 5))
for i, (group, values) in enumerate(rmse_results.items()):
    plt.bar(x + i * width, values, width, label=group)

plt.xticks(x + width * 1.5, labels, rotation=20)
plt.ylabel("RMSE (mm)")
plt.title("Impact de la compression forte sur la précision des modèles")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("results/rmse_compression.png", dpi=300)

# --- Missing data ---
plt.figure(figsize=(8, 4))
plt.bar(labels, missing_results)
plt.ylabel("Données manquantes (%)")
plt.title("Pourcentage de non-détections (compression forte)")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("results/missing_data_compression.png", dpi=300)
