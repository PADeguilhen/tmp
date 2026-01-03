import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

PIXEL_TO_MM = 0.86

REFERENCE_CSV = "csv/reference.csv"
ORIGINAL_VIDEO = "videos/original.mp4"

# Associe débit → fichiers
DATA = {
    200: {
        "video": "videos/compressed/original_200k.mp4",
        "csv": "csv/mediapipe_original_200k.csv"
    },
    500: {
        "video": "videos/compressed/original_500k.mp4",
        "csv": "csv/mediapipe_original_500k.csv"
    },
    1000: {
        "video": "videos/compressed/original_1000k.mp4",
        "csv": "csv/mediapipe_original_1000k.csv"
    },
    2000: {
        "video": "videos/compressed/original_2000k.mp4",
        "csv": "csv/mediapipe_original_2000k.csv"
    }
}

# =========================
# METRICS
# =========================

def compute_rmse_mm(ref, pred):
    diff = (ref - pred) * PIXEL_TO_MM
    return np.sqrt(np.nanmean(diff ** 2))

def compute_psnr(video_ref, video_cmp):
    cap_ref = cv2.VideoCapture(video_ref)
    cap_cmp = cv2.VideoCapture(video_cmp)

    psnr_values = []

    while True:
        ret1, f1 = cap_ref.read()
        ret2, f2 = cap_cmp.read()

        if not ret1 or not ret2:
            break

        psnr_values.append(cv2.PSNR(f1, f2))

    cap_ref.release()
    cap_cmp.release()

    return np.mean(psnr_values)

# =========================
# LOAD REFERENCE
# =========================

ref = pd.read_csv(REFERENCE_CSV, sep=";")
coord_cols = [c for c in ref.columns if c.endswith("_x") or c.endswith("_y")]
ref = ref[coord_cols]

# =========================
# COMPUTE VALUES
# =========================

bitrates = []
rmse_values = []
psnr_values = []

for br, paths in DATA.items():
    print(f"Processing {br} kbps")

    pred = pd.read_csv(paths["csv"])
    pred = pred[coord_cols]

    rmse = compute_rmse_mm(ref, pred)
    psnr = compute_psnr(ORIGINAL_VIDEO, paths["video"])

    bitrates.append(br)
    rmse_values.append(rmse)
    psnr_values.append(psnr)

# =========================
# PLOT (DOUBLE AXIS)
# =========================

fig, ax1 = plt.subplots(figsize=(9, 5))

ax1.set_xlabel("Débit binaire (kb/s)")
ax1.set_ylabel("Erreur globale RMSE (mm)", color="tab:red")
ax1.plot(bitrates, rmse_values, "-o", color="tab:red", label="RMSE (mm)")
ax1.tick_params(axis="y", labelcolor="tab:red")
ax1.grid(True)

ax2 = ax1.twinx()
ax2.set_ylabel("PSNR moyen (dB)", color="tab:blue")
ax2.plot(bitrates, psnr_values, "-s", color="tab:blue", label="PSNR (dB)")
ax2.tick_params(axis="y", labelcolor="tab:blue")

plt.title("PSNR et erreur de reconnaissance en fonction du débit (MediaPipe)")
fig.tight_layout()
plt.savefig("results/psnr_vs_rmse_mediapipe.png", dpi=300)
plt.close()

print("Figure sauvegardée : results/psnr_vs_rmse_mediapipe.png")
