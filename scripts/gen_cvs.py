import subprocess
import os

# =========================
# CONFIGURATION
# =========================

VIDEO_ORIGINAL = "videos/demo.mp4"
VIDEO_ANON = "videos/anonymized_demo.mp4"

OUTPUT_VIDEO_DIR = "videos/compressed"
OUTPUT_CSV_DIR = "csv"

MODEL = "rtmw"   # "mediapipe" or "rtmw"
SIDE = "right"
SHOW = "n"

BITRATES = [200, 500, 1000, 2000]  # kbps

PYTHON = "python3"  # or full path if needed
FFMPEG = "ffmpeg"

# =========================
# PREPARE FOLDERS
# =========================

os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

# =========================
# SELECT MODEL SCRIPT
# =========================

if MODEL == "mediapipe":
	EXTRACT_SCRIPT = "scripts/extract_mediapipe.py"
elif MODEL == "rtmw":
	EXTRACT_SCRIPT = "scripts/extract_rtmw.py"
	BITRATES = [200] # pour le rtmw on ne regarde que la compression faible
else:
	raise ValueError("Unknown model")

# =========================
# PROCESS FUNCTION
# =========================

def process_video(input_video, tag, q3):
	for br in BITRATES:
		compress, estimation = False, False
		compressed_video = f"{OUTPUT_VIDEO_DIR}/{tag}_{br}k.mp4"
		output_csv = f"{OUTPUT_CSV_DIR}/{MODEL}_{tag}_{br}k.csv"

		if (compressed_video.split("/")[-1] not in os.listdir("videos/compressed")):
			compress = True
		if (q3 and (output_csv not in os.listdir("csv"))):
			estimation = True

		if (compress or (estimation and not q3)): print(f"\n=== Bitrate {br} kbps | {tag} | {MODEL} ===")

		# 1. Compress video
		if (compress):
			subprocess.run([
				FFMPEG, "-y",
				"-i", input_video,
				"-c:v", "libx264",
				"-b:v", f"{br}k",
				compressed_video
			], check=True)

		# 2. Run pose estimation
		if (q3 and estimation):
			subprocess.run([
				PYTHON,
				EXTRACT_SCRIPT,
				compressed_video,
				output_csv,
				SIDE,
				SHOW
			], check=True)

# =========================
# RUN
# =========================

if __name__ == "__main__":
	process_video(VIDEO_ORIGINAL, "original", True)
	process_video(VIDEO_ANON, "anonymized", True)
