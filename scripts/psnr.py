import cv2
import numpy as np

def compute_psnr(video_ref, video_comp):
    cap_ref = cv2.VideoCapture(video_ref)
    cap_cmp = cv2.VideoCapture(video_comp)

    psnr_values = []

    while True:
        ret1, f1 = cap_ref.read()
        ret2, f2 = cap_cmp.read()

        if not ret1 or not ret2:
            break

        psnr = cv2.PSNR(f1, f2)
        psnr_values.append(psnr)

    cap_ref.release()
    cap_cmp.release()

    return np.mean(psnr_values)


if __name__ == "__main__":
    psnr = compute_psnr("videos/demo.mp4", "videos/demo_500k.mp4")
    print(f"PSNR moyen : {psnr:.2f} dB")
