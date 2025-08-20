from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
import os
import time

## Used to time run-time
start_time = time.time()

def PSNR(original, noisy):
    mse = np.mean((original - noisy) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def SSIM(original, noisy, channel_axis=2):
    # Convert images to grayscale
    grayA = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
    
    # Compute SSIM between the two images
    score, _ = ssim(grayA, grayB, full=True)
    return score

def main():
    base_dir = "SIDD_Small_sRGB_Only/Data"  # <-- your dataset root

    for name in sorted(os.listdir(base_dir)):
        scene_dir = os.path.join(base_dir, name)
        if not os.path.isdir(scene_dir):
            continue

        scene_instance = name.split('_', 1)[0]

        gt_path = os.path.join(scene_dir, "GT_SRGB_010.PNG")
        noisy_path = os.path.join(scene_dir, "NOISY_SRGB_010.PNG")

        if not (os.path.exists(gt_path) and os.path.exists(noisy_path)):
            # print(f"Skipping {name}: missing GT or NOISY image")
            continue

        original = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        noisy = cv2.imread(noisy_path, cv2.IMREAD_COLOR)
        # original = cv2.imread("GT_SRGB_010.png")
        # noisy = cv2.imread("NOISY_SRGB_010.png", 1)
        psnr = PSNR(original, noisy)
        ssim_score = SSIM(original, noisy)
        print(f"PSNR value for {scene_instance} is {round(psnr, 3)} dB")
        print(f"SSIM score for {scene_instance} is {round(ssim_score, 3)}")
        print()
     
if __name__ == "__main__":
    main()
    
#Calculates time it takes to run code
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.4f} seconds")