import cv2
import numpy as np
from tifffile import imread
import imageio.v2 as imageio  # or imageio.imwrite if you’re using imageio 2.x or 3.x
import os
from glob import glob

# === CONFIG ===

#ADJUST THE PATH TO YOUR FOLDER CONTAINING THE IMAGES same as Below

folder = r'C:\Users\konat\Desktop\Thesis\Digtal Image Correlation\muDIC-master\Master_thesis\Examples\set2'
# folder = r'C:\Users\konat\Desktop\Thesis\#Resources\14_HM_tif'
output_folder = os.path.join(folder, 'aligned')
os.makedirs(output_folder, exist_ok=True)
image_paths = sorted(glob(os.path.join(folder, '*.tif')))
ref_image_path = image_paths[0]

# === Detection Function ===
def detect_markers(image):
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray[:100, :] = 255  # Remove timestamp
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 10)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centers = []

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 500 or area > 3000:
            continue
        if hierarchy[0][i][3] != -1:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            circularity = 4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2 + 1e-5)
            if 0.75 < circularity < 1.2 and radius > 7:
                centers.append((x, y))

    # Sort by X and then Y to maintain consistent order
    centers = sorted(centers, key=lambda p: (p[0], p[1]))
    return np.float32(centers[:5]) if len(centers) >= 5 else None

# === Get reference marker points ===
ref_img = imread(ref_image_path)
ref_pts = detect_markers(ref_img)

assert ref_pts is not None and len(ref_pts) == 5, "Reference markers not detected properly."

# === Process All Frames ===
for path in image_paths:
    img = imread(path)
    moving_pts = detect_markers(img)
    if moving_pts is None or len(moving_pts) < 5:
        print(f"Skipping {os.path.basename(path)} - Not enough markers")
        continue

    # Compute Affine Transform (3 points only)
    matrix = cv2.estimateAffinePartial2D(moving_pts[:5], ref_pts[:5],method=cv2.RANSAC)[0]
    aligned = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
    
    # # Compute Homography Transform using all 5 marker points
    # matrix, _ = cv2.findHomography(moving_pts, ref_pts)

    # # Apply perspective warp using the homography matrix
    # aligned = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))

    
    # Save aligned image
    filename = os.path.basename(path)
    imageio.imwrite(os.path.join(output_folder, filename), aligned.astype(np.uint8))

    print(f"Aligned: {filename}")

print("All frames aligned and saved in:", output_folder)
