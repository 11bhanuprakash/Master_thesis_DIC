import cv2
import numpy as np
from tifffile import imread
import os
from glob import glob
import matplotlib.pyplot as plt

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
    centers = sorted(centers, key=lambda p: (p[0], p[1]))
    return np.float32(centers[:5]) if len(centers) >= 5 else None

def compute_displacement_sums(folder):
    image_paths = sorted(glob(os.path.join(folder, '*.tif')))
    ref_img = imread(image_paths[0])
    ref_pts = detect_markers(ref_img)
    assert ref_pts is not None and len(ref_pts) == 5, f"Markers not detected in {folder}"

    Ds = []
    frame_indices = []

    for i, path in enumerate(image_paths):
        img = imread(path)
        pts = detect_markers(img)
        if pts is None or len(pts) < 5:
            print(f"Skipping {os.path.basename(path)} - Not enough markers")
            continue

        # Calculate sum of distances from ref_pts to current pts
        distances = [np.linalg.norm(pts[j] - ref_pts[j]) for j in range(5)]
        Ds.append(sum(distances))
        frame_indices.append(i) 

    return frame_indices, Ds

# === Paths ===
folder_raw = r'C:\Users\konat\Desktop\Thesis\Digtal Image Correlation\muDIC-master\Master_thesis\Examples\set2'
folder_aligned = r'C:\Users\konat\Desktop\Thesis\Digtal Image Correlation\muDIC-master\Master_thesis\Examples\set2\aligned'

# === Compute displacement sums ===
frames_raw, Ds1 = compute_displacement_sums(folder_raw)
frames_aligned, Ds2 = compute_displacement_sums(folder_aligned)

# === Plot the displacement over frames ===
plt.figure(figsize=(12, 6))
plt.plot(frames_raw, Ds1, marker='o', color='red', label='Raw: Sum of Marker Displacements (Ds1)')
plt.plot(frames_aligned, Ds2, marker='s', color='blue', label='Aligned: Sum of Marker Displacements (Ds2)')

plt.xlabel("Frame Index")
plt.ylabel("Sum of Distances (pixels)")
plt.title("Total Displacement of 5 Markers vs Reference Frame")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
