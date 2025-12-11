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

def compute_angles(folder):
    image_paths = sorted(glob(os.path.join(folder, '*.tif')))
    ref_img = imread(image_paths[0])
    ref_pts = detect_markers(ref_img)
    assert ref_pts is not None and len(ref_pts) == 5, f"Markers not detected in {folder}"

    pt1_ref, pt2_ref, pt3_ref, pt4_ref = ref_pts[:4]
    vec1_ref = pt3_ref - pt1_ref  # 1 → 3
    vec2_ref = pt4_ref - pt2_ref  # 2 → 4

    angle_vec1, angle_vec2, indices = [], [], []
    for i, path in enumerate(image_paths):
        img = imread(path)
        pts = detect_markers(img)
        if pts is None or len(pts) < 5:
            print(f"Skipping {os.path.basename(path)} - Not enough markers")
            continue
        pt1, pt2, pt3, pt4 = pts[:4]
        vec1 = pt3 - pt1
        vec2 = pt4 - pt2

        dot1 = np.dot(vec1, vec1_ref)
        norm1 = np.linalg.norm(vec1) * np.linalg.norm(vec1_ref)
        theta1 = np.degrees(np.arccos(np.clip(dot1 / (norm1 + 1e-10), -1.0, 1.0)))
        angle_vec1.append(theta1)

        dot2 = np.dot(vec2, vec2_ref)
        norm2 = np.linalg.norm(vec2) * np.linalg.norm(vec2_ref)
        theta2 = np.degrees(np.arccos(np.clip(dot2 / (norm2 + 1e-10), -1.0, 1.0)))
        angle_vec2.append(theta2)

        indices.append(i)
    return indices, angle_vec1, angle_vec2, ref_img, ref_pts

# === Paths ===
folder_raw = r'C:\Users\konat\Desktop\Thesis\Digtal Image Correlation\muDIC-master\Master_thesis\Examples\set2'
folder_aligned = r'C:\Users\konat\Desktop\Thesis\Digtal Image Correlation\muDIC-master\Master_thesis\Examples\set2\aligned'

# === Process both folders ===
frames_raw, angles1_raw, angles2_raw, ref_img, ref_pts = compute_angles(folder_raw)
frames_aligned, angles1_aligned, angles2_aligned, _, _ = compute_angles(folder_aligned)

# === Visualize Detected Markers on Reference Image from Raw ===
ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2RGB) if len(ref_img.shape) == 2 else ref_img.copy()
for idx, (x, y) in enumerate(ref_pts):
    x, y = int(x), int(y)
    cv2.circle(ref_img_rgb, (x, y), 10, (0, 255, 0), 2)
    cv2.circle(ref_img_rgb, (x, y), 3, (0, 0, 255), -1)
    cv2.putText(ref_img_rgb, str(idx + 1), (x + 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)

plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(ref_img_rgb, cv2.COLOR_BGR2RGB))
plt.title("Reference Image with Numbered Detected Markers")
plt.axis("off")

# === Angle Plot with Colors ===
plt.figure(figsize=(12, 6))
plt.plot(frames_raw, angles1_raw, 'o--', color='#1f77b4', label='Before Alignment: Angle 1→3')
plt.plot(frames_raw, angles2_raw, 'x--', color='orange', label='Before Alignment: Angle 2→4')
plt.plot(frames_aligned, angles1_aligned, 'o-', color='#1f77b4', label='After Alignment: Angle 1→3')
plt.plot(frames_aligned, angles2_aligned, 'x-', color='orange', label='After Alignment: Angle 2→4')

plt.xlabel("Frame Index")
plt.ylabel("Angle (degrees)")
plt.title("Change in Angles Before and After Alignment")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
