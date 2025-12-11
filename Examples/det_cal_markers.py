import cv2
import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt

#ADJUST THE PATH TO YOUR FOLDER CONTAINING THE IMAGES same as Below


path = r'C:\Users\konat\Desktop\Thesis\Digtal Image Correlation\muDIC-master\Master_thesis\Examples\set2\001.tif'

img = imread(path)

# --- normalize for display (optional but helps saturation/contrast) ---
# If the image is 16-bit, this brings it to 0-255.
if img.dtype != np.uint8:
    img_disp = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
else:
    img_disp = img.copy()

# grayscale copy only for detection
gray = img_disp if len(img_disp.shape) == 2 else cv2.cvtColor(img_disp, cv2.COLOR_BGR2GRAY)
gray[:24, :] = 255  # remove timestamp region in the *detection* image

# threshold for marker detection
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 21, 10)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# draw on a color version of the *display* image
output = img_disp.copy()
if len(output.shape) == 2:
    output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area < 500 or area > 3000:
        continue

    if hierarchy[0][i][3] != -1:  # has parent = ring
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        circularity = 4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2 + 1e-5)
        if 0.75 < circularity < 1.2 and radius > 7:
            cv2.circle(output, (int(x), int(y)), int(radius), (0, 0, 255), 10)
            cv2.circle(output, (int(x), int(y)), 3, (0, 0, 255), -1)

# show with correct RGB ordering and full 0–255 range
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
plt.title("Ring Marker Detection")
plt.axis("off")
plt.show()
