import sys
from os.path import abspath
sys.path.extend([abspath(".")])
import os
from glob import glob
import muDIC as dic
import logging
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from tifffile import imread

# Set logging level
logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)

# --- load images ---

#ADJUST THE PATH TO YOUR FOLDER CONTAINING THE IMAGES same as Below

# path = r'C:\Users\konat\Desktop\Thesis\Digtal Image Correlation\muDIC-master\Master_thesis\Examples\set2\aligned\circle_hist_matched'
path = r'C:\Users\konat\Desktop\Thesis\Digtal Image Correlation\muDIC-master\Master_thesis\Examples\set2\aligned'


images = dic.IO.image_stack_from_folder(path, file_type='.tif')


# Automatically get the first image in the folder
image_files = sorted(glob(os.path.join(path, '*.tif')))
if not image_files:
    raise RuntimeError(" No .tif images found in the specified path.")

ref_image_path = image_files[0]


ref_img = imread(ref_image_path)

# Ensure it's single channel
if ref_img.ndim == 3:
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

# Normalize and convert to 8-bit
ref_img = cv2.normalize(ref_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

blurred = cv2.GaussianBlur(ref_img, (9, 9), 2)

# Detect circle using Hough Transform
circles = cv2.HoughCircles(
    blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
    param1=50, param2=30, minRadius=400, maxRadius=650
)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    x_center, y_center, radius = circles[0]

    # Define ROI bounding box inside the circle
    margin = 0.7 # 70% of radius
    x_start = int(x_center - radius * margin)
    x_end = int(x_center + radius * margin)
    y_start = int(y_center - radius * margin)
    y_end = int(y_center + radius * margin)

    roi = [[x_start, y_start], [x_end, y_end]]
else:
    raise RuntimeError("Circle could not be detected. Please check image quality.")



# Plot the reference image with bounding box ROI
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(ref_img, cmap='gray')
rect = patches.Rectangle(
    (x_start, y_start),
    x_end - x_start,
    y_end - y_start,
    linewidth=2,
    edgecolor='r',
    facecolor='none',
    label='Auto ROI'
)
circle = plt.Circle(
    (x_center, y_center),
    radius,
    color='blue',
    fill=False,
    linewidth=1.5,
    linestyle='--',
    label='Detected Circle'
)
ax.add_patch(rect)
ax.add_patch(circle)
ax.set_title("Automatically Detected ROI and Circle")
ax.set_xlabel("X (pixels)")
ax.set_ylabel("Y (pixels)")
ax.legend()
plt.tight_layout()
# plt.show()


# --- build mesh with automatic ROI ---
# mesher = dic.Mesher(deg_e=3, deg_n=3, type="spline")
mesher = dic.Mesher(deg_e=3, deg_n=3, type="Q4")  # Q4 allows ROI selection

# mesh = mesher.mesh(images, ROI=roi, GUI=False, n_ely=5, n_elx=4)
mesh = mesher.mesh(images,Xc1=x_start, Xc2=x_end,Yc1=y_start, Yc2=y_end,n_ely=3, n_elx=3,GUI=False)


# print("Xc1 =", mesh.Xc1)
# print("Xc2 =", mesh.Xc2)
# print("Yc1 =", mesh.Yc1)
# print("Yc2 =", mesh.Yc2)

# --- DIC Settings ---
settings = dic.DICInput(mesh, images)
settings.max_nr_im = 500
settings.ref_update = [15]
settings.maxit = 20
settings.tol = 1.e-6
settings.interpolation_order = 4
settings.store_internals = True
settings.noconvergence = "ignore"

job = dic.DICAnalysis(settings)
dic_results = job.run()

# --- post-processing ---
fields = dic.post.viz.Fields(dic_results, upscale=10)

disp = fields.disp()  
Ux = disp[0,0,:,:, -1]
Uy = disp[0,1,:,:, -1]
Umag = np.sqrt(Ux**2 + Uy**2)
x = fields.coords()[0,0,:,:, -1]
y = fields.coords()[0,1,:,:, -1]

# --- plot Ux, Uy, |U| ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6))

cf1 = ax1.contourf(x, y, Ux, 50, cmap='coolwarm')
fig.colorbar(cf1, ax=ax1, label="Uₓ (px)")
ax1.set_title("X‐direction Displacement (Uₓ)")
ax1.set_xlabel("X (px)"); ax1.set_ylabel("Y (px)")
ax1.set_aspect('equal')

cf2 = ax2.contourf(x, y, Uy, 50, cmap='viridis')
fig.colorbar(cf2, ax=ax2, label="Uᵧ (px)")
ax2.set_title("Y‐direction Displacement (Uᵧ)")
ax2.set_xlabel("X (px)"); ax2.set_ylabel("Y (px)")
ax2.set_aspect('equal')

cf3 = ax3.contourf(x, y, Umag, 50, cmap='plasma')
fig.colorbar(cf3, ax=ax3, label="|U| (px)")
ax3.set_title("Total Displacement Magnitude |U|")
ax3.set_xlabel("X (px)"); ax3.set_ylabel("Y (px)")
ax3.set_aspect('equal')

plt.tight_layout()

# --- corner tracking ---
x_all = fields.coords()[0,0,:,:,:]  # shape: [rows, cols, frames]
y_all = fields.coords()[0,1,:,:,:]

top_left_x = x_all[0, 0, :]
top_left_y = y_all[0, 0, :]
top_right_x = x_all[-1, 0, :]
top_right_y = y_all[-1, 0, :]
bottom_left_x = x_all[0, -1, :]
bottom_left_y = y_all[0, -1, :]
bottom_right_x = x_all[-1, -1, :]
bottom_right_y = y_all[-1, -1, :]

# --- plot corner motion ---
plt.figure(figsize=(8, 6))
plt.plot(top_left_x, top_left_y, 'o-', label='Top-Left')
plt.plot(top_right_x, top_right_y, 'o-', label='Top-Right')
plt.plot(bottom_left_x, bottom_left_y, 'o-', label='Bottom-Left')
plt.plot(bottom_right_x, bottom_right_y, 'o-', label='Bottom-Right')
plt.title("Motion of ROI Corners Over Time")
plt.xlabel("X position (px)")
plt.ylabel("Y position (px)")
plt.gca().invert_yaxis()
plt.grid(True)
plt.legend()
plt.tight_layout()

# --- diameter calculations ---
D1 = np.sqrt((top_left_x - bottom_right_x)**2 + (top_left_y - bottom_right_y)**2)
D2 = np.sqrt((top_right_x - bottom_left_x)**2 + (top_right_y - bottom_left_y)**2)

real_world_diameter_mm = 45.0
reference_pixel_diameter = D1[0]
scale = real_world_diameter_mm / reference_pixel_diameter
D1_mm = D1 * scale
D2_mm = D2 * scale

# --- plot diameters ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(D1_mm, label='D1: Top-Left ↔ Bottom-Right', marker='o')
ax.plot(D2_mm, label='D2: Top-Right ↔ Bottom-Left', marker='s')
ax.set_title("Diagonal Diameters (D1 & D2) Over Time (in mm)")
ax.set_xlabel("Frame Number")
ax.set_ylabel("Distance (mm)")
ax.grid(True)
ax.legend()
scale_text = f"Scale: {scale:.3f} mm/pixel"
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.text(
    xlim[1], ylim[0], scale_text,
    ha='right', va='bottom',
    fontsize=10, color='black',
    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.2')
)

plt.tight_layout()
plt.show()
