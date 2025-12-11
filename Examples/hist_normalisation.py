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
import math
from tifffile import imwrite

# Set logging level
logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)

# --- load images ---

#ADJUST THE PATH TO YOUR FOLDER CONTAINING THE IMAGES same as Below


path = r'C:\Users\konat\Desktop\Thesis\Digtal Image Correlation\muDIC-master\Master_thesis\Examples\set2\aligned'
images = dic.IO.image_stack_from_folder(path, file_type='.tif')

# --- auto-detect circular ROI from reference image ---
# ref_image_path = path + '\\001.tif'
# ref_img = imread(ref_image_path).astype(np.uint8)

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
    margin = 0.6 # 80% of radius
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



# === normalize histograms ONLY under the detected circle and save ===


out_dir = os.path.join(path, "circle_hist_matched")   # output folder
os.makedirs(out_dir, exist_ok=True)

# helper: grayscale w/o normalization (preserves bit-depth)
def to_gray_native(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

# helper: dtype range
def dtype_max_for(arr):
    if np.issubdtype(arr.dtype, np.integer):
        return np.iinfo(arr.dtype).max
    # fallback (shouldn't happen for TIFFs here)
    return int(np.nanmax(arr))

# build circular mask (same size as images)
def circle_mask(shape, cx, cy, r):
    H, W = shape
    yy, xx = np.ogrid[:H, :W]
    return (xx - cx)**2 + (yy - cy)**2 <= r**2

# Compute CDF from masked pixels
def masked_cdf_uN(img_uN, mask, levels):
    vals = img_uN[mask].ravel()
    # bincount over full range
    hist = np.bincount(vals, minlength=levels).astype(np.float64)
    cdf  = np.cumsum(hist)
    if cdf[-1] > 0:
        cdf /= cdf[-1]
    return cdf

# Build LUT that maps src CDF -> ref CDF (monotone)- LUT(lookup table) maping
def cdf_match_lut(src_cdf, ref_cdf):
    levels = len(src_cdf)
    lut = np.zeros(levels, dtype=np.uint32)
    j = 0
    for i in range(levels):
        # move j until ref_cdf[j] >= src_cdf[i]
        while j < levels-1 and ref_cdf[j] < src_cdf[i]:
            j += 1
        lut[i] = j
    # choose output dtype later
    return lut

# --- load reference again at native depth (no cv2.normalize!) ---
ref_native = to_gray_native(imread(ref_image_path))
dtype_max  = dtype_max_for(ref_native)
levels     = dtype_max + 1

# circular mask in reference size
mask_circ = circle_mask(ref_native.shape, x_center, y_center, radius)

# reference CDF inside circle
ref_cdf = masked_cdf_uN(ref_native, mask_circ, levels)

# process every frame
for fp in image_files:
    img = to_gray_native(imread(fp))
    # if any frame has different dtype, cast to reference dtype to share LUT domain
    if img.dtype != ref_native.dtype:
        img = img.astype(ref_native.dtype)

    # src CDF inside the SAME circle region
    src_cdf = masked_cdf_uN(img, mask_circ, levels)

    # build LUT (maps 0..dtype_max -> 0..dtype_max)
    lut = cdf_match_lut(src_cdf, ref_cdf)
    # apply LUT (vectorized)
    corrected = lut[img]

    # cast back to original dtype
    corrected = corrected.astype(ref_native.dtype)

    # save with SAME filename into subfolder
    imwrite(os.path.join(out_dir, os.path.basename(fp)), corrected)

print(f"Circle-matched images saved to: {out_dir}  (filenames unchanged)")

# === PLOTS: histograms BEFORE vs AFTER (inside the blue circle) ===

# helper: masked histogram with fixed #bins for plotting
def masked_hist_binned(img, mask, n_bins=256, max_val=None):
    if max_val is None:
        max_val = np.iinfo(img.dtype).max if np.issubdtype(img.dtype, np.integer) else float(img.max())
    vals = img[mask].ravel()
    hist, _ = np.histogram(vals, bins=n_bins, range=(0, max_val))
    return hist

# collect BEFORE histograms (originals) and AFTER histograms (matched)
pre_hists, post_hists = [], []

# BEFORE (original images in `image_files`)
for fp in image_files:
    img = to_gray_native(imread(fp))
    if img.dtype != ref_native.dtype:
        img = img.astype(ref_native.dtype)
    pre_hists.append(masked_hist_binned(img, mask_circ, n_bins=256, max_val=dtype_max))

# AFTER (matched images we just wrote to out_dir with same filenames)
for fp in image_files:
    img_after = to_gray_native(imread(os.path.join(out_dir, os.path.basename(fp))))
    post_hists.append(masked_hist_binned(img_after, mask_circ, n_bins=256, max_val=dtype_max))

pre_hists  = np.array(pre_hists)
post_hists = np.array(post_hists)

# subplot layout
n_imgs = len(image_files)
n_cols = 3
n_rows = math.ceil(n_imgs / n_cols)

# BEFORE grid
fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3*n_rows))
axes1 = np.array(axes1).ravel()
for i in range(n_imgs):
    axes1[i].plot(pre_hists[i], linewidth=1.2)
    axes1[i].set_title(f'Image {i+1} — BEFORE (inside circle)')
    axes1[i].set_xlabel('Intensity bins')
    axes1[i].set_ylabel('Pixel Count')
    axes1[i].grid(True, alpha=0.3)
for j in range(n_imgs, len(axes1)):
    axes1[j].axis('off')
plt.suptitle('Histograms — BEFORE (masked to detected circle)', fontsize=14)
plt.tight_layout()

# AFTER grid
fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3*n_rows))
axes2 = np.array(axes2).ravel()
for i in range(n_imgs):
    axes2[i].plot(post_hists[i], linewidth=1.2)
    axes2[i].set_title(f'Image {i+1} — AFTER (inside circle)')
    axes2[i].set_xlabel('Intensity bins')
    axes2[i].set_ylabel('Pixel Count')
    axes2[i].grid(True, alpha=0.3)
for j in range(n_imgs, len(axes2)):
    axes2[j].axis('off')
plt.suptitle('Histograms — AFTER circle-based histogram matching', fontsize=14)
plt.tight_layout()


# --- build mesh with automatic ROI ---
# mesher = dic.Mesher(deg_e=3, deg_n=3, type="spline")
mesher = dic.Mesher(deg_e=3, deg_n=3, type="Q4")  # Q4 allows ROI selection

# mesh = mesher.mesh(images, ROI=roi, GUI=False, n_ely=5, n_elx=4)
mesh = mesher.mesh(images,Xc1=x_start, Xc2=x_end,Yc1=y_start, Yc2=y_end,n_ely=3, n_elx=3,GUI=False)



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
#plt.show()




#---------------------------------------------------------------------------------------------------------


# === STATS: prove calibration with mean/variance inside circle ===
import pandas as pd

def masked_stats(arr, mask):
    vals = arr[mask].astype(np.float64)
    m = vals.mean()
    v = vals.var(ddof=0)
    return m, v

def hist_l1_distance(img_a, img_b, mask, n_bins=256, max_val=None):
    if max_val is None:
        max_val = np.iinfo(img_a.dtype).max
    a = img_a[mask].ravel()
    b = img_b[mask].ravel()
    ha, _ = np.histogram(a, bins=n_bins, range=(0, max_val), density=True)
    hb, _ = np.histogram(b, bins=n_bins, range=(0, max_val), density=True)
    return float(np.abs(ha - hb).sum())   # smaller is better

# load reference (before & after)
ref_before = to_gray_native(imread(ref_image_path))
ref_after  = to_gray_native(imread(os.path.join(out_dir, os.path.basename(ref_image_path))))

ref_mean_before, ref_var_before = masked_stats(ref_before, mask_circ)
ref_mean_after,  ref_var_after  = masked_stats(ref_after,  mask_circ)

rows = []
for i, fp in enumerate(image_files, start=1):
    name = os.path.basename(fp)

    # before
    img_b = to_gray_native(imread(fp))
    mean_b, var_b = masked_stats(img_b, mask_circ)
    # after
    img_a = to_gray_native(imread(os.path.join(out_dir, name)))
    mean_a, var_a = masked_stats(img_a, mask_circ)

    # errors vs reference
    mean_err_b = mean_b - ref_mean_before
    var_err_b  = var_b  - ref_var_before
    mean_err_a = mean_a - ref_mean_after
    var_err_a  = var_a  - ref_var_after

    # normalized histogram distance (optional but nice)
    l1_b = hist_l1_distance(img_b, ref_before, mask_circ, n_bins=256, max_val=dtype_max)
    l1_a = hist_l1_distance(img_a, ref_after,  mask_circ, n_bins=256, max_val=dtype_max)

    rows.append({
        "image": name,
        "mean_before": mean_b,
        "var_before":  var_b,
        "mean_after":  mean_a,
        "var_after":   var_a,
        "mean_err_before": mean_err_b,
        "var_err_before":  var_err_b,
        "mean_err_after":  mean_err_a,
        "var_err_after":   var_err_a,
        "hist_L1_before":  l1_b,
        "hist_L1_after":   l1_a
    })

df = pd.DataFrame(rows)

# quick aggregate view
def pct_reduction(col_before, col_after):
    num = (df[col_before].abs().sum() - df[col_after].abs().sum())
    den = df[col_before].abs().sum() + 1e-12
    return 100.0 * num / den

print("\n=== Calibration summary (inside circle) ===")
print(f"Mean error reduction vs ref: {pct_reduction('mean_err_before','mean_err_after'):.2f}%")
print(f"Variance error reduction vs ref: {pct_reduction('var_err_before','var_err_after'):.2f}%")
print(f"Histogram L1 distance reduction: {pct_reduction('hist_L1_before','hist_L1_after'):.2f}%")

# save detailed table
stats_csv = os.path.join(out_dir, "circle_hist_stats.csv")
df.to_csv(stats_csv, index=False)
print(f"Saved per-image stats to: {stats_csv}")





# import matplotlib.pyplot as plt
# import numpy as np

# === Visualization of calibration stats ===
img_ids = np.arange(1, len(df) + 1)

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# ---- MEAN ----
axs[0].plot(img_ids, df["mean_before"], 'o--', color='gray', label='Before Calibration')
axs[0].plot(img_ids, df["mean_after"], 'o-', color='blue', label='After Calibration')
axs[0].axhline(ref_mean_before, color='red', linestyle='--', label='Reference Mean')
for i, (mb, ma) in enumerate(zip(df["mean_before"], df["mean_after"])):
    axs[0].text(i+1, ma + 0.5, f"{ma:.1f}", ha='center', fontsize=8, color='blue')
axs[0].set_title("Mean Intensity Inside Circle (Before vs After Calibration)")
axs[0].set_xlabel("Image #")
axs[0].set_ylabel("Mean Intensity")
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# ---- VARIANCE ----
axs[1].plot(img_ids, df["var_before"], 's--', color='gray', label='Before Calibration')
axs[1].plot(img_ids, df["var_after"], 's-', color='green', label='After Calibration')
axs[1].axhline(ref_var_before, color='red', linestyle='--', label='Reference Variance')
for i, (vb, va) in enumerate(zip(df["var_before"], df["var_after"])):
    axs[1].text(i+1, va + 200, f"{va:.0f}", ha='center', fontsize=8, color='green')
axs[1].set_title("Variance Inside Circle (Before vs After Calibration)")
axs[1].set_xlabel("Image #")
axs[1].set_ylabel("Variance")
axs[1].legend()
axs[1].grid(True, alpha=0.3)

plt.tight_layout()
out_plot = os.path.join(out_dir, "calibration_stats_plot.png")
plt.savefig(out_plot, dpi=300)
plt.show()

print(f"Saved calibration summary plot to: {out_plot}")
