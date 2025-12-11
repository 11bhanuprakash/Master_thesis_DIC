import os
import sys
from os.path import abspath
sys.path.extend([abspath(".")])
import cv2
import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from glob import glob
from tifffile import imread, imwrite
import imageio.v2 as imageio
import logging
import muDIC as dic
import pandas as pd
import json


# =========================================
# --- 1. ALIGNMENT USING MARKERS ---
# =========================================
def detect_markers(image):
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray[:100, :] = 255  # remove timestamp
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


def align_images(folder):
    print("\n=== ALIGNING IMAGES BASED ON REFERENCE MARKERS ===")
    output_folder = os.path.join(folder, 'aligned')
    os.makedirs(output_folder, exist_ok=True)
    image_paths = sorted(glob(os.path.join(folder, '*.tif')))
    ref_img = imread(image_paths[0])
    ref_pts = detect_markers(ref_img)
    assert ref_pts is not None and len(ref_pts) == 5, "Reference markers not detected properly."

    for path in image_paths:
        img = imread(path)
        moving_pts = detect_markers(img)
        if moving_pts is None or len(moving_pts) < 5:
            print(f"Skipping {os.path.basename(path)} - Not enough markers")
            continue
        matrix = cv2.estimateAffinePartial2D(moving_pts[:5], ref_pts[:5], method=cv2.RANSAC)[0]
        aligned = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
        imageio.imwrite(os.path.join(output_folder, os.path.basename(path)), aligned.astype(np.uint8))
        print(f"Aligned: {os.path.basename(path)}")

    print("All frames aligned and saved in:", output_folder)
    return output_folder


# =========================================
# --- 2. HISTOGRAM NORMALIZATION ---
# =========================================
def normalize_histograms(path, x_center, y_center, radius):
    print("\n=== PERFORMING HISTOGRAM NORMALIZATION UNDER DETECTED CIRCLE ===")
    image_files = sorted(glob(os.path.join(path, '*.tif')))
    out_dir = os.path.join(path, "circle_hist_matched")
    os.makedirs(out_dir, exist_ok=True)

    def to_gray_native(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    def dtype_max_for(arr):
        if np.issubdtype(arr.dtype, np.integer):
            return np.iinfo(arr.dtype).max
        return int(np.nanmax(arr))

    def circle_mask(shape, cx, cy, r):
        H, W = shape
        yy, xx = np.ogrid[:H, :W]
        return (xx - cx)**2 + (yy - cy)**2 <= r**2

    def masked_cdf_uN(img_uN, mask, levels):
        vals = img_uN[mask].ravel()
        hist = np.bincount(vals, minlength=levels).astype(np.float64)
        cdf = np.cumsum(hist)
        if cdf[-1] > 0:
            cdf /= cdf[-1]
        return cdf

    def cdf_match_lut(src_cdf, ref_cdf):
        levels = len(src_cdf)
        lut = np.zeros(levels, dtype=np.uint32)
        j = 0
        for i in range(levels):
            while j < levels-1 and ref_cdf[j] < src_cdf[i]:
                j += 1
            lut[i] = j
        return lut

    ref_image_path = image_files[0]
    ref_native = to_gray_native(imread(ref_image_path))
    dtype_max = dtype_max_for(ref_native)
    levels = dtype_max + 1

    mask_circ = circle_mask(ref_native.shape, x_center, y_center, radius)
    ref_cdf = masked_cdf_uN(ref_native, mask_circ, levels)

    for fp in image_files:
        img = to_gray_native(imread(fp))
        if img.dtype != ref_native.dtype:
            img = img.astype(ref_native.dtype)
        src_cdf = masked_cdf_uN(img, mask_circ, levels)
        lut = cdf_match_lut(src_cdf, ref_cdf)
        corrected = lut[img]
        corrected = corrected.astype(ref_native.dtype)
        imwrite(os.path.join(out_dir, os.path.basename(fp)), corrected)

    print(f"Circle-matched images saved to: {out_dir}")
    return out_dir


# =========================================
# --- 3. DIC ANALYSIS ---
# =========================================
def run_dic_pipeline(path, roi_shape, config, cfg_file, detect_circle=True, normalize=False):
    print("\n=== RUNNING DIC ANALYSIS ===")
    logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)
    images = dic.IO.image_stack_from_folder(path, file_type='.tif')
    image_files = sorted(glob(os.path.join(path, '*.tif')))
    if not image_files:
        raise RuntimeError("No .tif images found in specified path.")

    if detect_circle and roi_shape == 'C':
        ref_image_path = image_files[0]
        ref_img = imread(ref_image_path)
        if ref_img.ndim == 3:
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        ref_img = cv2.normalize(ref_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        blurred = cv2.GaussianBlur(ref_img, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
            param1=50, param2=30, minRadius=400, maxRadius=650
        )
        if circles is None:
            raise RuntimeError("Circle could not be detected. Please check image quality.")

        circles = np.round(circles[0, :]).astype("int")
        x_center, y_center, radius = circles[0]
        margin = 0.6
        x_start, x_end = int(x_center - radius * margin), int(x_center + radius * margin)
        y_start, y_end = int(y_center - radius * margin), int(y_center + radius * margin)

        # ask for histogram normalization
        if normalize:
            path = normalize_histograms(path, x_center, y_center, radius)
            images = dic.IO.image_stack_from_folder(path, file_type='.tif')

        # plot ROI
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(ref_img, cmap='gray')
        rect = patches.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
                                 linewidth=2, edgecolor='r', facecolor='none', label='Auto ROI')
        circle = plt.Circle((x_center, y_center), radius, color='blue', fill=False,
                            linewidth=1.5, linestyle='--', label='Detected Circle')
        ax.add_patch(rect)
        ax.add_patch(circle)
        ax.legend()
        plt.tight_layout()

    else:
        print("Skipping auto circle detection (user selected N or rectangular ROI).")
        if roi_shape == 'C':
            x_start, x_end, y_start, y_end = 250, 850, 250, 850
        else:
            # x_start, x_end, y_start, y_end = 100, 430, 50, 660
            x_start, x_end, y_start, y_end = 100, 420, 55, 660
            

    mesher = dic.Mesher(deg_e=3, deg_n=3, type="Q4")
    mesh = mesher.mesh(images, Xc1=x_start, Xc2=x_end, Yc1=y_start, Yc2=y_end, n_ely=config["n_ely"], n_elx=config["n_elx"], GUI=False)

    settings = dic.DICInput(mesh, images)
    settings.max_nr_im = 500
    settings.ref_update = [30]
    settings.maxit = 20
    settings.tol = 1.e-6
    settings.interpolation_order = 4
    settings.store_internals = True
    settings.noconvergence = "ignore"

    job = dic.DICAnalysis(settings)
    dic_results = job.run()

    fields = dic.post.viz.Fields(dic_results, upscale=10)
    disp = fields.disp()
    Ux, Uy = disp[0,0,:,:,20], disp[0,1,:,:,20]
    Umag = np.sqrt(Ux**2 + Uy**2)
    x, y = fields.coords()[0,0,:,:,1], fields.coords()[0,1,:,:,1]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    cf1 = ax1.contourf(x, y, Ux, 50, cmap='coolwarm'); plt.colorbar(cf1, ax=ax1, label="Uₓ (px)")
    cf2 = ax2.contourf(x, y, Uy, 50, cmap='viridis'); plt.colorbar(cf2, ax=ax2, label="Uᵧ (px)")
    cf3 = ax3.contourf(x, y, Umag, 50, cmap='plasma'); plt.colorbar(cf3, ax=ax3, label="|U| (px)")
    for a, t in zip([ax1, ax2, ax3],
                    ["X-direction Displacement (Uₓ)",
                     "Y-direction Displacement (Uᵧ)",
                     "Total Displacement Magnitude |U|"]):
        a.set_title(t); a.set_xlabel("X (px)"); a.set_ylabel("Y (px)"); a.set_aspect('equal')
    plt.tight_layout()
    # plt.show()
    
    df = pd.DataFrame({
    "x": x.flatten(),
    "y": y.flatten(),
    "Ux": Ux.flatten(),
    "Uy": Uy.flatten(),
    "Umag": Umag.flatten()
    })
    
    # Define a common export folder for CSVs
    #ADJUST THE PATH TO SAVE CSV FILES same as Below
    csv_output_folder = r"C:\Users\konat\Desktop\Master_Thesis_DIC(Code)\Examples\CSV files"
    os.makedirs(csv_output_folder, exist_ok=True)

 #   csv_path = os.path.join(csv_output_folder, f"displacement_field_{config['n_ely']}x{config['n_elx']}.csv")
    # Use the JSON file name (without extension) for CSV
    json_base = os.path.splitext(os.path.basename(cfg_file))[0]
    csv_path = os.path.join(csv_output_folder, f"{json_base}.csv")

    df.to_csv(csv_path, index=False)
    print(f" Displacement data saved to: {csv_path}")


# =========================================
# --- 4. MAIN PROGRAM ---
# =========================================


def main():
    print("\n=== Digital Image Correlation Batch Automation ===")
    
    # Folder containing all your JSON config files
    # json_folder = r"C:\Users\konat\Desktop\Master_Thesis_DIC(Code)\Examples\JSON"
    json_folder = r"C:\Users\konat\Desktop\Master_Thesis_DIC(Code)\Examples\JSON_1"

    # Collect all .json files in that folder
    config_files = sorted(glob(os.path.join(json_folder, "*.json")))

    if not config_files:
        print(f"No JSON files found in {json_folder}")
        return

    for cfg_file in config_files:
        print(f"\n=== Running pipeline for {cfg_file} ===")
        with open(cfg_file, 'r') as f:
            config = json.load(f)

        base_folder = config["base_folder"]
        has_markers = config.get("has_reference_markers", False)
        roi_shape = config.get("roi_shape", "C").upper()
        normalize = config.get("normalize_histogram", False)

        if roi_shape not in ['C', 'R']:
            print(f"Invalid roi_shape in {cfg_file}, skipping.")
            continue

        if has_markers:
            aligned_folder = align_images(base_folder)
            data_folder = aligned_folder
            detect_circle = True
        else:
            data_folder = base_folder
            detect_circle = False

        run_dic_pipeline(data_folder, roi_shape, config, cfg_file, detect_circle, normalize)

    print("\n=== All configurations completed successfully ===")



if __name__ == "__main__":
    main()
