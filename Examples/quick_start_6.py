import sys
from os.path import abspath
sys.path.extend([abspath(".")])

import muDIC as dic
import logging
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# Set logging level
logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)

# --- load images and build mesh exactly as before ---
# path = r'C:\Users\konat\Desktop\Thesis\Digtal Image Correlation\muDIC-master\Master_thesis\Examples\set2\aligned'
# path = r'C:\Users\konat\Desktop\Thesis\Digtal Image Correlation\muDIC-master\Master_thesis\Examples\set2\aligned\circle_hist_matched'
path = r'C:\Users\konat\Desktop\Thesis\#Resources\no_grayscale_shifts\no_grayscale_shifts'
# path = r'C:\Users\konat\Desktop\Thesis\#Resources\grayscale_shifts\grayscale_shifts'

#path = r'C:\Users\konat\Desktop\Thesis\#Resources\sample13\sample13\res\050_ref=400'
images = dic.IO.image_stack_from_folder(path, file_type='.tif')
# mesher = dic.Mesher(deg_e=3, deg_n=3, type="spline")
mesher = dic.Mesher(deg_e=3, deg_n=3, type="Q4")  # Using Q4 elements
# mesh = mesher.mesh(images, GUI=True, n_ely=5, n_elx=4)
mesh = mesher.mesh(images,Xc1 = 100,Xc2 = 430,Yc1 = 50,Yc2 = 660,n_ely=5,n_elx=3, GUI=False)


settings = dic.DICInput(mesh, images)
settings.max_nr_im = 500
settings.ref_update = [30]
settings.maxit = 5
settings.tol = 1.e-6
settings.interpolation_order = 4
settings.store_internals = True
settings.noconvergence = "ignore"

job = dic.DICAnalysis(settings)
dic_results = job.run()

# post-processing
fields = dic.post.viz.Fields(dic_results, upscale=10)

# extract displacement fields at final frame
disp = fields.disp()  
Ux = disp[0,0,:,:, -1]
Uy = disp[0,1,:,:, -1]


Umag = np.sqrt(Ux**2 + Uy**2)
x = fields.coords()[0,0,:,:, -1]
y = fields.coords()[0,1,:,:, -1]

# print("x =",x)
# print("y =",y)

# plot Ux, Uy, |U|
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

# Extract X and Y coordinate arrays over time
x_all = fields.coords()[0,0,:,:,:]  # shape: [rows, cols, frames]
y_all = fields.coords()[0,1,:,:,:]  # same

# frames = x_all.shape[2]

# Prepare corner coordinate trajectories
top_left_x = x_all[0, 0, :]
top_left_y = y_all[0, 0, :]

top_right_x = x_all[-1, 0, :] #beacuse of transpose in mesher
top_right_y = y_all[-1, 0, :] #beacuse of transpose in mesher

bottom_left_x = x_all[0, -1, :] #beacuse of transpose in mesher
bottom_left_y = y_all[0, -1, :] #beacuse of transpose in mesher

bottom_right_x = x_all[-1, -1, :]
bottom_right_y = y_all[-1, -1, :]

# Plot
plt.figure(figsize=(8, 6))
plt.plot(top_left_x, top_left_y, 'o-', label='Top-Left')
plt.plot(top_right_x, top_right_y, 'o-', label='Top-Right')
plt.plot(bottom_left_x, bottom_left_y, 'o-', label='Bottom-Left')
plt.plot(bottom_right_x, bottom_right_y, 'o-', label='Bottom-Right')

plt.title("Motion of ROI Corners Over Time")
plt.xlabel("X position (px)")
plt.ylabel("Y position (px)")
plt.gca().invert_yaxis()  # Image origin is top-left
plt.grid(True)
plt.legend()
plt.tight_layout()


# Calculate D1: Top-Left to Bottom-Right
D1 = np.sqrt((top_left_x - bottom_right_x)**2 + (top_left_y - bottom_right_y)**2)

# Calculate D2: Top-Right to Bottom-Left
D2 = np.sqrt((top_right_x - bottom_left_x)**2 + (top_right_y - bottom_left_y)**2)

# Define real-world reference diameter in mm
real_world_diameter_mm = 40.0

# Use first frame of D1 as reference pixel distance
reference_pixel_diameter = D1[0]  # or D2[0], assuming both are similar

# Compute scale: mm per pixel
scale = real_world_diameter_mm / reference_pixel_diameter

# Convert D1 and D2 from pixels to millimeters
D1_mm = D1 * scale
D2_mm = D2 * scale


import matplotlib.pyplot as plt

# Plot the scaled diameters
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(D1_mm, label='D1: Top-Left ↔ Bottom-Right', marker='o')
ax.plot(D2_mm, label='D2: Top-Right ↔ Bottom-Left', marker='s')

ax.set_title("Diagonal Diameters (D1 & D2) Over Time (in mm)")
ax.set_xlabel("Frame Number")
ax.set_ylabel("Distance (mm)")
ax.grid(True)
ax.legend()
# Add scale text to the plot
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
