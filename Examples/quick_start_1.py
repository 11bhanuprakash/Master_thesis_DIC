import sys
from os.path import abspath
sys.path.extend([abspath(".")])

import muDIC as dic
import logging
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# For scale bar
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import numpy as np

# Set logging level
logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)

# Path to folder containing images
#path = r'./Sample12/'
path = r'C:\Users\konat\Desktop\Thesis\Digtal Image Correlation\muDIC-master\Master_thesis\Examples\sample12'

# Load images
images = dic.IO.image_stack_from_folder(path, file_type='.tif')

# Generate mesh (set GUI=True if you want interactive selection)
#mesher = dic.Mesher(deg_e=3, deg_n=3,type="q4")
mesher = dic.Mesher(deg_e=3, deg_n=3, type="spline")

mesh = mesher.mesh(images, GUI=True, n_ely=10, n_elx=8)
#mesh = mesher.mesh(images, GUI=True, n_ely=10, n_elx=6)
#mesh = mesher.mesh(images, GUI=True, n_ely=20, n_elx=10)

# Instantiate settings
settings = dic.DICInput(mesh, images)
settings.max_nr_im = 500
settings.ref_update = [15]
settings.maxit = 20
settings.tol = 1.e-6
settings.interpolation_order = 4
settings.store_internals = True
settings.noconvergence = "ignore"

# Run DIC analysis
job = dic.DICAnalysis(settings)
dic_results = job.run()

# Post-processing
fields = dic.post.viz.Fields(dic_results, upscale=10)
viz = dic.Visualizer(fields, images=images)

# Show displacement field (component (1,1) is total displacement magnitude)
#viz.show(field="displacement", component=(1,1), frame=-1)


# Extract displacements and coordinates
disp = fields.disp()
# print("-----------------------------------------------------")
# print(disp)
# print("-----------------------------------------------------")


# Get data from fields object
disp_x = disp[0, 0, :, :, -1]  # Ux
disp_y = disp[0, 1, :, :, -1]  # Uy
disp_mag = np.sqrt(disp_x**2 + disp_y**2)  # |U|
x = fields.coords()[0, 0, :, :, -1] 
y = fields.coords()[0, 1, :, :, -1]
# print(x)
# print(y)

# Scale bar config
mm_per_pixel = 0.1
bar_length_mm = 10
bar_length_px = bar_length_mm / mm_per_pixel
fontprops = fm.FontProperties(size=10)

# Create 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# --- Ux ---
cf1 = ax1.contourf(x, y, disp_x, 50, cmap='coolwarm')
plt.colorbar(cf1, ax=ax1, label="Uₓ (px)")
scalebar1 = AnchoredSizeBar(ax1.transData, bar_length_px, f'{bar_length_mm} mm',
                            'lower right', pad=0.5, color='black',
                            frameon=False, size_vertical=1, fontproperties=fontprops)
ax1.add_artist(scalebar1)
ax1.set_title("X-direction Displacement (Uₓ)")
ax1.set_xlabel("X (px)")
ax1.set_ylabel("Y (px)")
ax1.set_aspect('equal')

# --- Uy ---
cf2 = ax2.contourf(x, y, disp_y, 50, cmap='viridis')
plt.colorbar(cf2, ax=ax2, label="Uᵧ (px)")
scalebar2 = AnchoredSizeBar(ax2.transData, bar_length_px, f'{bar_length_mm} mm',
                            'lower right', pad=0.5, color='black',
                            frameon=False, size_vertical=1, fontproperties=fontprops)
ax2.add_artist(scalebar2)
ax2.set_title("Y-direction Displacement (Uᵧ)")
ax2.set_xlabel("X (px)")
ax2.set_ylabel("Y (px)")
ax2.set_aspect('equal')

# --- |U| Magnitude ---
cf3 = ax3.contourf(x, y, disp_mag, 50, cmap='plasma')
plt.colorbar(cf3, ax=ax3, label="|U| (px)")
scalebar3 = AnchoredSizeBar(ax3.transData, bar_length_px, f'{bar_length_mm} mm',
                            'lower right', pad=0.5, color='black',
                            frameon=False, size_vertical=1, fontproperties=fontprops)
ax3.add_artist(scalebar3)
ax3.set_title("Total Displacement Magnitude (|U|)")
ax3.set_xlabel("X (px)")
ax3.set_ylabel("Y (px)")
ax3.set_aspect('equal')

plt.tight_layout()
plt.show()
