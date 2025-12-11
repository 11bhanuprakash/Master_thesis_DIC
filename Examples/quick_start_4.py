import sys
from os.path import abspath
sys.path.extend([abspath(".")])

import muDIC as dic
import logging
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import numpy as np

# Set logging level
logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)

# Path to folder containing images
path = r'C:\Users\konat\Desktop\Thesis\Digtal Image Correlation\muDIC-master\Master_thesis\Examples\cam_tif_images'

# Load images
images = dic.IO.image_stack_from_folder(path, file_type='.tif')

# Generate mesh (GUI selection occurs here)
mesher = dic.Mesher(deg_e=3, deg_n=3, type="spline")
mesh = mesher.mesh(images, GUI=True, n_ely=10, n_elx=8)

# Instantiate and run DIC analysis
settings = dic.DICInput(mesh, images)
job = dic.DICAnalysis(settings)
dic_results = job.run()

# Post-processing
fields = dic.post.viz.Fields(dic_results, upscale=10)
viz = dic.Visualizer(fields, images=images)

# Calibration factor
mm_per_pixel = 0.1  # Set your calibration factor

# Get data
coords = fields.coords()
disp = fields.disp()
num_frames = coords.shape[-1]

# Get initial reference coordinates
x_ref = coords[0, 0, :, :, 0]
y_ref = coords[0, 1, :, :, 0]

# Initial dimensions
initial_width_px = x_ref.max() - x_ref.min()
initial_length_px = y_ref.max() - y_ref.min()
initial_width_mm = initial_width_px * mm_per_pixel
initial_length_mm = initial_length_px * mm_per_pixel

print(f"Initial width: {initial_width_mm:.2f} mm ({initial_width_px:.2f} px)")
print(f"Initial length: {initial_length_mm:.2f} mm ({initial_length_px:.2f} px)")

# Arrays to store frame-by-frame dimensions
lengths = []
widths = []
diagonals = []

# Loop through frames
for frame_idx in range(num_frames):
    x_def = x_ref + disp[0, 0, :, :, frame_idx]
    y_def = y_ref + disp[0, 1, :, :, frame_idx]

    width_px = x_def.max() - x_def.min()
    length_px = y_def.max() - y_def.min()
    diagonal_px = np.sqrt(width_px**2 + length_px**2)

    width_mm = width_px * mm_per_pixel
    length_mm = length_px * mm_per_pixel
    diagonal_mm = diagonal_px * mm_per_pixel

    widths.append(width_mm)
    lengths.append(length_mm)
    diagonals.append(diagonal_mm)

print(f"Diagonal length: {diagonal_mm:.2f} mm ({diagonal_px:.2f} px)")

# Convert to arrays
lengths = np.array(lengths)
widths = np.array(widths)
diagonals = np.array(diagonals)
frames = np.arange(num_frames)

# Plot all three: width, length, diagonal
plt.figure(figsize=(10, 5))
# plt.plot(frames, lengths, label='Length (mm)', marker='o')
# plt.plot(frames, widths, label='Width (mm)', marker='s')
plt.plot(frames, diagonals, label='Diagonal (mm)', marker='^')
plt.xlabel("Image Sample (Frame Number)")
plt.ylabel("Length / Width / Diagonal (mm)")
plt.title("ROI Dimensions Across Image Samples")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
