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
path = r'C:\Users\konat\Desktop\Thesis\Digtal Image Correlation\muDIC-master\Master_thesis\Examples\sample12'

# Load images
images = dic.IO.image_stack_from_folder(path, file_type='.tif')

# Generate mesh (GUI selection occurs here)
mesher = dic.Mesher(deg_e=3, deg_n=3, type="spline")
mesh = mesher.mesh(images, GUI=True, n_ely=10, n_elx=8)

# Instantiate and run DIC analysis
settings = dic.DICInput(mesh, images)
# ... configure settings if needed ...
job = dic.DICAnalysis(settings)
dic_results = job.run()

# Post-processing
fields = dic.post.viz.Fields(dic_results, upscale=10)
viz = dic.Visualizer(fields, images=images)

# Calibration factor
mm_per_pixel = 0.1  # Set your calibration factor

num_frames = fields.coords().shape[-1]


coords = fields.coords()
disp = fields.disp()

num_frames = fields.coords().shape[-1]

# Print initial width and length (reference, frame 0)
x_ref = coords[0, 0, :, :, 0]
y_ref = coords[0, 1, :, :, 0]
initial_width_px = x_ref.max() - x_ref.min()
initial_length_px = y_ref.max() - y_ref.min()
initial_width_mm = initial_width_px * mm_per_pixel
initial_length_mm = initial_length_px * mm_per_pixel

print(f"Initial width: {initial_width_mm:.2f} mm ({initial_width_px:.2f} px)")
print(f"Initial length: {initial_length_mm:.2f} mm ({initial_length_px:.2f} px)")

lengths = []
widths = []

for frame_idx in range(num_frames):
    x_def = x_ref + disp[0, 0, :, :, frame_idx]
    y_def = y_ref + disp[0, 1, :, :, frame_idx]
    
    width_px = x_def.max() - x_def.min()
    length_px = y_def.max() - y_def.min()
    # width_px = x_def
    # length_px = y_def
    width_mm = width_px * mm_per_pixel
    length_mm = length_px * mm_per_pixel
    
    lengths.append(length_mm)
    widths.append(width_mm)

lengths = np.array(lengths)
widths = np.array(widths)
frames = np.arange(num_frames)

plt.figure(figsize=(10, 5))
plt.plot(frames, lengths, label='Length (mm)', marker='o')
plt.plot(frames, widths, label='Width (mm)', marker='s')
plt.xlabel("Image Sample (Frame Number)")
plt.ylabel("Length / Width (mm)")
plt.title("ROI Length and Width Across Image Samples")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#----------------------------------------------------------------------------------------------------------
#--- Automatically calculate initial length and width from the first frame ---
# coords = fields.coords()
# x0 = coords[0, 0, :, :, 0]   # x-coords, first frame
# y0 = coords[0, 1, :, :, 0]   # y-coords, first frame

# initial_width_px = x0.max() - x0.min()
# initial_length_px = y0.max() - y0.min()
# initial_width_mm = initial_width_px * mm_per_pixel
# initial_length_mm = initial_length_px * mm_per_pixel

# print(f"Initial width: {initial_width_mm:.2f} mm ({initial_width_px:.2f} px)")
# print(f"Initial length: {initial_length_mm:.2f} mm ({initial_length_px:.2f} px)")

# lengths = []
# widths = []

# for frame_idx in range(num_frames):
#     x = coords[0, 0, :, :, frame_idx]
#     y = coords[0, 1, :, :, frame_idx]
    
#     width_px = x.max() - x.min()
#     length_px = y.max() - y.min()
#     width_mm = width_px * mm_per_pixel
#     length_mm = length_px * mm_per_pixel
    
#     lengths.append(length_mm)
#     widths.append(width_mm)

# coords = fields.coords()     # shape: (1, 2, n_elx, n_ely, n_frames)
# disp = fields.disp()        # shape: (1, 2, n_elx, n_ely, n_frames)

# lengths = []
# widths = []

# for frame_idx in range(num_frames):
#     # Deformed node positions for current frame
#     x_def = coords[0, 0, :, :, 0] + disp[0, 0, :, :, frame_idx]
#     y_def = coords[0, 1, :, :, 0] + disp[0, 1, :, :, frame_idx]
    
#     width_px = x_def.max() - x_def.min()
#     length_px = y_def.max() - y_def.min()
#     width_mm = width_px * mm_per_pixel
#     length_mm = length_px * mm_per_pixel
    
#     lengths.append(length_mm)
#     widths.append(width_mm)


# lengths = np.array(lengths)
# widths = np.array(widths)
# frames = np.arange(num_frames)


# # --- Plot Actual Length and Width (mm) vs Image Samples ---
# plt.figure(figsize=(10, 5))
# plt.plot(frames, lengths, label='Length (mm)', marker='o')
# plt.plot(frames, widths, label='Width (mm)', marker='s')
# plt.xlabel("Image Sample (Frame Number)")
# plt.ylabel("Length / Width (mm)")
# plt.title("ROI Length and Width Across Image Samples")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()



#-------------------------------------------------------------------------------------------

# import sys
# from os.path import abspath
# sys.path.extend([abspath(".")])

# import muDIC as dic
# import logging
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

# # For scale bar
# from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
# import matplotlib.font_manager as fm
# import numpy as np

# # Set logging level
# logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)

# # Path to folder containing images
# #path = r'./Sample12/'
# path = r'C:\Users\konat\Desktop\Thesis\Digtal Image Correlation\muDIC-master\muDIC-master\Examples\Sample12'

# # Load images
# images = dic.IO.image_stack_from_folder(path, file_type='.tif')

# # ... [Previous imports and setup remain unchanged] ...

# # Generate mesh (GUI selection occurs here)
# mesher = dic.Mesher(deg_e=3, deg_n=3, type="spline")
# mesh = mesher.mesh(images, GUI=True, n_ely=10, n_elx=8)

# # Instantiate and run DIC analysis
# settings = dic.DICInput(mesh, images)
# # ... [Your existing settings configuration] ...
# job = dic.DICAnalysis(settings)
# dic_results = job.run()

# # Post-processing
# fields = dic.post.viz.Fields(dic_results, upscale=10)
# viz = dic.Visualizer(fields, images=images)

# # Extract coordinates from LAST FRAME
# coords = fields.coords()
# x = coords[0, 0, :, :, -1]
# y = coords[0, 1, :, :, -1]

# # Extract displacement fields FIRST
# disp = fields.disp()  # Get displacement tensor
# disp_x = disp[0, 0, :, :, -1]  # Ux from last frame
# disp_y = disp[0, 1, :, :, -1]  # Uy from last frame
# disp_mag = np.sqrt(disp_x**2 + disp_y**2)  # Magnitude


# # Calculate ROI dimensions
# mm_per_pixel = 0.1  # Your calibration factor
# width_px = x.max() - x.min()
# height_px = y.max() - y.min()
# width_mm = width_px * mm_per_pixel
# height_mm = height_px * mm_per_pixel

# print(f"\nROI Dimensions:")
# print(f"Width:  {width_mm:.2f} mm ({width_px} px)")
# print(f"Height: {height_mm:.2f} mm ({height_px} px)")

# # Create figure with dimension display
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# # Add ROI dimensions to figure title
# fig.suptitle(f"Selected ROI: {width_mm:.1f} mm × {height_mm:.1f} mm", 
#             y=1.02, fontsize=14, fontweight='bold')

# # --- Plot Ux ---
# cf1 = ax1.contourf(x, y, disp_x, 50, cmap='coolwarm')
# # ... [Your existing Ux plot configuration] ...

# # --- Plot Uy --- 
# cf2 = ax2.contourf(x, y, disp_y, 50, cmap='viridis')
# # ... [Your existing Uy plot configuration] ...

# # --- Plot |U| ---
# cf3 = ax3.contourf(x, y, disp_mag, 50, cmap='plasma')
# # ... [Your existing |U| plot configuration] ...

# plt.tight_layout()
# plt.show()
