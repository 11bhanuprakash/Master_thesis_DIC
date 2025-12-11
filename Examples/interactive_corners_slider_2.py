import sys
from os.path import abspath
sys.path.extend([abspath(".")])

import muDIC as dic
import logging
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# Set logging level
logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)

# --- load images and build mesh exactly as before ---

#ADJUST THE PATH TO YOUR FOLDER CONTAINING THE IMAGES same as Below

path = r'C:\Users\konat\Desktop\Thesis\Digtal Image Correlation\muDIC-master\Master_thesis\Examples\set2\aligned\circle_hist_matched'
images = dic.IO.image_stack_from_folder(path, file_type='.tif')
mesher = dic.Mesher(deg_e=3, deg_n=3, type="spline")
# mesher = dic.Mesher(deg_e=3, deg_n=3, type="Q4")
mesh = mesher.mesh(images, GUI=True, n_ely=5, n_elx=4)

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

# post-processing
fields = dic.post.viz.Fields(dic_results, upscale=10)

# Extract X and Y coordinate arrays over time
x_all = fields.coords()[0, 0, :, :, :]  # shape: [rows, cols, frames]
y_all = fields.coords()[0, 1, :, :, :]  # same

frames = x_all.shape[2]

# Prepare corner coordinate arrays
corner_labels = ['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right']

def get_corners(frame):
    return [
        (x_all[0, -1, frame], y_all[0, -1, frame]),  # Top-Left  
        (x_all[-1, -1, frame], y_all[-1, -1, frame]),  
        (x_all[0, 0, frame], y_all[0, 0, frame]),      
        (x_all[-1, 0, frame], y_all[-1, 0, frame])   
    ]

# Load reference image (first frame)
ref_image = images[0]  # shape: (height, width)

# Set up the plot
# fig, ax = plt.subplots(figsize=(8, 6))
# plt.subplots_adjust(bottom=0.2)  # Make space for slider

fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)


# Show reference image as background
ax.imshow(ref_image, cmap='gray', origin='upper')

# Initial frame
init_frame = 0
corners = get_corners(init_frame)
scat = ax.scatter(
    [c[0] for c in corners],
    [c[1] for c in corners],
    s=100,
    c=['r', 'g', 'b', 'm'],
    label=corner_labels
)

# Annotations
annotations = []
for i, label in enumerate(corner_labels):
    ann = ax.annotate(
        label,
        (corners[i][0], corners[i][1]),
        textcoords="offset points",
        xytext=(5, 5),
        ha='left',
        color=scat.get_facecolors()[i]
    )
    annotations.append(ann)

ax.set_title(f"ROI Corners - Frame {init_frame+1}/{frames}")
ax.set_xlabel("X position (px)")
ax.set_ylabel("Y position (px)")
ax.invert_yaxis()
ax.grid(True)

# Slider
axframe = plt.axes([0.15, 0.08, 0.7, 0.05])
frame_slider = Slider(axframe, 'Frame', 1, frames, valinit=init_frame+1, valstep=1)

# Update function
def update(val):
    frame = int(frame_slider.val) - 1
    new_corners = get_corners(frame)
    scat.set_offsets(np.array(new_corners))
    ax.set_title(f"ROI Corners - Frame {frame+1}/{frames}")

    for i, ann in enumerate(annotations):
        ann.xy = (new_corners[i][0], new_corners[i][1])
        ann.set_position((new_corners[i][0] + 5, new_corners[i][1] + 5))

    fig.canvas.draw_idle()

frame_slider.on_changed(update)

# plt.tight_layout()
plt.show()
