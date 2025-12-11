import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button   # <-- use Button instead of RadioButtons
from glob import glob
from scipy.interpolate import griddata
import pickle

# -----------------------------------------------------
# 1. Folder containing CSVs
# -----------------------------------------------------

#ADJUST THE PATH TO YOUR FOLDER CONTAINING THE CSV files same as Below
csv_folder = r"C:\Users\konat\Desktop\Master_Thesis_DIC(Code)\Examples\CSV files"
csv_files = sorted(glob(os.path.join(csv_folder, "*.csv")))
if not csv_files:
    raise FileNotFoundError("No CSV files found in folder.")

# -----------------------------------------------------
# 2. Helper: Load and reshape data (with interpolation)
# -----------------------------------------------------
def load_data(file_path):
    df = pd.read_csv(file_path)
    x = df["x"].values
    y = df["y"].values
    Ux = df["Ux"].values
    Uy = df["Uy"].values
    Umag = df["Umag"].values

    n_grid = 100
    xi = np.linspace(x.min(), x.max(), n_grid)
    yi = np.linspace(y.min(), y.max(), n_grid)
    X, Y = np.meshgrid(xi, yi)

    Ux_grid = griddata((x, y), Ux, (X, Y), method='cubic')
    Uy_grid = griddata((x, y), Uy, (X, Y), method='cubic')
    Umag_grid = griddata((x, y), Umag, (X, Y), method='cubic')

    return X, Y, Ux_grid, Uy_grid, Umag_grid

# -----------------------------------------------------
# 3. Initialize figure and axes
# -----------------------------------------------------
x, y, Ux, Uy, Umag = load_data(csv_files[0])

fig = plt.figure(figsize=(14, 5))

# Axes for the three contour plots (moved up to leave room at bottom)
# ax1 = plt.axes([0.22, 0.22, 0.22, 0.72])
# ax2 = plt.axes([0.47, 0.22, 0.22, 0.72])
# ax3 = plt.axes([0.72, 0.22, 0.22, 0.72])

# MUCH BETTER LEFT UTILIZATION
ax1 = plt.axes([0.08, 0.22, 0.25, 0.72])
ax2 = plt.axes([0.37, 0.22, 0.25, 0.72])
ax3 = plt.axes([0.66, 0.22, 0.25, 0.72])


# -----------------------------------------------------
# 4. Initial contour plots
# -----------------------------------------------------
c1 = ax1.contourf(x, y, Ux, levels=50, cmap='coolwarm')
c2 = ax2.contourf(x, y, Uy, levels=50, cmap='viridis')
c3 = ax3.contourf(x, y, Umag, levels=50, cmap='plasma')

cb1 = fig.colorbar(c1, ax=ax1, fraction=0.046)
cb2 = fig.colorbar(c2, ax=ax2, fraction=0.046)
cb3 = fig.colorbar(c3, ax=ax3, fraction=0.046)

for ax, title in zip(
    [ax1, ax2, ax3],
    ["X-direction Displacement (Uₓ)",
     "Y-direction Displacement (Uᵧ)",
     "Total Displacement Magnitude |U|"]
):
    ax.set_title(title)
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.set_aspect("equal")

# No suptitle now
fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.18, wspace=0.35)

# -----------------------------------------------------
# 5. Update function: re-draw contours cleanly
# -----------------------------------------------------
def update_plot(file_path):
    global c1, c2, c3

    x, y, Ux, Uy, Umag = load_data(file_path)

    for ax in [ax1, ax2, ax3]:
        ax.clear()

    c1 = ax1.contourf(x, y, Ux, levels=50, cmap='coolwarm')
    c2 = ax2.contourf(x, y, Uy, levels=50, cmap='viridis')
    c3 = ax3.contourf(x, y, Umag, levels=50, cmap='plasma')

    cb1.update_normal(c1)
    cb2.update_normal(c2)
    cb3.update_normal(c3)

    for ax, title in zip(
        [ax1, ax2, ax3],
        ["X-direction Displacement (Uₓ)",
         "Y-direction Displacement (Uᵧ)",
         "Total Displacement Magnitude |U|"]
    ):
        ax.set_title(title)
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        ax.set_aspect("equal")

    fig.canvas.draw_idle()

# -----------------------------------------------------
# 6. Bottom row of buttons (one per file, centered)
# -----------------------------------------------------
labels = [os.path.basename(f) for f in csv_files]

buttons = []

n = len(labels)
bottom_btn = 0.05
height_btn = 0.06

# Max width available for the whole row
max_row_width = 0.9   # 90% of the figure width

# Pick button width so everything fits and can still be centered
btn_width = min(0.18, max_row_width / n)

# Total width occupied by all buttons
row_width = btn_width * n

# Start x so that the row is centered
start_x = 0.5 - row_width / 2

def make_callback(path):
    def _cb(event):
        update_plot(path)
    return _cb

for i, (label, path) in enumerate(zip(labels, csv_files)):
    ax_btn = plt.axes([
        start_x + i * btn_width,  # x position
        bottom_btn,               # y position
        btn_width * 0.95,         # width
        height_btn                # height
    ])
    btn = Button(ax_btn, label)
    btn.label.set_fontsize(14)   # larger font inside
    btn.on_clicked(make_callback(path))
    buttons.append(btn)


# -----------------------------------------------------
# 7. Optional: save / load figure session
# -----------------------------------------------------
def save_figure_session(filename="dic_viewer_session.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(fig, f)
    print(f"   Full interactive session saved to {filename}")

def load_figure_session(filename="dic_viewer_session.pkl"):
    with open(filename, "rb") as f:
        loaded_fig = pickle.load(f)
    plt.show(block=True)
    return loaded_fig

plt.show()

########################################################################################################################################
#uncommented working code below if you have scale info (converting from pixels to mm), if not use above code
########################################################################################################################################


# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Button   # <-- use Button instead of RadioButtons
# from glob import glob
# from scipy.interpolate import griddata
# import pickle

# #enter the scale in mm per pixel here

# SCALE_MM_PER_PX = 0.0885
# # -----------------------------------------------------
# # 1. Folder containing CSVs
# # -----------------------------------------------------  
# #ADJUST THE PATH TO YOUR FOLDER CONTAINING THE IMAGES same as Below

# csv_folder = r"C:\Users\konat\Desktop\Master_Thesis_DIC(Code)\Examples\CSV files"
# csv_files = sorted(glob(os.path.join(csv_folder, "*.csv")))
# if not csv_files:
#     raise FileNotFoundError("No CSV files found in folder.")

# # -----------------------------------------------------
# # 2. Helper: Load and reshape data (with interpolation)
# # -----------------------------------------------------
# def load_data(file_path):
#     df = pd.read_csv(file_path)

#     # raw values in pixels
#     x_px = df["x"].values
#     y_px = df["y"].values
#     Ux_px = df["Ux"].values
#     Uy_px = df["Uy"].values
#     Umag_px = df["Umag"].values

#     # convert to mm
#     s = SCALE_MM_PER_PX
#     x = x_px * s
#     y = y_px * s
#     Ux = Ux_px * s
#     Uy = Uy_px * s
#     Umag = Umag_px * s

#     # Build a uniform grid automatically (interpolation) in mm
#     n_grid = 100  # adjust resolution (100x100 grid is smooth)
#     xi = np.linspace(x.min(), x.max(), n_grid)
#     yi = np.linspace(y.min(), y.max(), n_grid)
#     X, Y = np.meshgrid(xi, yi)

#     # Interpolate scattered data onto grid (still in mm units)
#     Ux_grid = griddata((x, y), Ux, (X, Y), method='cubic')
#     Uy_grid = griddata((x, y), Uy, (X, Y), method='cubic')
#     Umag_grid = griddata((x, y), Umag, (X, Y), method='cubic')

#     return X, Y, Ux_grid, Uy_grid, Umag_grid

# # -----------------------------------------------------
# # 3. Initialize figure and axes
# # -----------------------------------------------------
# x, y, Ux, Uy, Umag = load_data(csv_files[0])

# fig = plt.figure(figsize=(14, 5))

# # MUCH BETTER LEFT UTILIZATION
# ax1 = plt.axes([0.08, 0.22, 0.25, 0.72])
# ax2 = plt.axes([0.37, 0.22, 0.25, 0.72])
# ax3 = plt.axes([0.66, 0.22, 0.25, 0.72])


# # -----------------------------------------------------
# # 4. Initial contour plots
# # -----------------------------------------------------
# c1 = ax1.contourf(x, y, Ux, levels=50, cmap='coolwarm')
# c2 = ax2.contourf(x, y, Uy, levels=50, cmap='viridis')
# c3 = ax3.contourf(x, y, Umag, levels=50, cmap='plasma')

# cb1 = fig.colorbar(c1, ax=ax1, fraction=0.046)
# cb2 = fig.colorbar(c2, ax=ax2, fraction=0.046)
# cb3 = fig.colorbar(c3, ax=ax3, fraction=0.046)

# for ax, title in zip(
#     [ax1, ax2, ax3],
#     ["X-direction Displacement (Uₓ)",
#      "Y-direction Displacement (Uᵧ)",
#      "Total Displacement Magnitude |U|"]
# ):
#     ax.set_title(title)
#     ax.set_xlabel("X (mm)")
#     ax.set_ylabel("Y (mm)")
#     ax.set_aspect("equal")

# # No suptitle now
# fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.18, wspace=0.35)

# # -----------------------------------------------------
# # 5. Update function: re-draw contours cleanly
# # -----------------------------------------------------
# def update_plot(file_path):
#     global c1, c2, c3

#     x, y, Ux, Uy, Umag = load_data(file_path)

#     for ax in [ax1, ax2, ax3]:
#         ax.clear()

#     c1 = ax1.contourf(x, y, Ux, levels=50, cmap='coolwarm')
#     c2 = ax2.contourf(x, y, Uy, levels=50, cmap='viridis')
#     c3 = ax3.contourf(x, y, Umag, levels=50, cmap='plasma')

#     cb1.update_normal(c1)
#     cb2.update_normal(c2)
#     cb3.update_normal(c3)

#     for ax, title in zip(
#         [ax1, ax2, ax3],
#         ["X-direction Displacement (Uₓ)",
#          "Y-direction Displacement (Uᵧ)",
#          "Total Displacement Magnitude |U|"]
#     ):
#         ax.set_title(title)
#         ax.set_xlabel("X (mm)")
#         ax.set_ylabel("Y (mm)")
#         ax.set_aspect("equal")

#     fig.canvas.draw_idle()

# # -----------------------------------------------------
# # 6. Bottom row of buttons (one per file, centered)
# # -----------------------------------------------------
# labels = [os.path.basename(f) for f in csv_files]

# buttons = []

# n = len(labels)
# bottom_btn = 0.05
# height_btn = 0.06

# # Max width available for the whole row
# max_row_width = 0.9   # 90% of the figure width

# # Pick button width so everything fits and can still be centered
# btn_width = min(0.18, max_row_width / n)

# # Total width occupied by all buttons
# row_width = btn_width * n

# # Start x so that the row is centered
# start_x = 0.5 - row_width / 2

# def make_callback(path):
#     def _cb(event):
#         update_plot(path)
#     return _cb

# for i, (label, path) in enumerate(zip(labels, csv_files)):
#     ax_btn = plt.axes([
#         start_x + i * btn_width,  # x position
#         bottom_btn,               # y position
#         btn_width * 0.95,         # width
#         height_btn                # height
#     ])
#     btn = Button(ax_btn, label)
#     btn.label.set_fontsize(14)   # larger font inside
#     btn.on_clicked(make_callback(path))
#     buttons.append(btn)


# # -----------------------------------------------------
# # 7. Optional: save / load figure session
# # -----------------------------------------------------
# def save_figure_session(filename="dic_viewer_session.pkl"):
#     with open(filename, "wb") as f:
#         pickle.dump(fig, f)
#     print(f"   Full interactive session saved to {filename}")

# def load_figure_session(filename="dic_viewer_session.pkl"):
#     with open(filename, "rb") as f:
#         loaded_fig = pickle.load(f)
#     plt.show(block=True)
#     return loaded_fig

# plt.show()

