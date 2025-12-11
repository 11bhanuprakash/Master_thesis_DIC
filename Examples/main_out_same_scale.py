import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
from glob import glob

# -----------------------------------------------------
# 1. Folder containing CSVs
# -----------------------------------------------------
#ADJUST THE PATH TO YOUR FOLDER CONTAINING THE IMAGES same as Below

csv_folder = r"C:\Users\konat\Desktop\Thesis\Digtal Image Correlation\muDIC-master\Master_thesis\Examples\CSV_2 files"
csv_files = sorted(glob(os.path.join(csv_folder, "*.csv")))
if not csv_files:
    raise FileNotFoundError("No CSV files found in folder.")


from scipy.interpolate import griddata

def load_data(file_path):
    df = pd.read_csv(file_path)
    x = df["x"].values
    y = df["y"].values
    Ux = df["Ux"].values
    Uy = df["Uy"].values
    Umag = df["Umag"].values

    # Build a uniform grid automatically (interpolation)
    n_grid = 100  # adjust resolution (100x100 grid is smooth)
    xi = np.linspace(x.min(), x.max(), n_grid)
    yi = np.linspace(y.min(), y.max(), n_grid)
    X, Y = np.meshgrid(xi, yi)

    # Interpolate scattered data onto grid
    Ux_grid = griddata((x, y), Ux, (X, Y), method='cubic')
    Uy_grid = griddata((x, y), Uy, (X, Y), method='cubic')
    Umag_grid = griddata((x, y), Umag, (X, Y), method='cubic')

    return X, Y, Ux_grid, Uy_grid, Umag_grid


# -----------------------------------------------------
# 3. Initialize figure and axes
# -----------------------------------------------------
x, y, Ux, Uy, Umag = load_data(csv_files[0])

# --- FIXED COLOR SCALE: determine limits from first file ---
Ux_min, Ux_max = np.nanmin(Ux), np.nanmax(Ux)
Uy_min, Uy_max = np.nanmin(Uy), np.nanmax(Uy)
Umag_min, Umag_max = np.nanmin(Umag), np.nanmax(Umag)

fig = plt.figure(figsize=(14, 5))
ax_radio = plt.axes([0.02, 0.3, 0.15, 0.4])
ax1 = plt.axes([0.25, 0.1, 0.22, 0.8])
ax2 = plt.axes([0.50, 0.1, 0.22, 0.8])
ax3 = plt.axes([0.75, 0.1, 0.22, 0.8])

labels = [os.path.basename(f) for f in csv_files]
radio = RadioButtons(ax_radio, labels, active=0)

# -----------------------------------------------------
# 4. Initial contour plots (kept for re-use)
# -----------------------------------------------------
c1 = ax1.contourf(x, y, Ux, levels=50, cmap='coolwarm', vmin=Ux_min, vmax=Ux_max)
c2 = ax2.contourf(x, y, Uy, levels=50, cmap='viridis', vmin=Uy_min, vmax=Uy_max)
c3 = ax3.contourf(x, y, Umag, levels=50, cmap='plasma', vmin=Umag_min, vmax=Umag_max)


cb1 = fig.colorbar(c1, ax=ax1, fraction=0.046)
cb2 = fig.colorbar(c2, ax=ax2, fraction=0.046)
cb3 = fig.colorbar(c3, ax=ax3, fraction=0.046)

for ax, title in zip(
    [ax1, ax2, ax3],
    ["X-direction Displacement (Uₓ)", "Y-direction Displacement (Uᵧ)", "Total Displacement Magnitude |U|"]
):
    ax.set_title(title)
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.set_aspect("equal")

fig.suptitle(f"File: {os.path.basename(csv_files[0])}", fontsize=14)
# fig.tight_layout()
fig.subplots_adjust(left=0.22, right=0.98, top=0.9, bottom=0.1, wspace=0.35)


# -----------------------------------------------------
# 5. Update function: re-draw contours cleanly
# -----------------------------------------------------
def update_plot(file_path):
    global c1, c2, c3
    x, y, Ux, Uy, Umag = load_data(file_path)

    # clear only the image layers (keep layout)
    for ax in [ax1, ax2, ax3]:
        ax.clear()

    # redraw contours
    #turbo
    c1 = ax1.contourf(x, y, Ux, levels=50, cmap='coolwarm', vmin=Ux_min, vmax=Ux_max)
    c2 = ax2.contourf(x, y, Uy, levels=50, cmap='viridis', vmin=Uy_min, vmax=Uy_max)
    c3 = ax3.contourf(x, y, Umag, levels=50, cmap='plasma', vmin=Umag_min, vmax=Umag_max)


    # refresh colorbars
    cb1.update_normal(c1)
    cb2.update_normal(c2)
    cb3.update_normal(c3)

    # reapply labels and titles (lost after clear)
    for ax, title in zip(
        [ax1, ax2, ax3],
        ["X-direction Displacement (Uₓ)", "Y-direction Displacement (Uᵧ)", "Total Displacement Magnitude |U|"]
    ):
        ax.set_title(title)
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        ax.set_aspect("equal")

    fig.suptitle(f"File: {os.path.basename(file_path)}", fontsize=14)
    fig.canvas.draw_idle()

# -----------------------------------------------------
# 6. Radio button callback
# -----------------------------------------------------
def on_select(label):
    update_plot(os.path.join(csv_folder, label))

radio.on_clicked(on_select)
plt.show()


import pickle

def save_figure_session(filename="dic_viewer_session.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(fig, f)
    print(f"   Full interactive session saved to {filename}")

def load_figure_session(filename="dic_viewer_session.pkl"):
    with open(filename, "rb") as f:
        loaded_fig = pickle.load(f)
    plt.show(block=True)
    return loaded_fig
