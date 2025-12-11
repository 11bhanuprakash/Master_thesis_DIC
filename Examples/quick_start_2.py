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
path = r'C:\Users\konat\Desktop\Thesis\Digtal Image Correlation\muDIC-master\Master_thesis\Examples\set2'
images = dic.IO.image_stack_from_folder(path, file_type='.tif')
#mesher = dic.Mesher(deg_e=3, deg_n=3, type="spline")
mesher = dic.Mesher(deg_e=3, deg_n=3, type="Q4")  # Using Q4 elements
mesh = mesher.mesh(images, GUI=True, n_ely=5, n_elx=4)

print("Xc1 =", mesh.Xc1)
print("Xc2 =", mesh.Xc2)
print("Yc1 =", mesh.Yc1)
print("Yc2 =", mesh.Yc2)


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

# extract displacement fields at final frame
disp = fields.disp()  
Ux = disp[0,0,:,:, -1]
Uy = disp[0,1,:,:, -1]

print("Ux =", Ux)
print("Uy =", Uy)

Umag = np.sqrt(Ux**2 + Uy**2)
x = fields.coords()[0,0,:,:, -1]
y = fields.coords()[0,1,:,:, -1]

print("x =",x)
print("y =",y)

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
plt.show()