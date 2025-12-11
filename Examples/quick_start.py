# # This allows for running the example when the repo has been cloned
# import sys
# from os.path import abspath
# sys.path.extend([abspath(".")])

# import muDIC as dic
# import logging
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt



# # Set the amount of info printed to terminal during analysis
# logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)

# # Path to folder containing images
# #path = r'./example_data/' # Use this formatting on Linux and Mac OS
# #path = r'./Sample12/'
# # path = r'C:\Users\konat\Desktop\Thesis\#Resources\sample13\sample13\data'  # Use this formatting on Windows
# path = r'C:\Users\konat\Desktop\Thesis\#Resources\no_grayscale_shifts\no_grayscale_shifts'
# # path = r'C:\Users\konat\Desktop\Thesis\#Resources\grayscale_shifts\grayscale_shifts'


# # Generate image instance containing all images found in the folder
# images = dic.IO.image_stack_from_folder(path, file_type='.tif')
# #images.set_filter(dic.filtering.lowpass_gaussian, sigma=1.)


# # Generate mesh
# mesher = dic.Mesher(deg_e=3, deg_n=3,type="q4")
# # mesher = dic.Mesher(deg_e=3, deg_n=3,type="spline")

# # If you want to see use a GUI, set GUI=True below
# #mesh = mesher.mesh(images,Xc1=316,Xc2=523,Yc1=209,Yc2=1055,n_ely=36,n_elx=9, GUI=False)
# # mesh = mesher.mesh(images, GUI=True, n_ely=5, n_elx=3)
# mesh = mesher.mesh(images,Xc1 = 100,Xc2 = 430,Yc1 = 50,Yc2 = 660,n_ely=6,n_elx=3, GUI=False)



# # Instantiate settings object and set some settings manually
# settings = dic.DICInput(mesh, images)
# settings.max_nr_im = 500
# settings.ref_update = [26]
# settings.maxit = 20
# settings.tol = 1.e-6
# settings.interpolation_order = 4
# # If you want to access the residual fields after the analysis, this should be set to True
# settings.store_internals = True

# # This setting defines the behaviour when convergence is not obtained
# settings.noconvergence = "ignore"

# # Instantiate job object
# job = dic.DICAnalysis(settings)

# # Running DIC analysis
# dic_results = job.run()

# print("dic_results :", dic_results)

# # Calculate field values
# fields = dic.post.viz.Fields(dic_results,upscale=10)

# # Show a field
# viz = dic.Visualizer(fields,images=images)

# # # Uncomment the line below to see the results
# viz.show(field="displacement", component = (1,0), frame=1) 
# # # viz.show(field="eng strain",component= (1,1), frame=1)

# # # # X-direction displacement (U_x)
# #viz.show(field="displacement", component=(1,1), frame=1)

# # # # Y-direction displacement (U_y)
# #viz.show(field="displacement", component=(0,1), frame=-1)

# # viz.show(field="displacement", component=(0,0), frame=1)

# # Example: X-direction displacement (U_x)
# # fig = viz.show(field="displacement", component=(1,0), frame=0)
# # plt.title("X-direction Displacement (U_x)", fontsize=14)

# # Example: Y-direction displacement (U_y)
# fig = viz.show(field="displacement", component=(0,1), frame=5)
# # plt.title("Y-direction Displacement (U_y)", fontsize=14)

# # Example: Magnitude displacement
# # fig = viz.show(field="displacement", component=(0,0), frame=1)
# # plt.title("Total Displacement Magnitude (|U|)", fontsize=14)
# # Get displacement field arrays








## ------------------------------------------------------------------------------------

# This allows for running the example when the repo has been cloned
import sys
from os.path import abspath
sys.path.extend([abspath(".")])

import muDIC as dic
import logging

# Set the amount of info printed to terminal during analysis
logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)

# Path to folder containing images
# path = r'./example_data/' # Use this formatting on Linux and Mac OS
#path = r'c:\path\to\example_data\\'  # Use this formatting on Windows
path = r'C:\Users\konat\Desktop\Thesis\Digtal Image Correlation\muDIC-master\Master_thesis\Examples\example_data'

# Generate image instance containing all images found in the folder
images = dic.IO.image_stack_from_folder(path, file_type='.tif')
#images.set_filter(dic.filtering.lowpass_gaussian, sigma=1.)


# Generate mesh
mesher = dic.Mesher(deg_e=3, deg_n=3,type="q4")

# If you want to see use a GUI, set GUI=True below
mesh = mesher.mesh(images,Xc1=316,Xc2=523,Yc1=209,Yc2=1055,n_ely=80,n_elx=20, GUI=False)

# Instantiate settings object and set some settings manually
settings = dic.DICInput(mesh, images)
settings.max_nr_im = 500
settings.ref_update = [15]
settings.maxit = 20
settings.tol = 1.e-6
settings.interpolation_order = 4
# If you want to access the residual fields after the analysis, this should be set to True
settings.store_internals = True

# This setting defines the behaviour when convergence is not obtained
settings.noconvergence = "ignore"

# Instantiate job object
job = dic.DICAnalysis(settings)

# Running DIC analysis
dic_results = job.run()

# Calculate field values
fields = dic.post.viz.Fields(dic_results,upscale=10)

# Show a field
viz = dic.Visualizer(fields,images=images)

# Uncomment the line below to see the results
viz.show(field="displacement", component = (1,1), frame=-1)
