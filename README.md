Overview

This project implements an automated Digital Image Correlation (DIC) workflow for monitoring and quantifying shrinkage during microwave drying. Built on the open-source μDIC framework in Python, the system incorporates fiducial marker detection, automatic ROI extraction, histogram equalization, full-field displacement computation, and mesh-based dimensional measurement. Internal deformation is analyzed through point-tracking within the ROI, and a dynamic slider tool enables visualisation of marker motion across image sequences. The workflow is fully automated using JSON configuration files and outputs structured CSV data for efficient post-processing.

This thesis work was implemented by leveraging μDIC without modifying its fundamental architecture; instead, additional processing modules and tools were developed on top of it.
🔗 μDIC documentation: https://mudic.readthedocs.io/en/latest/

Implemented Modules (Examples Folder)
1. main_test.py

Reads all required parameters from JSON configuration files, executes the automated DIC workflow, and saves measurement results into CSV files.

2. main_out.py

Performs post-processing by reading CSV outputs and generating visualisation or summary results.

3. det_cal_markers.py

Detects circular fiducial markers using contour-based and shape-based analysis with adaptive thresholding.

4. allign_det_markers.py

Aligns all images in the sequence using the detected markers to ensure consistent spatial reference throughout the stack.

5. interactive-corners-slider.py

Introduces an interactive slider tool to visualise movement of corner markers across the image sequence.

6. interactive-corners-slider_2.py

Slider-based tool for visualising corner point displacement on the reference image.

7. multi_internal_pt.py

Computes and plots internal geometric measures (D11, D21, internal diagonals, diameters) to quantify internal deformation.

8. auto_roi.py

Automatically extracts the circular Region of Interest (ROI) using image-driven selection methods.

9. verification_angle.py

Computes and visualises pre- and post-calibration angle differences in a single plot.

10. verification_disp.py

Calculates the sum of marker displacements before and after processing and displays them in a combined plot.

11. pixel_to_mm.py

Converts displacement values from pixels to millimetres using real-world scale calibration.

12. temp.py

Main script for executing the full pipeline via command-line interface.

13. temp-1.py

Reads JSON files that exclude parameters such as n_ely and n_elx.

Documentation

If you have any doubts regarding usage or implementation details, please refer to the Documentation folder.
