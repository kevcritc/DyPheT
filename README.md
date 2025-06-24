# DyPheT
DyPheT is an Assembloid analysis tool, which is a Python-based application designed for automated tracking and analysis of rotating 3D cell structures (assembloids) in time-lapse microscopy videos. It aligns video frames using robust translation and rotation correction, applies polar image warping for rotational alignment, and performs per-frame cell detection based on colour segmentation.

##Features
-Robust frame alignment (translation + rotation) using OpenCV warpAffine and polar transformation.

-Template matching with optional ORB or uniform sampling for rotation estimation.

-Cell detection and tracking via HSV-based segmentation and ellipse fitting.

-Trajectory and movement analysis, including MSD and vector field visualization.

-Compatible with videos in .wmv format containing paired RGB and Overlay sequences.

- File names should be in pairs one for the brightfield overlay in the form XX_Overlay_Y.wmv and XX_RGB_Y.wmv


Requirements
Python 3.7+
OpenCV (cv2)
NumPy
Pandas
Matplotlib

It is recommended to use a virtual environment or Conda environment.

How it Works
The pipeline works in two main stages:

Preprocessing and Frame Registration

Each frame is centered by estimating the organoid’s centroid.

The rotation is estimated using template matching in polar-warped space.

The full affine matrix is calculated and stored per frame.

Cell Detection and Analysis

Cells are detected based on HSV thresholds and morphological filtering.

Per-frame detection results are saved and used for tracking.

Output includes cell trajectories, vector fields, and CSV data summaries.

## Developers:
**Kevin Critchley**
Christopher Akhunbay-Fudge
Heiko Wurdak
Ryan Mathew

## Polite request:
If you use this software for publication (official or otherwise) then, please acknowledge the developers.
