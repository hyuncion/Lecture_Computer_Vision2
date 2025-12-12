# Project 1: Structure from Motion (SfM)

This project implements a classical **Structure from Motion (SfM)** pipeline in Python. The goal is to reconstruct a sparse 3D scene and estimate camera poses from multiple input images. Starting from an initial two-view reconstruction, the pipeline incrementally adds new views and outputs intermediate 3D point clouds for visualization.

---

## Overview

The SfM pipeline follows these main steps:

1. **Feature Extraction & Matching (SIFT)**
   - Extract SIFT keypoints/descriptors from two images
   - Match descriptors using BFMatcher (L2, cross-check)

2. **Essential Matrix Estimation (RANSAC)**
   - Estimate the Essential Matrix using robust RANSAC to filter outliers

3. **Pose Recovery**
   - Recover relative rotation and translation from the Essential Matrix
   - Keep only inlier correspondences via `recoverPose`

4. **Triangulation**
   - Triangulate 3D points from two-view correspondences

5. **Incremental Reconstruction (PnP + Triangulation)**
   - For each additional view, estimate camera pose using PnP RANSAC
   - Triangulate new points and accumulate into a global point cloud
   - Save intermediate `.ply` outputs for each step (2-view → 32-view)

---

## Dataset Preparation

1. Create a dataset directory: ./dataset_name/ 
2. Place input images in the directory using the following naming convention: 0000.jpg, 0016.jpg, 0017.jpg, ..., 0031.jpg
- Images are processed in groups of 16 views (starting from 0000 and 0016).
3. Add the camera intrinsic matrix file: K.txt
  
---

## Requirements

- Python 3  
- OpenCV (with contrib modules for SIFT)
- Open3D (for saving `.ply` files)
- NumPy  

Example installation:
```bash
pip install numpy open3d
```
---

## How to Run

Run the script:
```bash
python sfm_pipeline.py
```
The script will generate .ply files representing reconstructed 3D points and camera positions at each step of reconstruction:
```
points2_view.ply
points3_view.ply
...
points32_view.ply
```
All outputs are saved into the configured output folder.

---

## Additional Notes

### Step 6 — Camera Calibration
The intrinsic matrix (`K.txt`) should be obtained by calibrating your camera using a checkerboard pattern. You can compute it using OpenCV calibration functions such as:
- `findChessboardCorners`
- `calibrateCamera`

### Step 7 — Running with Your Own Dataset
To run the pipeline on your own images:
- Update dataset paths or folder names in the script
- Preserve the directory structure and file naming convention
- Provide a valid `K.txt` (intrinsic matrix)

---

## Key Learning Outcomes

- Robust correspondence handling using **RANSAC**
- Essential Matrix estimation and pose ambiguity resolution
- Two-view geometry and triangulation
- Incremental SfM using **PnP RANSAC**
- Practical multi-view 3D reconstruction workflow

This project is implemented for educational purposes and demonstrates a standard SfM pipeline using classical computer vision techniques.
