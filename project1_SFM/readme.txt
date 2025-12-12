This README provides instructions on how to execute the Structure from Motion (SfM) pipeline implemented in Python.
The code reconstructs a 3D scene and estimates camera poses from multiple input images.

How to Execute the Code

1. Prepare the dataset:
- Place your input images in the `./{dataset_name}/` folder.
- Ensure the images are named as 0000.jpg, 0016.jpg, ..., up to 0031.jpg (16 images per group).
- Include the intrinsic matrix file as `K.txt` in the same folder.

2. Install required libraries:
- OpenCV (with contrib modules for SIFT)
- Open3D (only for save .ply files)
- NumPy

3. Run the script:
Simply execute the Python script.
The script will generate `.ply` files representing 3D point clouds and camera positions for each step of reconstruction (from 2-view to 32-view) in the output folder.

Additional Notes

Step 6 - Camera Calibration:
The intrinsic matrix (K) was obtained using the Camera Calibration Toolbox. You need to perform your own calibration by capturing images of a checkerboard pattern and processing them using OpenCV’s calibration tools.

Step 7 - Running with Your Own Dataset:
To run the pipeline with your own dataset, simply modify the file paths or folder names in the script to match your image locations and the intrinsic matrix file. Ensure that the file naming convention and directory structure are preserved.
