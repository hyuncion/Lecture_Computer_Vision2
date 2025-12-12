# 🎓 Computer Vision Course Projects

This repository contains two projects completed as part of a **graduate-level Computer Vision course**.  
The projects cover both **3D geometric vision** and **2D image understanding**, combining classical computer vision techniques with modern learning-based approaches.

- **Project 1:** Structure from Motion (SfM) — multi-view 3D reconstruction  
- **Project 2:** Image Recognition & Retrieval — feature representation and evaluation on Caltech-20  

Each project is implemented with a strong focus on understanding the underlying theory through hands-on experimentation.

---

## 📁 Project Overview

### 📌 Project 1: Structure from Motion (SfM)

This project implements a classical **Structure from Motion (SfM)** pipeline that reconstructs a sparse 3D scene and estimates camera poses from multiple images.

**Key components**
- SIFT feature extraction and matching
- Essential Matrix estimation with RANSAC
- Camera pose recovery and ambiguity resolution
- Two-view triangulation
- Incremental reconstruction using PnP RANSAC
- Progressive 3D point cloud generation (2-view → 32-view)

**Outputs**
- `.ply` files containing reconstructed 3D points and camera positions at each step

**Core focus**
- Multi-view geometry
- Epipolar constraints
- Robust estimation and incremental SfM

---

### 📌 Project 2: Image Recognition & Retrieval (Caltech-20)

This project studies how different image representations and classifiers affect **classification** and **retrieval** performance on the Caltech-20 dataset.

**Image representations**
- Bag-of-Words (BoW) with SIFT
- BoW + Spatial Pyramid
- VGG-13 deep features (pre-trained)
- VGG-19 deep features (pre-trained)

**Classifiers**
- Linear SVM
- Random Forest
- 2-layer Fully Connected Network (PyTorch)

**Evaluation tasks**
- Multi-class image classification
- Image retrieval (Top-k accuracy, mAP)
- Confusion matrix analysis

**Core focus**
- Comparison between handcrafted and deep features
- Effect of weak geometric encoding
- Influence of classifier choice

---

## 🛠 Tools & Libraries

- Python 3  
- OpenCV (with contrib modules for SIFT)
- Open3D (for 3D point cloud export)
- NumPy
- scikit-learn
- PyTorch (FC classifier only)

---

## 📊 Key Outcomes

- Practical understanding of **2D–3D geometry** through SfM
- Clear performance gap between handcrafted features and deep CNN embeddings
- Importance of geometric constraints in 3D reconstruction
- Insight into evaluation differences between classification and retrieval
- End-to-end experimental design for computer vision systems

---

## 📌 Notes

- All implementations are for **educational purposes**.
- Camera intrinsics for SfM are obtained via checkerboard-based calibration.
- CNN backbones are kept **frozen** to ensure fair feature comparison.
- Results may vary depending on dataset, random seed, and hardware.

---

## ✨ Takeaway

This repository demonstrates how **classical computer vision principles** (geometry, feature matching, optimization) and **modern learning-based representations** can be applied to solve complementary vision problems.  
Together, the two projects provide a coherent view of how images can be used to understand both **3D structure** and **semantic content**.
