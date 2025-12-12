# Project 2: Caltech-20 Image Recognition & Retrieval

This repository contains our implementation for **Programming Assignment 2** of a graduate-level Computer Vision course.  
The project focuses on comparing **handcrafted features** and **deep learning–based representations** for both **image classification** and **image retrieval** tasks on the **Caltech-20** dataset.

Unlike a black-box pipeline, this implementation is designed to clearly expose each stage—feature extraction, model training, and evaluation—to facilitate analysis and comparison.

---

## 🔍 What This Project Covers

### Image Representations
- **Bag-of-Visual-Words (BoW)** using SIFT
- **BoW + Spatial Pyramid** (weak geometric encoding)
- **VGG-13** deep features (pre-trained, frozen)
- **VGG-19** deep features (pre-trained, frozen)

### Classifiers
- **Linear SVM**
- **Random Forest**
- **2-Layer Fully Connected Network** (PyTorch)

### Evaluation Tasks
- **Multi-class image classification**
- **Image retrieval** (Top-k accuracy, mAP)
- **Confusion matrix analysis**

---

## 1. Installation

Install required dependencies using:

```bash
pip install -r requirements.txt
```

PyTorch is required only for the FC classifier.
CNN backbones use pre-trained ImageNet weights and are not fine-tuned.

---

## 2. Dataset Setup

Prepare the **Caltech-20** dataset with the following directory structure:
```text
caltech20/
├── ant
│ ├── image_0001.jpg
│ ├── image_0002.jpg
│ └── ...
├── beaver
├── camera
└── ...
```

Each sub-folder corresponds to one object category, resulting in a total of **20 classes**.

---

## 3. Running Experiments

### Run all feature–classifier combinations

```bash
python src/main.py --root /path/to/caltech20 --all
```
Example:
```bash
python src/main.py --root /home/user/data/caltech20 --all
```

This command trains and evaluates **all 4 × 3 feature–classifier combinations**.

Generated outputs under `outputs/`:
- `features/*.npy` — cached feature descriptors  
- `models/` — trained classifiers  
- `reports/confusion_*.png` — confusion matrices  
- `reports/retrieval_*.json` — image retrieval metrics  

---

### Run a single experiment

```bash
python src/main.py --root /path/to/caltech20 --feature FEATURE --classifier CLASSIFIER
```
Example:
```bash
python src/main.py --root /home/user/data/caltech20 --feature bow_sp --classifier svm
```

#### Available options

**Feature**

| Option   | Description               |
|----------|---------------------------|
| `bow`    | Bag-of-Words (SIFT)       |
| `bow_sp` | BoW + Spatial Pyramid     |
| `vgg13`  | VGG-13 deep features      |
| `vgg19`  | VGG-19 deep features      |

**Classifier**

| Option | Description              |
|--------|--------------------------|
| `svm`  | Linear SVM               |
| `rf`   | Random Forest            |
| `fc`   | 2-Layer FC (PyTorch)     |

---

## 4. Image Retrieval Experiments

### Run all retrieval experiments

```bash
python src/retrieval.py --all
```

### Run a single retrieval experiment

```bash
python src/retrieval.py --feature FEATURE.npy --labels LABELS.npy
```

Example:
```bash
python src/retrieval.py \
  --feature outputs/features/vgg19_test.npy \
  --labels outputs/test_labels.npy
```

### Supported feature files

- `bow_train.npy`, `bow_test.npy`
- `bow_sp_train.npy`, `bow_sp_test.npy`
- `vgg13_train.npy`, `vgg13_test.npy`
- `vgg19_train.npy`, `vgg19_test.npy`

---

## 📊 Outputs & Analysis

The project produces:
- Classification accuracies across **12 feature–classifier combinations**
- **Confusion matrices** for qualitative error analysis
- **Image retrieval metrics** (Top-k accuracy, mAP)
- A direct comparison between **handcrafted features** and **deep CNN embeddings**

---

## 🎯 Key Learning Outcomes

- Performance gap between handcrafted features and deep CNN representations
- Effectiveness of weak geometric encoding via **Spatial Pyramid**
- Influence of classifier choice on fixed feature representations
- Practical evaluation of **image retrieval vs. classification**
- End-to-end experimental design in computer vision

---

## 📌 Notes

- This project is implemented for **educational purposes**.
- CNN backbones are **not fine-tuned** to ensure fair comparison across methods.
- Results may vary depending on random seed and hardware configuration.

This repository represents a structured exploration of classical and deep-learning-based image recognition methods within a unified experimental framework.
