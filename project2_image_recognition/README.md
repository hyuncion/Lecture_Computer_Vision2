
# Caltech‑20 Image Recognition – Programming Assignment 2 (Starter Code)

This repository follows the specification given in the assignment PDF.  
It provides:

* **4 feature extractors** – Bag‑of‑Words (BoW), BoW + Spatial Pyramid, VGG‑13, VGG‑19  
* **3 classifiers** – Linear SVM, Random Forest, 2‑layer Fully‑Connected (PyTorch)  
* Evaluation utilities for **image retrieval** and **classification + confusion matrices**.

---

## 1. Installation

```bash
pip install -r requirements.txt
```

## 2. Dataset

Place the **Caltech‑20** folder (20 sub‑folders named *ant, beaver, …, saxophone*) anywhere on disk:

```
caltech20/
 ├── ant
 │   ├── image_0001.jpg
 │   └── ...
 ├── beaver
 └── ...
```

## 3. Quick start

### Train & test every (feature, classifier) combination

```bash
python src/main.py --root /path/to/caltech20  --all

ex.
python src/main.py --root /home/cvlab-dgx/project/seungjun/cv/caltech20/caltech20  --all
```

The script produces under `outputs/`

* `features/*.npy` ‑ cached image descriptors  
* `models/*` ‑ trained classifiers  
* `reports/confusion_*png` – 12 confusion matrices  
* `reports/retrieval_*json` – retrieval metrics (top‑k & mAP)

### Single experiment

```bash
python src/main.py --root /path/to/caltech20 --feature model_name1 --classifier model_name2

ex.
python src/main.py --root /home/cvlab-dgx/project/seungjun/cv/caltech20/caltech20 --feature bow_sp --classifier svm
```

Available options

| `--feature`   | `bow`, `bow_sp`, `vgg13`, `vgg19` |
|---------------|-----------------------------------|
| `--classifier`| `svm`, `rf`, `fc`                 |


### Image retrieval all experiment

```bash
python src/retrieval.py --all
```

### Image retrieval single experiments

```bash
python src/retrieval.py --feature /path/to/fature.npy --labels /path/to/labels.npy

ex.
python src/retrieval.py --feature /home/cvlab-dgx/project/seungjun/cv/outputs/features/vgg19_train.npy --labels /home/cvlab-dgx/project/seungjun/cv/outputs/train_labels.npy
```
Available options

| `--features`   | `bow_sp_test.npy`, `bow_sp_train.npy`, `bow_test.npy`, `bow_train.npy`, `vgg13_test.npy`, `vgg13_train.npy`, `vgg19_test.npy`, `vgg19_train.npy` |
| `--labels`   | `test_labels.npy`, `train_labels.npy` |

It collects:

* **4 retrieval results** (qualitative & quantitative)  
* **12 confusion matrices** + **mean accuracies**  
* Comparison table of 4×3 combinations