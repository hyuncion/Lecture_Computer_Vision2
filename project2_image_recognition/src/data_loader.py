
import os, glob, random
from typing import List, Tuple
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')

def scan_dataset(root: str):
    """Return lists of (path,label) sorted by label."""
    root = Path(root)
    samples, labels, label_to_idx = [], [], {}
    for idx, class_dir in enumerate(sorted(d for d in root.iterdir() if d.is_dir())):
        label_to_idx[class_dir.name] = idx
        for fp in class_dir.glob('*'):
            if fp.suffix.lower() in IMG_EXT:
                samples.append(str(fp))
                labels.append(idx)
    return samples, labels, label_to_idx

def split_dataset(samples, labels, test_size=0.3, random_state=0):
    return train_test_split(samples, labels,
                            test_size=test_size,
                            stratify=labels,
                            random_state=random_state)
