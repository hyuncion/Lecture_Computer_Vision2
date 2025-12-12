
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

def plot_confusion(y_true, y_pred, classes, title, out_path):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    plt.figure(figsize=(8,8))
    sns.heatmap(cm, cmap="magma", xticklabels=classes, yticklabels=classes,
                square=True, cbar=False, annot=False)
    plt.title(f"{title}\nAcc={acc:.3f}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return acc
