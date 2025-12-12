
import argparse, os, joblib
import numpy as np
import json 
from pathlib import Path
from tqdm import tqdm
from data_loader import scan_dataset, split_dataset
from evaluation import plot_confusion
from features.bow import BOWExtractor
from features.spatial_pyramid import SpatialPyramidBOW
from features.vgg import VGGExtractor
from classifiers.svm import SVMClassifier
from classifiers.rf import RFClassifier
from classifiers.fc import FCClassifier

FEATURE_MAP = {
    "bow": BOWExtractor,
    "bow_sp": SpatialPyramidBOW,
    "vgg13": lambda: VGGExtractor("vgg13"),
    "vgg19": lambda: VGGExtractor("vgg19"),
}

CLASS_MAP = {
    "svm": SVMClassifier,
    "rf": RFClassifier,
    "fc": FCClassifier,
}

def cache(path, func):
    path = Path(path)
    if path.exists():
        return joblib.load(path)
    data = func()
    joblib.dump(data, path)
    return data

def main(args):
    samples, labels, idx_map = scan_dataset(args.root)
    train_x, test_x, train_y, test_y = split_dataset(samples, labels, test_size=0.3, random_state=0)
    os.makedirs("outputs", exist_ok=True)
    np.save("outputs/train_labels.npy", np.array(train_y))
    np.save("outputs/test_labels.npy",  np.array(test_y))
    classes = list(idx_map.keys())
    os.makedirs("outputs/features", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)
    

    feats = {}
    for feat_name in (FEATURE_MAP.keys() if args.all or args.feature is None
                      else [args.feature]):
        extractor = (FEATURE_MAP[feat_name]() if callable(FEATURE_MAP[feat_name])
                     else FEATURE_MAP[feat_name]())

        # ─────────────────────────────────────────────
        # 1) BoW·VGG 등의 conv layer → train set으로만 fit
        if hasattr(extractor, "fit"):
            extractor.fit(train_x)

        # 2) npy 캐시 경로
        tr_path = Path(f"outputs/features/{feat_name}_train.npy")
        te_path = Path(f"outputs/features/{feat_name}_test.npy")

        # 3) 이미 저장돼 있으면 불러오고, 없으면 계산 → np.save
        if tr_path.exists() and te_path.exists():
            train_feat = np.load(tr_path)
            test_feat  = np.load(te_path)
        else:
            train_feat = extractor.transform(train_x)
            test_feat  = extractor.transform(test_x)
            np.save(tr_path, train_feat)
            np.save(te_path,  test_feat)
        # ─────────────────────────────────────────────

        feats[feat_name] = (train_feat, test_feat)

    results = {}
    combos = [(f, c) for f in feats.keys()
                      for c in (CLASS_MAP.keys() if args.all or args.classifier is None else [args.classifier])]
    for feat_name, cls_name in combos:
        print(f"==> {feat_name} + {cls_name}")
        Cls = CLASS_MAP[cls_name]
        clf = Cls() if cls_name != "fc" else Cls(epochs=args.fc_epochs)
        tr_feat, te_feat = feats[feat_name]
        clf.fit(tr_feat, train_y)
        preds = clf.predict(te_feat)
        acc = plot_confusion(test_y, preds, classes,
                             f"{feat_name}+{cls_name}",
                             f"outputs/reports/confusion_{feat_name}_{cls_name}.png")
        results[f"{feat_name}_{cls_name}"] = acc
    with open("outputs/reports/summary.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Path to caltech20 dataset")
    parser.add_argument("--feature", choices=["bow", "bow_sp", "vgg13", "vgg19"])
    parser.add_argument("--classifier", choices=["svm", "rf", "fc"])
    parser.add_argument("--all", action="store_true", help="Run all 12 combinations")
    parser.add_argument("--fc_epochs", type=int, default=20)
    args = parser.parse_args()
    main(args)
