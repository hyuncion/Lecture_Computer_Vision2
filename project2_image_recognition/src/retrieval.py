import numpy as np
from sklearn.metrics import average_precision_score
import argparse, json, os
from pathlib import Path

def l2(a, b):
    return np.linalg.norm(a - b, axis=1)

def evaluate_retrieval(features, labels, topk=5):
    N = features.shape[0]
    topk_acc, mAP = [], []
    for i in range(N):
        dists = l2(features[i], features)
        idxs  = np.argsort(dists)[1:]          # exclude itself
        top   = labels[idxs[:topk]]
        topk_acc.append((top == labels[i]).mean())

        # binary - same-class vs others
        rel = (labels == labels[i]).astype(int)
        mAP.append(average_precision_score(rel, -dists))
    return float(np.mean(topk_acc)), float(np.mean(mAP))

# ────────────────────────────────────────────────────────────────
def evaluate_file(fpath, lpath, topk=5):
    x = np.load(fpath)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)  # 추가
    y = np.load(lpath)
    return evaluate_retrieval(x, y, topk)
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--all", action="store_true",
                     help="evaluate every *.npy under outputs/features/")
    grp.add_argument("--feature", help="single feature .npy file")

    parser.add_argument("--labels", help="label .npy (for single run)")
    parser.add_argument("--topk", type=int, default=5, help="Top-k")
    args = parser.parse_args()

    results = {}

    if args.all:
        feat_dir = Path("outputs/features")
        rep_dir  = Path("outputs/reports"); rep_dir.mkdir(parents=True, exist_ok=True)

        for fp in feat_dir.glob("*.npy"):
            if fp.name.endswith("_train.npy"):
                label_fp = Path("outputs/train_labels.npy")
            elif fp.name.endswith("_test.npy"):
                label_fp = Path("outputs/test_labels.npy")
            else:
                continue

            if not label_fp.exists():
                print(f"[Skip] {fp.name}: {label_fp} not found")
                continue

            top5, mAP = evaluate_file(fp, label_fp, topk=args.topk)
            results[fp.name] = {"top5": top5, "mAP": mAP}
            print(f"{fp.name:25s}  Top-{args.topk}: {top5:.3f}   mAP: {mAP:.3f}")

        # 텍스트 & JSON 두 형식으로 저장
        txt_path = rep_dir / "retrieval_results.txt"
        with open(txt_path, "w") as f:
            for k, v in sorted(results.items()):
                f.write(f"{k}\tTop{args.topk}:{v['top5']:.4f}\tmAP:{v['mAP']:.4f}\n")
        json_path = rep_dir / "retrieval_results.json"
        json_path.write_text(json.dumps(results, indent=2))

        print(f"\nSaved summary to {txt_path} and {json_path}")

    else:
        if args.labels is None:
            parser.error("--labels is required when --feature is used")
        top5, mAP = evaluate_file(Path(args.feature), Path(args.labels), args.topk)
        print(f"Top-{args.topk} acc: {top5:.3f},  mAP: {mAP:.3f}")
