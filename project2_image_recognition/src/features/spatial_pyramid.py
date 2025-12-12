import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from tqdm import tqdm


class SpatialPyramidBOW:
    """
    BoW + Spatial Pyramid (levels 0,1,2)  with weighted concatenation.
      - Level 0 : 1×1   weight = 1     (coarsest, 가장 중요)
      - Level 1 : 2×2   weight = 1/2
      - Level 2 : 4×4   weight = 1/4
    최종 벡터는 Root-Hellinger(L1→sqrt→L2) 로 정규화.
    """

    def __init__(self,
                 num_words: int = 800,
                 pca_dim: int = 64,
                 dense_step: int = 8,
                 levels: int = 2,
                 batch_size: int = 2048):
        self.num_words = num_words
        self.pca_dim = pca_dim
        self.dense_step = dense_step
        self.levels = levels

        self.sift = cv2.SIFT_create()
        self.pca = PCA(n_components=pca_dim, whiten=True, random_state=0)
        self.kmeans = MiniBatchKMeans(n_clusters=num_words,
                                      batch_size=batch_size,
                                      random_state=0)

    # ────────────────────────── helpers ──────────────────────────
    def _dense_sift(self, gray):
        h, w = gray.shape
        kps = [cv2.KeyPoint(x, y, self.dense_step)
               for y in range(0, h, self.dense_step)
               for x in range(0, w, self.dense_step)]
        _, d = self.sift.compute(gray, kps)
        coords = None
        if d is not None:
            coords = np.array([[kp.pt[0] / w, kp.pt[1] / h] for kp in kps],
                              dtype=np.float32)  # (N,2) in [0,1]
        return d, coords

    @staticmethod
    def _rootsift(desc):
        desc /= (desc.sum(axis=1, keepdims=True) + 1e-7)
        return np.sqrt(desc)

    def _level_hist(self, words, coords, level):
        """level-l (cells×cells) histogram, weight = 1/2^{L-l}."""
        cells = 2 ** level
        hist = np.zeros((cells, cells, self.num_words), dtype=np.float32)
        for w, (x, y) in zip(words, coords):
            ix, iy = int(x * cells), int(y * cells)
            # clamp for edge cases
            ix = min(ix, cells - 1)
            iy = min(iy, cells - 1)
            hist[iy, ix, w] += 1
        weight = 1.0 / (2 ** (self.levels - level))
        return (hist.reshape(-1) * weight)

    @staticmethod
    def _final_norm(vec):
        if vec.sum() > 0:
            vec /= vec.sum()
            vec = np.sqrt(vec)
            vec /= (np.linalg.norm(vec) + 1e-9)
        return vec.astype(np.float32)

    # ─────────────────────────── train ───────────────────────────
    def fit(self, img_paths, verbose=True):
        bank = []
        for p in tqdm(img_paths, disable=not verbose):
            gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            d, _ = self._dense_sift(gray)
            if d is not None:
                bank.append(self._rootsift(d))
        bank = np.vstack(bank).astype(np.float32)

        reduced = self.pca.fit_transform(bank)
        self.kmeans.fit(reduced)

    # ───────────────────────── transform ─────────────────────────
    def transform(self, img_paths, verbose=False):
        feats = []
        iterator = tqdm(img_paths, disable=not verbose)
        for p in iterator:
            gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            d, coords = self._dense_sift(gray)
            if d is None:
                dim = self.num_words * sum((2 ** l) ** 2
                                           for l in range(self.levels + 1))
                feats.append(np.zeros(dim, dtype=np.float32))
                continue

            d = self._rootsift(d)
            d = self.pca.transform(d)
            words = self.kmeans.predict(d)

            pyramid = []
            for l in range(self.levels + 1):
                pyramid.append(self._level_hist(words, coords, l))
            vec = np.concatenate(pyramid)
            feats.append(self._final_norm(vec))
        return np.vstack(feats)   # shape: (B, D)
