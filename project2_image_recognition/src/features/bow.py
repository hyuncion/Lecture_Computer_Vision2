import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from tqdm import tqdm


class BOWExtractor:
    """
    Dense-SIFT  →  RootSIFT  →  PCA  →  MiniBatch-KMeans  →  BoW histogram
    최종 히스토그램은 L1 정규화 → sqrt(Hellinger) → L2 정규화.
    """

    def __init__(self,
                 num_words: int = 800,
                 pca_dim: int = 64,
                 dense_step: int = 8,
                 batch_size: int = 2048):
        self.num_words = num_words
        self.pca_dim = pca_dim
        self.dense_step = dense_step

        self.sift = cv2.SIFT_create()
        self.pca = PCA(n_components=pca_dim, whiten=True, random_state=0)
        self.kmeans = MiniBatchKMeans(n_clusters=num_words,
                                      batch_size=batch_size,
                                      random_state=0)

    # ────────────────────────── internal ──────────────────────────
    def _dense_sift(self, gray):
        """Dense-SIFT keypoints at a fixed step."""
        h, w = gray.shape
        kps = [cv2.KeyPoint(x, y, self.dense_step)
               for y in range(0, h, self.dense_step)
               for x in range(0, w, self.dense_step)]
        _, desc = self.sift.compute(gray, kps)
        return desc  # (N,128) or None

    @staticmethod
    def _rootsift(desc):
        """L1-norm → sqrt : RootSIFT."""
        desc /= (desc.sum(axis=1, keepdims=True) + 1e-7)
        return np.sqrt(desc)

    @staticmethod
    def _histogram(words, vocab_size):
        hist, _ = np.histogram(words, bins=np.arange(vocab_size + 1))
        if hist.sum() > 0:
            hist = hist / hist.sum()           # L1
            hist = np.sqrt(hist)               # Hellinger
            hist = hist / (np.linalg.norm(hist) + 1e-9)  # L2
        return hist.astype(np.float32)

    # ─────────────────────────── public ───────────────────────────
    def fit(self, img_paths, verbose=True):
        """PCA + KMeans 학습 (train set 전용)."""
        desc_bank = []
        for p in tqdm(img_paths, disable=not verbose):
            gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            d = self._dense_sift(gray)
            if d is not None:
                desc_bank.append(self._rootsift(d))
        desc_bank = np.vstack(desc_bank).astype(np.float32)

        reduced = self.pca.fit_transform(desc_bank)
        self.kmeans.fit(reduced)

    def transform(self, img_paths, verbose=False):
        feats = []
        iterator = tqdm(img_paths, disable=not verbose)
        for p in iterator:
            gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            d = self._dense_sift(gray)
            if d is None:
                feats.append(np.zeros(self.num_words, dtype=np.float32))
                continue

            d = self._rootsift(d)
            d = self.pca.transform(d)                # (N,pca_dim)
            words = self.kmeans.predict(d)           # (N,)
            feats.append(self._histogram(words, self.num_words))
        return np.vstack(feats)                       # (B,num_words)
