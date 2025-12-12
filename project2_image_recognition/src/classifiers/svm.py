from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import numpy as np, random, joblib


class SVMClassifier:
    """
    Linear SVM + StandardScaler
    모든 난수 시드를 하나의 `seed` 값으로 고정
    """

    def __init__(self,
                 C: float = 2.0,
                 max_iter: int = 20_000,
                 seed: int = 0):
        # ── 1. 전역 시드 고정 ──────────────────────────
        np.random.seed(seed)
        random.seed(seed)

        # ── 2. 파이프라인 생성 (LinearSVC에 random_state 전달) ──
        self.model = make_pipeline(
            StandardScaler(with_mean=False),
            LinearSVC(C=C,
                      max_iter=max_iter,
                      class_weight="balanced",
                      dual=False,
                      random_state=seed)
        )

    # scikit‑learn 호환 API
    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
