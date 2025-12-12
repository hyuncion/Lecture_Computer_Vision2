from sklearn.ensemble import RandomForestClassifier
import joblib


class RFClassifier:
    """
    Random Forest (사전 튜닝 값)
    - 1000 trees, sqrt features, balanced_subsample
    - n_jobs = -1 : 모든 CPU 사용
    """

    def __init__(self,
                 n_estimators: int = 1000,
                 max_depth: int = 20,
                 min_samples_leaf: int = 1,
                 n_jobs: int = -1,
                 random_state: int = 0):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            bootstrap=True,
            class_weight="balanced_subsample",
            n_jobs=n_jobs,
            random_state=random_state
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
