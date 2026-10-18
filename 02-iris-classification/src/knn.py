"""
k-nearest neighbors classifier and evaluation metrics implemented from scratch with NumPy.
"""

import numpy as np


class KNNClassifier:
    """Majority vote among the k closest training points (Euclidean distance).
    Ties are broken by the smallest total distance among the tied classes."""

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_, self.y_ = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(self.y_)
        return self

    def _neighbors(self, X):
        d = np.sqrt(((np.asarray(X, float)[:, None, :] - self.X_[None, :, :]) ** 2).sum(axis=2))
        idx = np.argsort(d, axis=1, kind="stable")[:, : self.k]
        return idx, np.take_along_axis(d, idx, axis=1)

    def predict_proba(self, X):
        idx, _ = self._neighbors(X)
        labels = self.y_[idx]
        return np.stack([(labels == c).mean(axis=1) for c in self.classes_], axis=1)

    def predict(self, X):
        idx, dist = self._neighbors(X)
        labels = self.y_[idx]
        out = []
        for lab, dst in zip(labels, dist):
            counts = {c: (lab == c).sum() for c in self.classes_}
            top = max(counts.values())
            tied = [c for c, n in counts.items() if n == top]
            out.append(min(tied, key=lambda c: dst[lab == c].sum()))
        return np.array(out)


class StandardScaler:
    def fit(self, X):
        self.mean_, self.std_ = X.mean(0), X.std(0)
        self.std_[self.std_ == 0] = 1
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_


def confusion_matrix(y_true, y_pred, labels):
    m = np.zeros((len(labels), len(labels)), int)
    pos = {l: i for i, l in enumerate(labels)}
    for t, p in zip(y_true, y_pred):
        m[pos[t], pos[p]] += 1
    return m


def classification_metrics(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels)
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    return {"accuracy": float(tp.sum() / cm.sum()), "precision": precision, "recall": recall,
            "f1": f1, "macro_f1": float(f1.mean()), "confusion": cm}


def stratified_kfold(y, k=5, seed=42):
    """Each fold keeps the class proportions of the full dataset."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        for i, chunk in enumerate(np.array_split(idx, k)):
            folds[i].extend(chunk)
    folds = [np.array(f) for f in folds]
    for i in range(k):
        yield np.concatenate([folds[j] for j in range(k) if j != i]), folds[i]


def cross_val_accuracy(make_model, X, y, k=5, repeats=10, scale=True):
    """Repeated stratified k-fold: mean and std of accuracy over k * repeats test folds."""
    scores = []
    for r in range(repeats):
        for tr, te in stratified_kfold(y, k, seed=r):
            Xtr, Xte = X[tr], X[te]
            if scale:
                sc = StandardScaler().fit(Xtr)
                Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            m = make_model().fit(Xtr, y[tr])
            scores.append((m.predict(Xte) == y[te]).mean())
    return float(np.mean(scores)), float(np.std(scores))
