"""
Binary logistic regression with L2 regularization, written from scratch in NumPy.
Loss:  mean log-loss + (lam / 2) * ||w||^2      Optimizer: mini-batch Adam.
"""

import numpy as np


def sigmoid(z):
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1 + ez)
    return out


def log_loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class LogisticRegression:
    def __init__(self, lam=1e-5, lr=0.01, epochs=12, batch=256, seed=0):
        self.lam, self.lr, self.epochs, self.batch, self.seed = lam, lr, epochs, batch, seed

    def loss_and_grad(self, X, y, w, b):
        p = sigmoid(X @ w + b)
        err = (p - y) / len(y)
        loss = log_loss(y, p) + 0.5 * self.lam * float(w @ w)
        return loss, X.T @ err + self.lam * w, float(err.sum())

    def fit(self, X, y, X_val=None, y_val=None):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        w, b = np.zeros(d), 0.0
        m, v, mb, vb, t = np.zeros(d), np.zeros(d), 0.0, 0.0, 0
        b1, b2, eps = 0.9, 0.999, 1e-8
        self.history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}
        for ep in range(1, self.epochs + 1):
            for idx in np.array_split(rng.permutation(n), max(1, n // self.batch)):
                _, gw, gb = self.loss_and_grad(X[idx], y[idx], w, b)
                t += 1
                m = b1 * m + (1 - b1) * gw; v = b2 * v + (1 - b2) * gw * gw
                mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb * gb
                w -= self.lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
                b -= self.lr * (mb / (1 - b1 ** t)) / (np.sqrt(vb / (1 - b2 ** t)) + eps)
            self.w, self.b = w, b
            self.history["epoch"].append(ep)
            self.history["train_loss"].append(self.loss_and_grad(X, y, w, b)[0])
            if X_val is not None:
                pv = self.predict_proba(X_val)
                self.history["val_loss"].append(log_loss(y_val, pv))
                self.history["val_acc"].append(float(((pv >= 0.5) == y_val).mean()))
        return self

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict_proba(self, X):
        return sigmoid(self.decision_function(X))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def gradient_check(X, y, lam=1e-3, n_checks=20, eps=1e-6, seed=0):
    """Centered finite differences against the analytic gradient on random weights."""
    rng = np.random.default_rng(seed)
    model = LogisticRegression(lam=lam)
    w, b = rng.normal(0, 0.5, X.shape[1]), 0.1
    _, g, _ = model.loss_and_grad(X, y, w, b)
    worst = 0.0
    for j in rng.choice(X.shape[1], n_checks, replace=False):
        wp, wm = w.copy(), w.copy()
        wp[j] += eps; wm[j] -= eps
        num = (model.loss_and_grad(X, y, wp, b)[0] - model.loss_and_grad(X, y, wm, b)[0]) / (2 * eps)
        worst = max(worst, abs(num - g[j]) / max(1e-10, abs(num) + abs(g[j])))
    return worst


# ----------------------------------------------------------------------------- metrics from scratch
def confusion(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def classification_metrics(y, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    c = confusion(y, pred)
    precision = c["tp"] / max(1, c["tp"] + c["fp"])
    recall = c["tp"] / max(1, c["tp"] + c["fn"])
    return {"accuracy": (c["tp"] + c["tn"]) / len(y), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(1e-12, precision + recall), "roc_auc": roc_auc(y, prob),
            "log_loss": log_loss(y, prob), "confusion": c}


def roc_auc(y, score):
    """Mann-Whitney U statistic: probability that a random positive outranks a random negative."""
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    ranks = ((ends - counts + 1 + ends) / 2)[inv]  # average rank for ties
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve(y, score):
    order = np.argsort(-score)
    ys = y[order]
    tpr = np.r_[0, np.cumsum(ys) / ys.sum()]
    fpr = np.r_[0, np.cumsum(1 - ys) / (1 - ys).sum()]
    return fpr, tpr
