"""
Linear models implemented from scratch with NumPy, then benchmarked against scikit-learn.
"""

import numpy as np


class StandardScalerScratch:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_


class LinearRegressionNormalEquation:
    """Ordinary least squares: beta = (X'X)^-1 X'y, solved with a pseudo-inverse for stability."""

    def fit(self, X, y):
        Xb = np.c_[np.ones(len(X)), X]
        self.beta_ = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y
        self.intercept_, self.coef_ = self.beta_[0], self.beta_[1:]
        return self

    def predict(self, X):
        return self.intercept_ + X @ self.coef_


class RidgeClosedForm:
    """Ridge regression: beta = (X'X + lambda*I)^-1 X'y. The intercept is not penalized."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        Xb = np.c_[np.ones(len(X)), X]
        penalty = self.alpha * np.eye(Xb.shape[1])
        penalty[0, 0] = 0.0
        self.beta_ = np.linalg.solve(Xb.T @ Xb + penalty, Xb.T @ y)
        self.intercept_, self.coef_ = self.beta_[0], self.beta_[1:]
        return self

    def predict(self, X):
        return self.intercept_ + X @ self.coef_


class LinearRegressionGD:
    """Batch gradient descent on mean squared error with optional L2 penalty."""

    def __init__(self, lr=0.05, epochs=3000, alpha=0.0):
        self.lr, self.epochs, self.alpha = lr, epochs, alpha

    def fit(self, X, y):
        n, p = X.shape
        self.coef_ = np.zeros(p)
        self.intercept_ = y.mean()
        self.loss_history_ = []
        for _ in range(self.epochs):
            err = self.intercept_ + X @ self.coef_ - y
            self.loss_history_.append(float(np.mean(err ** 2)))
            grad_w = (2 / n) * (X.T @ err) + 2 * self.alpha * self.coef_ / n
            grad_b = (2 / n) * err.sum()
            self.coef_ -= self.lr * grad_w
            self.intercept_ -= self.lr * grad_b
        return self

    def predict(self, X):
        return self.intercept_ + X @ self.coef_


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred):
    return float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))


def kfold_indices(n, k=5, seed=42):
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, test


def cross_validate(make_model, X, y, k=5):
    """Scaling is fitted inside each fold to avoid leakage."""
    scores = []
    for tr, te in kfold_indices(len(X), k):
        sc = StandardScalerScratch().fit(X[tr])
        m = make_model().fit(sc.transform(X[tr]), y[tr])
        scores.append(rmse(y[te], m.predict(sc.transform(X[te]))))
    return float(np.mean(scores)), float(np.std(scores))
