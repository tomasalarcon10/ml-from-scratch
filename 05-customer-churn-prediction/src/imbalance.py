"""
Tools for imbalanced classification written from scratch in NumPy:
stratified splitting, random oversampling and SMOTE.
"""

import numpy as np


def stratified_split(y, fractions=(0.6, 0.2, 0.2), seed=42):
    """Splits indices so every part keeps the same share of each class."""
    rng = np.random.default_rng(seed)
    parts = [[] for _ in fractions]
    for cls in np.unique(y):
        idx = rng.permutation(np.where(y == cls)[0])
        cuts = np.round(np.cumsum(fractions)[:-1] * len(idx)).astype(int)
        for p, chunk in zip(parts, np.split(idx, cuts)):
            p.extend(chunk)
    return [rng.permutation(np.array(p)) for p in parts]


def random_oversample(X, y, seed=0):
    """Duplicates minority rows at random until both classes have the same size."""
    rng = np.random.default_rng(seed)
    minority = np.where(y == 1)[0]
    extra = rng.choice(minority, (y == 0).sum() - len(minority), replace=True)
    idx = np.r_[np.arange(len(y)), extra]
    return X[idx], y[idx]


def smote(X, y, k=5, binary_cols=(), seed=0):
    """
    SMOTE (Chawla et al., 2002): creates synthetic minority rows on the segment between
    a minority point and one of its k nearest minority neighbors.
    Distances use standardized features; binary columns are rounded back to 0 or 1.
    """
    rng = np.random.default_rng(seed)
    Xm = X[y == 1]
    n_new = (y == 0).sum() - len(Xm)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = (Xm - mu) / sd
    d2 = (Z ** 2).sum(1)[:, None] + (Z ** 2).sum(1)[None, :] - 2 * Z @ Z.T
    np.fill_diagonal(d2, np.inf)
    neighbors = np.argsort(d2, axis=1)[:, :k]
    base = rng.integers(0, len(Xm), n_new)
    nb = neighbors[base, rng.integers(0, k, n_new)]
    gap = rng.random((n_new, 1))
    synth = Xm[base] + gap * (Xm[nb] - Xm[base])
    for c in binary_cols:
        synth[:, c] = np.where(rng.random(n_new) < gap[:, 0], Xm[nb, c], Xm[base, c])
    return np.vstack([X, synth]), np.r_[y, np.ones(n_new, dtype=y.dtype)]
