"""
Principal Component Analysis written from scratch in NumPy:
covariance matrix, eigen-decomposition with the Jacobi rotation method, power iteration with deflation,
projection, reconstruction, and the Marchenko-Pastur law to separate signal from noise.
"""

import numpy as np


def standardize(X):
    mu, sd = X.mean(0), X.std(0, ddof=1)
    return (X - mu) / sd, mu, sd


def covariance(X):
    """Sample covariance: (X - mean)^T (X - mean) / (n - 1). On standardized data this is the correlation matrix."""
    Xc = X - X.mean(0)
    return Xc.T @ Xc / (len(X) - 1)


def jacobi_eigh(A, tol=1e-12, max_sweeps=50):
    """
    Cyclic Jacobi eigenvalue algorithm for a symmetric matrix.
    Each rotation zeroes one off-diagonal pair (p, q); repeated sweeps drive the matrix to diagonal form.
    The accumulated rotations are the eigenvectors. Returns eigenvalues (descending), eigenvectors (columns)
    and the relative off-diagonal norm before each sweep (its convergence history).
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    V = np.eye(n)
    history = []
    for _ in range(max_sweeps):
        off = np.sqrt(np.sum(np.tril(A, -1) ** 2)) / np.sqrt(np.sum(A ** 2))
        history.append(float(off))
        if off < tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = A[p, q]
                if abs(apq) < 1e-300:
                    continue
                theta = (A[q, q] - A[p, p]) / (2 * apq)
                t = np.sign(theta) / (abs(theta) + np.sqrt(theta ** 2 + 1)) if theta != 0 else 1.0
                c = 1 / np.sqrt(t ** 2 + 1); s = t * c
                Ap, Aq = A[:, p].copy(), A[:, q].copy()
                A[:, p] = c * Ap - s * Aq; A[:, q] = s * Ap + c * Aq
                Ap, Aq = A[p, :].copy(), A[q, :].copy()
                A[p, :] = c * Ap - s * Aq; A[q, :] = s * Ap + c * Aq
                Vp, Vq = V[:, p].copy(), V[:, q].copy()
                V[:, p] = c * Vp - s * Vq; V[:, q] = s * Vp + c * Vq
    vals = np.diag(A)
    order = np.argsort(-vals)
    return vals[order], V[:, order], history


def power_iteration(A, k, tol=1e-12, max_iter=10_000, seed=0):
    """
    Top-k eigenpairs by power iteration with deflation: multiply a random vector by A until its direction stops changing
    (that direction is the leading eigenvector), remove that component from A, repeat.
    Returns eigenvalues, eigenvectors and the Rayleigh-quotient history of each eigenpair.
    """
    rng = np.random.default_rng(seed)
    A = np.array(A, dtype=float)
    vals, vecs, history = [], [], []
    for _ in range(k):
        v = rng.normal(size=A.shape[0]); v /= np.linalg.norm(v)
        hist, lam = [], 0.0
        for _ in range(max_iter):
            w = A @ v
            v_new = w / np.linalg.norm(w)
            lam_new = float(v_new @ A @ v_new)
            hist.append(lam_new)
            if abs(lam_new - lam) < tol * max(1.0, abs(lam_new)):
                v = v_new; lam = lam_new
                break
            v, lam = v_new, lam_new
        vals.append(lam); vecs.append(v); history.append(hist)
        A = A - lam * np.outer(v, v)
    return np.array(vals), np.array(vecs).T, history


def align_signs(V, reference=None):
    """Eigenvectors are defined up to sign; make the largest-magnitude entry (or the reference) positive."""
    V = V.copy()
    for j in range(V.shape[1]):
        ref = V[:, j] if reference is None else reference[:, j]
        s = np.sign(ref[np.argmax(np.abs(ref))]) if reference is None else np.sign(V[:, j] @ ref)
        V[:, j] *= s if s != 0 else 1
    return V


class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(0)
        C = covariance(X)
        vals, vecs, self.jacobi_history_ = jacobi_eigh(C)
        k = self.n_components or len(vals)
        self.covariance_ = C
        self.eigenvalues_ = vals
        self.components_ = align_signs(vecs)[:, :k]
        self.explained_variance_ratio_ = vals / vals.sum()
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_

    def inverse_transform(self, Z):
        return Z @ self.components_.T + self.mean_


def marchenko_pastur(q, sigma2=1.0, points=400):
    """
    Eigenvalue density of the correlation matrix of pure noise with N variables and T observations (q = N / T).
    Eigenvalues above lambda_plus carry real structure; the rest is indistinguishable from randomness.
    """
    lam_minus, lam_plus = sigma2 * (1 - np.sqrt(q)) ** 2, sigma2 * (1 + np.sqrt(q)) ** 2
    x = np.linspace(lam_minus, lam_plus, points)
    density = np.sqrt(np.maximum((lam_plus - x) * (x - lam_minus), 0)) / (2 * np.pi * sigma2 * q * x)
    return lam_minus, lam_plus, x, density
