"""
Linear regression from scratch: least squares by the normal equation (solved with Gaussian elimination written by hand),
batch / stochastic / mini-batch gradient descent, the exact learning-rate stability limit, and classical inference.

Model:   y = X beta + e        Loss:   J(beta) = (1 / 2n) ||X beta - y||^2
Gradient:   grad J = X^T (X beta - y) / n        Hessian:   H = X^T X / n   (constant: J is a quadratic bowl)
Normal equation:   X^T X beta = X^T y
Gradient descent converges if and only if 0 < learning_rate < 2 / lambda_max(H).
"""

import numpy as np
from scipy import stats


def add_intercept(x):
    x = np.asarray(x, float)
    return np.c_[np.ones(len(x)), x if x.ndim > 1 else x[:, None]]


def solve_gaussian(A, b):
    """Solves A x = b by Gaussian elimination with partial pivoting and back substitution."""
    A = np.array(A, float); b = np.array(b, float).ravel()
    n = len(b)
    M = np.c_[A, b]
    for k in range(n):
        p = k + int(np.argmax(np.abs(M[k:, k])))
        M[[k, p]] = M[[p, k]]
        for i in range(k + 1, n):
            M[i, k:] -= M[i, k] / M[k, k] * M[k, k:]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - M[i, i + 1:n] @ x[i + 1:]) / M[i, i]
    return x


def normal_equation(X, y):
    return solve_gaussian(X.T @ X, X.T @ y)


def simple_closed_form(x, y):
    """One predictor: slope = cov(x, y) / var(x), intercept = mean(y) - slope * mean(x)."""
    xm, ym = x.mean(), y.mean()
    slope = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum()
    return np.array([ym - slope * xm, slope])


def loss(X, y, beta):
    r = X @ beta - y
    return float(r @ r / (2 * len(y)))


def gradient(X, y, beta):
    return X.T @ (X @ beta - y) / len(y)


def hessian(X):
    return X.T @ X / len(X)


def gradient_descent(X, y, lr, iters, beta0=None, tol=0.0):
    """Batch gradient descent; returns the final coefficients and the full path of coefficients and losses."""
    beta = np.zeros(X.shape[1]) if beta0 is None else np.array(beta0, float)
    path, losses = [beta.copy()], [loss(X, y, beta)]
    for _ in range(iters):
        g = gradient(X, y, beta)
        beta = beta - lr * g
        path.append(beta.copy()); losses.append(loss(X, y, beta))
        if not np.isfinite(losses[-1]) or abs(losses[-1]) > 1e12:
            break
        if tol and np.linalg.norm(g) < tol:
            break
    return beta, np.array(path), np.array(losses)


def minibatch_gd(X, y, lr, epochs, batch, beta0=None, seed=0, decay=0.0):
    """Stochastic (batch = 1) or mini-batch gradient descent with an optional 1 / (1 + decay * t) learning-rate schedule."""
    rng = np.random.default_rng(seed)
    beta = np.zeros(X.shape[1]) if beta0 is None else np.array(beta0, float)
    losses, t = [loss(X, y, beta)], 0
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(len(y)), max(1, len(y) // batch)):
            beta = beta - lr / (1 + decay * t) * gradient(X[idx], y[idx], beta)
            t += 1
        losses.append(loss(X, y, beta))
    return beta, np.array(losses)


def inference(X, y, beta, level=0.95):
    """Standard errors, t statistics, p-values and confidence intervals under the classical OLS assumptions."""
    n, p = X.shape
    resid = y - X @ beta
    sigma2 = resid @ resid / (n - p)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    pval = 2 * stats.t.sf(np.abs(t), n - p)
    q = stats.t.ppf(0.5 + level / 2, n - p)
    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
    return {"se": se, "t": t, "p": pval, "ci_low": beta - q * se, "ci_high": beta + q * se, "sigma": float(np.sqrt(sigma2)),
            "r2": float(r2), "adj_r2": float(1 - (1 - r2) * (n - 1) / (n - p)), "cov": cov, "t_crit": float(q)}
