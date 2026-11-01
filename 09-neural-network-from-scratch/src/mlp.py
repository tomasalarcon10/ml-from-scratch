"""
A multilayer perceptron written from scratch in NumPy, with the forward pass and backpropagation spelled out
equation by equation (no layer objects, no autograd).

Notation for layer l:   Z_l = A_{l-1} W_l + b_l      A_l = g(Z_l)      A_0 = X
Output:                 P = softmax(Z_L)             loss = mean cross-entropy + (lam / 2) sum ||W_l||^2
Backward:               dZ_L = (P - Y) / n
                        dW_l = A_{l-1}^T dZ_l + lam W_l      db_l = sum_rows(dZ_l)
                        dA_{l-1} = dZ_l W_l^T                dZ_{l-1} = dA_{l-1} * g'(Z_{l-1})
"""

import numpy as np

ACTIVATIONS = {
    "tanh": (np.tanh, lambda z, a: 1 - a ** 2),
    "relu": (lambda z: np.maximum(z, 0), lambda z, a: (z > 0).astype(z.dtype)),
    "sigmoid": (lambda z: 1 / (1 + np.exp(-z)), lambda z, a: a * (1 - a)),
}


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def one_hot(y, k):
    return np.eye(k)[y]


class MLP:
    def __init__(self, sizes, activation="tanh", seed=0, init="xavier"):
        rng = np.random.default_rng(seed)
        self.sizes, self.activation = sizes, activation
        self.g, self.dg = ACTIVATIONS[activation]
        self.W, self.b = [], []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            scale = np.sqrt(2 / n_in) if init == "he" else np.sqrt(1 / n_in) if init == "xavier" else 1.0
            self.W.append(rng.normal(0, scale, (n_in, n_out)))
            self.b.append(np.zeros(n_out))

    # ------------------------------------------------------------- forward
    def forward(self, X):
        """Returns the softmax output and the cache of every Z and A needed by backpropagation."""
        A, Zs, As = X, [], [X]
        L = len(self.W)
        for l in range(L):
            Z = A @ self.W[l] + self.b[l]
            A = softmax(Z) if l == L - 1 else self.g(Z)
            Zs.append(Z); As.append(A)
        return A, (Zs, As)

    def loss(self, X, y, lam=0.0):
        P, _ = self.forward(X)
        ce = -np.mean(np.log(P[np.arange(len(y)), y] + 1e-12))
        return float(ce + 0.5 * lam * sum((W ** 2).sum() for W in self.W))

    # ------------------------------------------------------------- backward
    def backward(self, X, y, lam=0.0):
        P, (Zs, As) = self.forward(X)
        n, L = len(y), len(self.W)
        dZ = (P - one_hot(y, P.shape[1])) / n
        dW, db = [None] * L, [None] * L
        for l in range(L - 1, -1, -1):
            dW[l] = As[l].T @ dZ + lam * self.W[l]
            db[l] = dZ.sum(0)
            if l > 0:
                dA = dZ @ self.W[l].T
                dZ = dA * self.dg(Zs[l - 1], As[l])
        ce = -np.mean(np.log(P[np.arange(n), y] + 1e-12))
        return float(ce + 0.5 * lam * sum((W ** 2).sum() for W in self.W)), dW, db

    # ------------------------------------------------------------- training
    def fit(self, X, y, epochs=2000, lr=0.5, lam=0.0, momentum=0.9, X_val=None, y_val=None,
            snapshot_at=(), record_grad_norms=False):
        vW = [np.zeros_like(W) for W in self.W]; vb = [np.zeros_like(b) for b in self.b]
        hist = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "grad_norms": []}
        snapshots = {}
        for ep in range(epochs + 1):
            if ep in snapshot_at:
                snapshots[ep] = ([W.copy() for W in self.W], [b.copy() for b in self.b])
            loss, dW, db = self.backward(X, y, lam)
            if not np.isfinite(loss):
                hist["epoch"].append(ep); hist["train_loss"].append(float("nan"))
                break
            if ep % 10 == 0 or ep == epochs:
                hist["epoch"].append(ep); hist["train_loss"].append(loss)
                hist["train_acc"].append(float((self.predict(X) == y).mean()))
                if X_val is not None:
                    hist["val_loss"].append(self.loss(X_val, y_val, lam))
                    hist["val_acc"].append(float((self.predict(X_val) == y_val).mean()))
                if record_grad_norms:
                    hist["grad_norms"].append([float(np.linalg.norm(g)) for g in dW])
            if ep == epochs:
                break
            for l in range(len(self.W)):
                vW[l] = momentum * vW[l] - lr * dW[l]; vb[l] = momentum * vb[l] - lr * db[l]
                self.W[l] += vW[l]; self.b[l] += vb[l]
        return hist, snapshots

    def predict_proba(self, X):
        return self.forward(X)[0]

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def hidden(self, X, layer):
        """Activations of hidden layer `layer` (1-based)."""
        return self.forward(X)[1][1][layer]

    def n_params(self):
        return int(sum(W.size + b.size for W, b in zip(self.W, self.b)))


def gradient_check(model, X, y, lam=1e-3, eps=1e-6, n_checks=30, seed=0):
    """Compares every backpropagation gradient with a centered finite difference of the loss."""
    rng = np.random.default_rng(seed)
    _, dW, db = model.backward(X, y, lam)
    worst = 0.0
    params = [(model.W, dW), (model.b, db)]
    for _ in range(n_checks):
        group, grads = params[rng.integers(2)]
        l = rng.integers(len(group))
        idx = tuple(rng.integers(0, s) for s in group[l].shape)
        orig = group[l][idx]
        group[l][idx] = orig + eps; lp = model.loss(X, y, lam)
        group[l][idx] = orig - eps; lm = model.loss(X, y, lam)
        group[l][idx] = orig
        num = (lp - lm) / (2 * eps)
        worst = max(worst, abs(num - grads[l][idx]) / max(1e-10, abs(num) + abs(grads[l][idx])))
    return worst


def make_spirals(n_per_class=200, k=3, noise=0.2, seed=0):
    """The classic intertwined spirals: impossible for a linear model, easy to see in 2D."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(k):
        r = np.linspace(0.05, 1, n_per_class)
        t = np.linspace(c * 4, (c + 1) * 4, n_per_class) + rng.normal(0, noise, n_per_class)
        X.append(np.c_[r * np.sin(t), r * np.cos(t)]); y.append(np.full(n_per_class, c))
    return np.vstack(X), np.concatenate(y)
