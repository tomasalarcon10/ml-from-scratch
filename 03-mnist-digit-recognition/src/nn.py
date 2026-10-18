"""
A minimal deep learning library written from scratch in NumPy:
dense, convolution (im2col), max pooling, ReLU, softmax cross-entropy, Adam,
plus a numerical gradient check to prove the backpropagation is correct.
"""

import gzip
from pathlib import Path

import numpy as np

DTYPE = np.float32


# ----------------------------------------------------------------------------- data
def load_mnist(folder: Path):
    def images(name):
        with gzip.open(folder / name, "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)

    def labels(name):
        with gzip.open(folder / name, "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)

    return (images("train-images-idx3-ubyte.gz"), labels("train-labels-idx1-ubyte.gz"),
            images("t10k-images-idx3-ubyte.gz"), labels("t10k-labels-idx1-ubyte.gz"))


# ----------------------------------------------------------------------------- layers
class Layer:
    params: dict = {}
    grads: dict = {}

    def forward(self, x, train=True):
        raise NotImplementedError

    def backward(self, g):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, n_in, n_out, rng):
        self.params = {"W": (rng.standard_normal((n_in, n_out)) * np.sqrt(2 / n_in)).astype(DTYPE),
                       "b": np.zeros(n_out, DTYPE)}
        self.grads = {}

    def forward(self, x, train=True):
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, g):
        self.grads["W"] = self.x.T @ g
        self.grads["b"] = g.sum(0)
        return g @ self.params["W"].T


class ReLU(Layer):
    def forward(self, x, train=True):
        self.mask = x > 0
        return x * self.mask

    def backward(self, g):
        return g * self.mask


class Flatten(Layer):
    def forward(self, x, train=True):
        self.shape = x.shape
        return x.reshape(len(x), -1)

    def backward(self, g):
        return g.reshape(self.shape)


class Dropout(Layer):
    def __init__(self, p, rng):
        self.p, self.rng = p, rng

    def forward(self, x, train=True):
        if not train or self.p == 0:
            return x
        self.mask = (self.rng.random(x.shape) > self.p).astype(DTYPE) / (1 - self.p)
        return x * self.mask

    def backward(self, g):
        return g * self.mask


def _im2col(x, k):
    """x: (N, C, H, W) -> (N*Ho*Wo, C*k*k) for a stride-1, no-padding convolution."""
    N, C, H, W = x.shape
    Ho, Wo = H - k + 1, W - k + 1
    s = x.strides
    patches = np.lib.stride_tricks.as_strided(x, shape=(N, C, Ho, Wo, k, k),
                                              strides=(s[0], s[1], s[2], s[3], s[2], s[3]))
    return patches.transpose(0, 2, 3, 1, 4, 5).reshape(N * Ho * Wo, C * k * k), Ho, Wo


class Conv2D(Layer):
    def __init__(self, c_in, c_out, k, rng):
        fan_in = c_in * k * k
        self.k, self.c_out = k, c_out
        self.params = {"W": (rng.standard_normal((c_out, c_in, k, k)) * np.sqrt(2 / fan_in)).astype(DTYPE),
                       "b": np.zeros(c_out, DTYPE)}
        self.grads = {}

    def forward(self, x, train=True):
        self.x_shape = x.shape
        cols, Ho, Wo = _im2col(np.ascontiguousarray(x), self.k)
        self.cols = cols
        out = cols @ self.params["W"].reshape(self.c_out, -1).T + self.params["b"]
        self.Ho, self.Wo = Ho, Wo
        return out.reshape(x.shape[0], Ho, Wo, self.c_out).transpose(0, 3, 1, 2)

    def backward(self, g):
        N, C, H, W = self.x_shape
        k = self.k
        g2 = g.transpose(0, 2, 3, 1).reshape(-1, self.c_out)
        self.grads["W"] = (g2.T @ self.cols).reshape(self.params["W"].shape)
        self.grads["b"] = g2.sum(0)
        dcols = (g2 @ self.params["W"].reshape(self.c_out, -1)).reshape(N, self.Ho, self.Wo, C, k, k)
        dx = np.zeros(self.x_shape, DTYPE)
        for i in range(k):
            for j in range(k):
                dx[:, :, i:i + self.Ho, j:j + self.Wo] += dcols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
        return dx


class MaxPool2(Layer):
    """2x2 max pooling. The gradient goes only to the first maximum of each window,
    so ties (very common on blank image regions) do not duplicate the gradient."""

    def forward(self, x, train=True):
        N, C, H, W = x.shape
        win = x.reshape(N, C, H // 2, 2, W // 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(N, C, H // 2, W // 2, 4)
        self.arg = win.argmax(-1)
        self.shape = x.shape
        return np.take_along_axis(win, self.arg[..., None], -1)[..., 0]

    def backward(self, g):
        N, C, H, W = self.shape
        d = np.zeros((N, C, H // 2, W // 2, 4), g.dtype)
        np.put_along_axis(d, self.arg[..., None], g[..., None], -1)
        return d.reshape(N, C, H // 2, W // 2, 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(N, C, H, W)


# ----------------------------------------------------------------------------- loss, model, optimizer
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def cross_entropy(logits, y):
    p = softmax(logits)
    loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    g = p.copy()
    g[np.arange(len(y)), y] -= 1
    return float(loss), (g / len(y)).astype(DTYPE)


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x, train=True, upto=None):
        for i, layer in enumerate(self.layers):
            if upto is not None and i == upto:
                return x
            x = layer.forward(x, train)
        return x

    def backward(self, g):
        for layer in reversed(self.layers):
            g = layer.backward(g)

    def parameters(self):
        for layer in self.layers:
            for name in getattr(layer, "params", {}):
                yield layer, name

    def n_params(self):
        return int(sum(l.params[n].size for l, n in self.parameters()))

    def predict(self, x, batch=2000):
        return np.concatenate([self.forward(x[i:i + batch], train=False).argmax(1) for i in range(0, len(x), batch)])

    def logits(self, x, batch=2000):
        return np.concatenate([self.forward(x[i:i + batch], train=False) for i in range(0, len(x), batch)])


class Adam:
    def __init__(self, model, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.model, self.lr, self.b1, self.b2, self.eps, self.t = model, lr, b1, b2, eps, 0
        self.m = {(id(l), n): np.zeros_like(l.params[n]) for l, n in model.parameters()}
        self.v = {(id(l), n): np.zeros_like(l.params[n]) for l, n in model.parameters()}

    def step(self):
        self.t += 1
        for l, n in self.model.parameters():
            key, g = (id(l), n), l.grads[n]
            self.m[key] = self.b1 * self.m[key] + (1 - self.b1) * g
            self.v[key] = self.b2 * self.v[key] + (1 - self.b2) * g * g
            mh = self.m[key] / (1 - self.b1 ** self.t)
            vh = self.v[key] / (1 - self.b2 ** self.t)
            l.params[n] -= (self.lr * mh / (np.sqrt(vh) + self.eps)).astype(DTYPE)


def train(model, Xtr, ytr, Xval, yval, epochs, batch, lr, seed=0, log=print):
    rng = np.random.default_rng(seed)
    opt = Adam(model, lr)
    hist = {"epoch": [], "train_loss": [], "train_acc": [], "val_acc": []}
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(Xtr))
        losses, correct = [], 0
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            logits = model.forward(Xtr[b], train=True)
            loss, g = cross_entropy(logits, ytr[b])
            model.backward(g)
            opt.step()
            losses.append(loss)
            correct += int((logits.argmax(1) == ytr[b]).sum())
        val_acc = float((model.predict(Xval) == yval).mean())
        hist["epoch"].append(ep); hist["train_loss"].append(float(np.mean(losses)))
        hist["train_acc"].append(correct / len(idx)); hist["val_acc"].append(val_acc)
        log(f"    epoch {ep}: loss {np.mean(losses):.4f}  train {correct/len(idx):.4f}  val {val_acc:.4f}")
    return hist


def gradient_check(model, x, y, n_checks=8, eps=1e-6, seed=0):
    """Compares backpropagation gradients with centered finite differences (float64)."""
    rng = np.random.default_rng(seed)
    for l, n in model.parameters():
        l.params[n] = l.params[n].astype(np.float64)
    global DTYPE
    old, DTYPE = DTYPE, np.float64
    logits = model.forward(x, train=False)
    _, g = cross_entropy(logits, y)
    model.backward(g.astype(np.float64))
    worst = 0.0
    for l, n in model.parameters():
        P, G = l.params[n], l.grads[n]
        for _ in range(n_checks):
            i = tuple(rng.integers(0, s) for s in P.shape)
            orig = P[i]
            P[i] = orig + eps; lp, _ = cross_entropy(model.forward(x, train=False), y)
            P[i] = orig - eps; lm, _ = cross_entropy(model.forward(x, train=False), y)
            P[i] = orig
            num = (lp - lm) / (2 * eps)
            rel = abs(num - G[i]) / max(1e-8, abs(num) + abs(G[i]))
            worst = max(worst, rel)
    DTYPE = old
    for l, n in model.parameters():
        l.params[n] = l.params[n].astype(old)
    return worst
