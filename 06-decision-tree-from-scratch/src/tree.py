"""
A CART decision tree classifier written from scratch in NumPy:
Gini and entropy impurity, information gain, exhaustive best-split search,
pre-pruning (depth, leaf size, minimum gain) and cost-complexity post-pruning.
"""

from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------------- impurity
def gini(counts):
    """Probability that two random samples from the node belong to different classes: 1 - sum p_k^2."""
    counts = np.asarray(counts, dtype=float)
    n = counts.sum(-1, keepdims=True)
    p = np.divide(counts, n, out=np.zeros_like(counts), where=n > 0)
    return 1.0 - (p ** 2).sum(-1)


def entropy(counts):
    """Bits needed to encode the class of a random sample from the node: -sum p_k log2 p_k."""
    counts = np.asarray(counts, dtype=float)
    n = counts.sum(-1, keepdims=True)
    p = np.divide(counts, n, out=np.zeros_like(counts), where=n > 0)
    logp = np.log2(p, out=np.zeros_like(p), where=p > 0)
    return -(p * logp).sum(-1)


CRITERIA = {"gini": gini, "entropy": entropy}


def information_gain(parent_counts, left_counts, right_counts, criterion="gini"):
    f = CRITERIA[criterion]
    nl, nr = np.sum(left_counts), np.sum(right_counts)
    n = nl + nr
    return float(f(parent_counts) - (nl / n) * f(left_counts) - (nr / n) * f(right_counts))


def scan_feature(x, y, n_classes, criterion="gini", min_samples_leaf=1):
    """
    Evaluates every possible threshold on one feature at once.
    Sorting the values and accumulating class counts gives the left/right class counts of every split
    in one pass, so the search costs O(n log n) per feature instead of O(n^2).
    Returns candidate thresholds (midpoints between distinct values) and their information gain.
    """
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order]
    onehot = np.eye(n_classes)[ys]
    left = np.cumsum(onehot, axis=0)[:-1]
    total = onehot.sum(0)
    right = total - left
    n = len(xs)
    nl = np.arange(1, n)
    valid = (xs[1:] > xs[:-1]) & (nl >= min_samples_leaf) & (n - nl >= min_samples_leaf)
    f = CRITERIA[criterion]
    gain = f(total) - (nl / n) * f(left) - ((n - nl) / n) * f(right)
    thresholds = (xs[1:] + xs[:-1]) / 2
    return thresholds[valid], gain[valid]


# ----------------------------------------------------------------------------- tree
@dataclass
class Node:
    counts: np.ndarray
    impurity: float
    depth: int
    feature: int = -1
    threshold: float = np.nan
    gain: float = 0.0
    left: "Node" = None
    right: "Node" = None
    id: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def is_leaf(self):
        return self.left is None

    @property
    def n(self):
        return int(self.counts.sum())


class DecisionTree:
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_gain=0.0):
        self.criterion, self.max_depth = criterion, max_depth
        self.min_samples_split, self.min_samples_leaf, self.min_gain = min_samples_split, min_samples_leaf, min_gain

    # ------------------------------------------------------------- fitting
    def fit(self, X, y, n_classes=None):
        X, y = np.asarray(X, float), np.asarray(y, int)
        self.n_classes_ = n_classes or int(y.max()) + 1
        self.n_features_ = X.shape[1]
        self.n_train_ = len(y)
        self._next_id = 0
        self.root_ = self._grow(X, y, 0)
        return self

    def _grow(self, X, y, depth):
        counts = np.bincount(y, minlength=self.n_classes_).astype(float)
        node = Node(counts=counts, impurity=float(CRITERIA[self.criterion](counts)), depth=depth, id=self._next_id)
        self._next_id += 1
        if (node.impurity == 0 or len(y) < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)):
            return node
        best = (0.0, -1, np.nan)
        for j in range(X.shape[1]):
            thr, gain = scan_feature(X[:, j], y, self.n_classes_, self.criterion, self.min_samples_leaf)
            if len(gain) and gain.max() > best[0] + 1e-12:
                k = int(np.argmax(gain))
                best = (float(gain[k]), j, float(thr[k]))
        if best[1] < 0 or best[0] <= self.min_gain:
            return node
        node.gain, node.feature, node.threshold = best
        mask = X[:, node.feature] <= node.threshold
        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        return node

    # ------------------------------------------------------------- prediction
    def _leaf(self, x):
        node = self.root_
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def predict_proba(self, X):
        return np.array([self._leaf(x).counts / self._leaf(x).counts.sum() for x in np.asarray(X, float)])

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def decision_path(self, x):
        node, path = self.root_, []
        while True:
            path.append(node)
            if node.is_leaf:
                return path
            node = node.left if x[node.feature] <= node.threshold else node.right

    # ------------------------------------------------------------- inspection
    def nodes(self, node=None):
        node = node or self.root_
        yield node
        if not node.is_leaf:
            yield from self.nodes(node.left)
            yield from self.nodes(node.right)

    def n_leaves(self):
        return sum(n.is_leaf for n in self.nodes())

    def depth(self):
        return max(n.depth for n in self.nodes())

    def feature_importances(self):
        """Total impurity decrease contributed by each feature, weighted by the share of samples reaching the node."""
        imp = np.zeros(self.n_features_)
        for n in self.nodes():
            if not n.is_leaf:
                imp[n.feature] += n.n / self.n_train_ * n.gain
        return imp / imp.sum() if imp.sum() > 0 else imp

    def to_dict(self, node=None):
        node = node or self.root_
        d = {"id": node.id, "counts": node.counts.astype(int).tolist(), "impurity": round(node.impurity, 4)}
        if not node.is_leaf:
            d.update(feature=node.feature, threshold=round(node.threshold, 3), gain=round(node.gain, 4),
                     left=self.to_dict(node.left), right=self.to_dict(node.right))
        return d

    # ------------------------------------------------------------- cost-complexity pruning
    def _weighted_impurity(self, node):
        return node.n / self.n_train_ * node.impurity

    def _subtree(self, node):
        """(sum of weighted leaf impurity, number of leaves) of the subtree rooted at node."""
        if node.is_leaf:
            return self._weighted_impurity(node), 1
        rl, ll = self._subtree(node.left)
        rr, lr = self._subtree(node.right)
        return rl + rr, ll + lr

    def pruning_path(self):
        """
        Weakest-link pruning (Breiman et al., 1984). For every internal node t:
            alpha(t) = (R(t) - R(T_t)) / (|leaves(T_t)| - 1)
        Repeatedly collapses the node with the smallest alpha. Returns the sequence of alphas
        and the total leaf impurity after each step (same definition as scikit-learn).
        """
        import copy
        tree = copy.deepcopy(self)
        alphas, impurities = [0.0], [tree._subtree(tree.root_)[0]]
        while not tree.root_.is_leaf:
            internal = [n for n in tree.nodes() if not n.is_leaf]
            scores = []
            for n in internal:
                r_sub, leaves = tree._subtree(n)
                scores.append((tree._weighted_impurity(n) - r_sub) / (leaves - 1))
            a = min(scores)
            for n, s in zip(internal, scores):
                if s <= a + 1e-12:
                    n.left = n.right = None
            alphas.append(float(a)); impurities.append(tree._subtree(tree.root_)[0])
        return np.array(alphas), np.array(impurities)

    def prune(self, alpha):
        """Returns a copy pruned with complexity parameter alpha."""
        import copy
        tree = copy.deepcopy(self)
        while not tree.root_.is_leaf:
            internal = [n for n in tree.nodes() if not n.is_leaf]
            scores = [(tree._weighted_impurity(n) - tree._subtree(n)[0]) / (tree._subtree(n)[1] - 1) for n in internal]
            a = min(scores)
            if a > alpha + 1e-12:
                break
            for n, s in zip(internal, scores):
                if s <= a + 1e-12:
                    n.left = n.right = None
        return tree


# ----------------------------------------------------------------------------- validation
def stratified_kfold(y, k=5, seed=0):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cls in np.unique(y):
        idx = rng.permutation(np.where(y == cls)[0])
        for i, chunk in enumerate(np.array_split(idx, k)):
            folds[i].extend(chunk)
    return [np.array(f) for f in folds]


def cross_val_accuracy(make_model, X, y, k=5, repeats=5):
    scores = []
    for r in range(repeats):
        folds = stratified_kfold(y, k, seed=r)
        for i in range(k):
            te = folds[i]
            tr = np.concatenate([folds[j] for j in range(k) if j != i])
            m = make_model().fit(X[tr], y[tr], n_classes=int(y.max()) + 1)
            scores.append((m.predict(X[te]) == y[te]).mean())
    return float(np.mean(scores)), float(np.std(scores))
