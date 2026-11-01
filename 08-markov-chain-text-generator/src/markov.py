"""
Markov chains written from scratch: n-gram counting, sampling with temperature, interpolated language model
with weights learned by EM, perplexity, transition matrices, stationary distributions and mixing.
"""

import re
from collections import Counter, defaultdict

import numpy as np

WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,!?;:\-]|\n")


def word_tokens(text):
    return WORD.findall(text)


def detokenize(tokens):
    out = ""
    for t in tokens:
        if t == "\n":
            out = out.rstrip(" ") + "\n"
        elif t in ".,!?;:" or not out or out.endswith("\n"):
            out += t
        else:
            out += " " + t
    return out


class MarkovChain:
    """An order-k Markov chain over any sequence of tokens: P(next | previous k tokens)."""

    def __init__(self, order):
        self.order = order

    def fit(self, tokens):
        self.counts = defaultdict(Counter)
        k = self.order
        for i in range(k, len(tokens)):
            self.counts[tuple(tokens[i - k:i])][tokens[i]] += 1
        self.vocab = sorted(set(tokens))
        return self

    def distribution(self, context):
        c = self.counts.get(tuple(context[-self.order:]) if self.order else ())
        if not c:
            return None
        total = sum(c.values())
        return {t: n / total for t, n in c.items()}

    def generate(self, length, seed=None, temperature=1.0, rng=None):
        """Samples one token at a time from P(next | context); temperature < 1 sharpens, > 1 flattens."""
        rng = rng or np.random.default_rng(0)
        k = self.order
        keys = list(self.counts)
        out = list(seed) if seed else list(keys[rng.integers(len(keys))])
        for _ in range(length):
            c = self.counts.get(tuple(out[-k:]) if k else ())
            if not c:
                c = self.counts[keys[rng.integers(len(keys))]]
            toks = list(c)
            w = np.array([c[t] for t in toks], float) ** (1 / temperature)
            out.append(toks[rng.choice(len(toks), p=w / w.sum())])
        return out


class InterpolatedLM:
    """
    Language model mixing orders 0..K (Jelinek-Mercer interpolation):
        P(w | h) = sum_j lambda_j * P_j(w | last j tokens),   P_0 = add-one unigram so nothing has probability zero.
    The weights lambda are learned on held-out data with the EM algorithm.
    """

    def __init__(self, max_order):
        self.K = max_order

    def fit(self, tokens):
        self.chains = [MarkovChain(j).fit(tokens) for j in range(self.K + 1)]
        self.totals = [{h: sum(c.values()) for h, c in ch.counts.items()} for ch in self.chains]
        uni = self.chains[0].counts[()]
        self.V = len(uni) + 1
        self.N = sum(uni.values())
        self.lam = np.full(self.K + 1, 1 / (self.K + 1))
        return self

    def _component_probs(self, tokens):
        """Matrix (positions x orders) with each order's estimate of the actual next token."""
        n = len(tokens)
        P = np.zeros((n - self.K, self.K + 1))
        uni = self.chains[0].counts[()]
        for r, i in enumerate(range(self.K, n)):
            w = tokens[i]
            P[r, 0] = (uni.get(w, 0) + 1) / (self.N + self.V)
            for j in range(1, self.K + 1):
                h = tuple(tokens[i - j:i])
                tot = self.totals[j].get(h)
                if tot:
                    P[r, j] = self.chains[j].counts[h].get(w, 0) / tot
        return P

    def fit_weights(self, tokens, iters=30):
        P = self._component_probs(tokens)
        for _ in range(iters):
            mix = P * self.lam
            resp = mix / mix.sum(1, keepdims=True)
            self.lam = resp.mean(0)
        return self.lam

    def cross_entropy(self, tokens):
        """Average bits per token on unseen text; perplexity = 2 ** cross_entropy."""
        P = self._component_probs(tokens)
        return float(-np.mean(np.log2(P @ self.lam)))


# ----------------------------------------------------------------------------- transition matrices
def transition_matrix(seq, states):
    idx = {s: i for i, s in enumerate(states)}
    C = np.zeros((len(states), len(states)))
    a = np.array([idx[s] for s in seq])
    np.add.at(C, (a[:-1], a[1:]), 1)
    return C / C.sum(1, keepdims=True), C


def stationary_power(P, tol=1e-14, max_iter=10_000):
    """pi = pi P: start uniform and keep stepping the chain until the distribution stops changing."""
    pi = np.full(len(P), 1 / len(P))
    history = []
    for _ in range(max_iter):
        new = pi @ P
        history.append(float(np.abs(new - pi).sum()))
        if history[-1] < tol:
            return new, history
        pi = new
    return pi, history


def stationary_eigen(P):
    """The stationary distribution is the left eigenvector of P with eigenvalue 1."""
    vals, vecs = np.linalg.eig(P.T)
    k = int(np.argmin(np.abs(vals - 1)))
    v = np.real(vecs[:, k])
    return v / v.sum(), vals


def entropy_rate(P, pi):
    """Average uncertainty (bits) about the next state once the chain is in equilibrium."""
    logP = np.log2(P, out=np.zeros_like(P), where=P > 0)
    return float(-(pi[:, None] * P * logP).sum())


def copy_rate(generated, training_ngrams, n):
    """Share of the n-grams in a generated text that appear word for word in the training text."""
    grams = [tuple(generated[i:i + n]) for i in range(len(generated) - n + 1)]
    return float(np.mean([g in training_ngrams for g in grams])) if grams else 0.0
