"""
Text preprocessing and TF-IDF written from scratch.
scipy is used only to store the resulting matrix in sparse form.
"""

import re
from collections import Counter

import numpy as np
from scipy import sparse

# English stop words WITHOUT negations: "not good" and "no plot" carry the sentiment.
STOP_WORDS = set("""
a about above after again against all am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself of off on once only or other our ours
ourselves out over own same she should so some such than that the their theirs them themselves then there these they
this those through to too under until up very was we were what when where which while who whom why will with would
you your yours yourself yourselves s t ll re ve d m o y also film movie one br
""".split())

TAG = re.compile(r"<[^>]+>")
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    """HTML removal, lowercase, words, n't -> not, possessives removed, stop words dropped."""
    text = TAG.sub(" ", text.lower())
    out = []
    for tok in TOKEN.findall(text):
        if tok.endswith("n't") or tok == "cannot":
            tok = "not"
        elif "'" in tok:
            tok = tok.split("'")[0]
        if tok not in STOP_WORDS and len(tok) > 1:
            out.append(tok)
    return out


def ngrams(tokens: list[str], n_max: int = 2) -> list[str]:
    grams = list(tokens)
    if n_max >= 2:
        grams += [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
    return grams


class TfidfVectorizer:
    """Sublinear TF x smoothed IDF with L2 row normalization (same definition as scikit-learn)."""

    def __init__(self, n_max=2, min_df=5, max_df=0.9, max_features=50_000):
        self.n_max, self.min_df, self.max_df, self.max_features = n_max, min_df, max_df, max_features

    def _docs(self, texts):
        return [Counter(ngrams(tokenize(t), self.n_max)) for t in texts]

    def fit_transform(self, texts):
        docs = self._docs(texts)
        n = len(docs)
        df, tf_total = Counter(), Counter()
        for d in docs:
            df.update(d.keys())
            tf_total.update(d)
        keep = [t for t, c in df.items() if self.min_df <= c <= self.max_df * n]
        keep.sort(key=lambda t: (-tf_total[t], t))
        terms = sorted(keep[: self.max_features])
        self.vocabulary_ = {t: i for i, t in enumerate(terms)}
        dfv = np.array([df[t] for t in terms], dtype=np.float64)
        self.idf_ = np.log((1 + n) / (1 + dfv)) + 1
        return self._matrix(docs)

    def transform(self, texts):
        return self._matrix(self._docs(texts))

    def _matrix(self, docs):
        rows, cols, vals = [], [], []
        voc = self.vocabulary_
        for r, d in enumerate(docs):
            for term, c in d.items():
                j = voc.get(term)
                if j is not None:
                    rows.append(r); cols.append(j); vals.append(1 + np.log(c))
        X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(docs), len(voc)), dtype=np.float64)
        X = X.multiply(self.idf_).tocsr()
        norms = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
        norms[norms == 0] = 1
        return sparse.diags(1 / norms) @ X

    def terms(self):
        return np.array(sorted(self.vocabulary_, key=self.vocabulary_.get))
