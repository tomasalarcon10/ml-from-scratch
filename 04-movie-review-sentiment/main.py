"""
Sentiment Analysis on Movie Reviews: text preprocessing, TF-IDF and logistic regression from scratch.
Run:  python main.py      (about 3 minutes on a laptop CPU)
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from scipy.sparse.linalg import svds
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer as SkTfidf
from sklearn.linear_model import LogisticRegression as SkLogReg

from src.text import TfidfVectorizer, tokenize, STOP_WORDS
from src.logreg import LogisticRegression, classification_metrics, gradient_check, roc_curve

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

BG, CARD, TEXT, TEXT2, MUTED, GRID = "#1a1a19", "#222220", "#f0efec", "#c3c2b7", "#8a8980", "#3a3936"
POS, NEG, AQUA, VIOLET, RED, AMBER = "#3987e5", "#d95926", "#199e70", "#9085e9", "#e66767", "#c98500"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": TEXT,
    "axes.labelcolor": TEXT2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
    "grid.color": GRID, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})


def plotly_layout(**kw):
    base = dict(paper_bgcolor=BG, plot_bgcolor=BG, font=dict(family="Inter, Segoe UI, Arial", color=TEXT, size=13),
                hoverlabel=dict(bgcolor=CARD, bordercolor=GRID, font=dict(color=TEXT)), margin=dict(l=20, r=20, t=70, b=20))
    base.update(kw)
    return base


def title(ax, text):
    ax.set_title(text, loc="left", fontsize=12.5, fontweight="bold", color=TEXT)


def main():
    # ------------------------------------------------------------- data
    raw = pd.read_csv(ROOT / "data" / "imdb_reviews.csv.gz")
    df = raw.drop_duplicates("review").reset_index(drop=True)
    y = (df["sentiment"] == "positive").astype(int).to_numpy()
    texts = df["review"].to_numpy()
    idx = np.random.default_rng(42).permutation(len(df))
    te, va, tr = idx[:10_000], idx[10_000:15_000], idx[15_000:]
    print(f"{len(raw):,} reviews, {len(raw) - len(df)} duplicates removed -> train {len(tr):,}, val {len(va):,}, test {len(te):,}")

    # ------------------------------------------------------------- features from scratch
    t0 = time.time()
    vec = TfidfVectorizer(n_max=2, min_df=5, max_df=0.9, max_features=50_000)
    Xtr = vec.fit_transform(texts[tr]); Xva = vec.transform(texts[va]); Xte = vec.transform(texts[te])
    vec_secs = time.time() - t0
    terms = vec.terms()
    print(f"TF-IDF: {Xtr.shape[1]:,} unigrams and bigrams in {vec_secs:.0f}s")

    grad_err = gradient_check(Xtr[:300], y[tr][:300].astype(float), eps=1e-5)
    print(f"gradient check, worst relative error: {grad_err:.1e}")

    # ------------------------------------------------------------- regularization selected on validation
    grid = [0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4]
    val_curve = []
    for lam in grid:
        m = LogisticRegression(lam=lam, lr=0.01, epochs=10).fit(Xtr, y[tr], Xva, y[va])
        val_curve.append({"lambda": lam, "val_accuracy": m.history["val_acc"][-1], "val_log_loss": m.history["val_loss"][-1]})
        print(f"  lambda {lam:g}: val accuracy {m.history['val_acc'][-1]:.4f}")
    best_lam = max(val_curve, key=lambda r: (r["val_accuracy"], -r["lambda"]))["lambda"]

    t0 = time.time()
    model = LogisticRegression(lam=best_lam, lr=0.01, epochs=10).fit(Xtr, y[tr], Xva, y[va])
    fit_secs = time.time() - t0
    prob = model.predict_proba(Xte)
    pred = (prob >= 0.5).astype(int)
    m_best = classification_metrics(y[te], prob)
    print(f"Scratch model (lambda {best_lam:g}): test accuracy {m_best['accuracy']:.4f}, AUC {m_best['roc_auc']:.4f}")

    # unigrams only, to measure what bigrams add
    vec1 = TfidfVectorizer(n_max=1, min_df=5, max_df=0.9, max_features=50_000)
    X1tr = vec1.fit_transform(texts[tr]); X1te = vec1.transform(texts[te])
    uni = LogisticRegression(lam=best_lam, lr=0.01, epochs=10).fit(X1tr, y[tr])
    m_uni = classification_metrics(y[te], uni.predict_proba(X1te))

    # scikit-learn with the same tokenizer as a correctness benchmark
    sk_vec = SkTfidf(tokenizer=tokenize, lowercase=False, token_pattern=None, ngram_range=(1, 2), min_df=5,
                     max_df=0.9, max_features=50_000, sublinear_tf=True)
    S_tr = sk_vec.fit_transform(texts[tr]); S_te = sk_vec.transform(texts[te])
    S_va = sk_vec.transform(texts[va])
    sk_C = max([1, 3, 10, 30, 100], key=lambda C: (SkLogReg(C=C, max_iter=2000).fit(S_tr, y[tr]).score(S_va, y[va]), -C))
    sk = SkLogReg(C=sk_C, max_iter=2000).fit(S_tr, y[tr])
    sk_prob = sk.predict_proba(S_te)[:, 1]
    m_sk = classification_metrics(y[te], sk_prob)
    vocab_overlap = len(set(sk_vec.vocabulary_) & set(vec.vocabulary_)) / len(vec.vocabulary_)
    agreement = float(((sk_prob >= 0.5) == pred).mean())
    print(f"scikit-learn: accuracy {m_sk['accuracy']:.4f}; vocabulary overlap {vocab_overlap:.3f}; agreement {agreement:.4f}")

    w = model.w
    order = np.argsort(w)
    top_neg = [(terms[i], float(w[i])) for i in order[:25]]
    top_pos = [(terms[i], float(w[i])) for i in order[::-1][:25]]

    # accuracy by review length
    n_words = np.array([len(t.split()) for t in texts[te]])
    q = np.quantile(n_words, [0, .2, .4, .6, .8, 1])
    length_rows = []
    for lo, hi in zip(q[:-1], q[1:]):
        mk = (n_words >= lo) & (n_words <= hi)
        length_rows.append({"from": int(lo), "to": int(hi), "accuracy": float((pred[mk] == y[te][mk]).mean()), "n": int(mk.sum())})

    # ------------------------------------------------------------- chart 1: 3D review space
    n3 = 3000
    Xs = Xte[:n3]
    score = model.decision_function(Xs)
    _, s_vals, Vt = svds(Xs, k=3)
    Vt = Vt[np.argsort(-s_vals)]
    lsa = np.asarray(Xs @ Vt[1:3].T)
    for k in range(2):  # remove the part of each topic axis already explained by sentiment
        beta = np.polyfit(score, lsa[:, k], 1)
        lsa[:, k] -= np.polyval(beta, score)
    topic_names = []
    for k in (1, 2):
        top = np.argsort(-np.abs(Vt[k]))[:4]
        topic_names.append(", ".join(terms[top]))
    snippets = [t.replace("<br />", " ")[:140].replace("\n", " ") + "..." for t in texts[te][:n3]]
    yy, pp = y[te][:n3], prob[:n3]
    fig3d = go.Figure()
    for label, color, nm in [(1, POS, "Positive review"), (0, NEG, "Negative review")]:
        ok = (yy == label) & ((pp >= 0.5) == label)
        bad = (yy == label) & ((pp >= 0.5) != label)
        for mask, sym, suffix, op in [(ok, "circle", "", 0.75), (bad, "x", " (misclassified)", 1.0)]:
            fig3d.add_trace(go.Scatter3d(
                x=score[mask], y=lsa[mask, 0], z=lsa[mask, 1], mode="markers", name=nm + suffix,
                marker=dict(size=2.4 if sym == "circle" else 2.6, color=color if sym == "circle" else RED, opacity=op, symbol=sym),
                customdata=np.c_[100 * pp[mask]], text=[snippets[i] for i in np.where(mask)[0]],
                hovertemplate="<b>P(positive) %{customdata[0]:.0f}%</b><br>%{text}<extra>" + nm + "</extra>"))
    axis = dict(backgroundcolor=BG, gridcolor=GRID, zerolinecolor=GRID, color=TEXT2, showticklabels=False)
    fig3d.update_layout(**plotly_layout(
        title=dict(text="3,000 test reviews in 3D: sentiment score learned by the model vs. two topic axes (hover to read a review)", x=0.02),
        scene=dict(xaxis=dict(title="Sentiment score  (negative  <-  ->  positive)", **axis),
                   yaxis=dict(title="Topic 1: " + topic_names[0], **axis), zaxis=dict(title="Topic 2: " + topic_names[1], **axis),
                   aspectmode="manual", aspectratio=dict(x=1.7, y=1, z=0.8),
                   camera=dict(eye=dict(x=0.3, y=-1.85, z=0.8), center=dict(x=0, y=0, z=-0.08))),
        legend=dict(font=dict(color=TEXT2), itemsizing="constant", x=0.01, y=0.92), height=780))
    fig3d.write_html(OUT / "01_review_space_3d.html", include_plotlyjs="cdn")
    try:
        fig3d.write_image(OUT / "01_review_space_3d.png", width=1600, height=900, scale=2)
    except Exception as exc:  # noqa: BLE001
        print(f"  (static 3D export skipped: {exc.__class__.__name__})")

    # ------------------------------------------------------------- chart 2: word clouds by learned weight
    def cloud(pairs, colors):
        freqs = {t: abs(v) for t, v in pairs}
        def color_fn(word, font_size, **_):
            return colors[0] if font_size > 70 else colors[1] if font_size > 30 else colors[2]
        return WordCloud(width=1100, height=700, background_color=BG, prefer_horizontal=0.95, max_words=150,
                         color_func=color_fn, random_state=7, margin=6).generate_from_frequencies(freqs)
    pos_pairs = [(terms[i], float(w[i])) for i in order[::-1][:150]]
    neg_pairs = [(terms[i], float(w[i])) for i in order[:150]]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6))
    axes[0].imshow(cloud(pos_pairs, ["#9ec5f5", "#6aa6ec", "#4d8fd9"]), interpolation="bilinear")
    axes[1].imshow(cloud(neg_pairs, ["#f3a27c", "#e2774a", "#cf6a3f"]), interpolation="bilinear")
    title(axes[0], "Words and phrases that push a review to POSITIVE")
    title(axes[1], "Words and phrases that push a review to NEGATIVE")
    for a in axes:
        a.axis("off")
    fig.text(0.01, 0.01, "Size is proportional to the weight learned by the logistic regression, not to raw frequency.", color=TEXT2, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "02_word_clouds.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 3: confusion matrix + ROC
    c = m_best["confusion"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 1.15]})
    cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    shade = np.array([[1, 0], [0, 1]], float)
    axes[0].imshow(shade, cmap=LinearSegmentedColormap.from_list("c", ["#3d2420", "#183a2f"]), vmin=0, vmax=1)
    labels = [["True negative", "False positive"], ["False negative", "True positive"]]
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i - 0.08, f"{cm[i, j]:,}", ha="center", va="center", fontsize=26, fontweight="bold",
                         color=AQUA if i == j else RED)
            axes[0].text(j, i + 0.2, f"{labels[i][j]}  ({100 * cm[i, j] / cm[i].sum():.1f}% of row)", ha="center",
                         va="center", fontsize=10, color=TEXT2)
    axes[0].set_xticks([0, 1], ["Predicted negative", "Predicted positive"]); axes[0].set_yticks([0, 1], ["Negative", "Positive"])
    axes[0].tick_params(colors=TEXT, length=0)
    for sp in axes[0].spines.values():
        sp.set_visible(False)
    title(axes[0], f"Confusion matrix on 10,000 unseen reviews: {100 * m_best['accuracy']:.2f}% correct")
    for p_, name, col in [(uni.predict_proba(X1te), f"Unigrams only (AUC {m_uni['roc_auc']:.3f})", MUTED),
                          (prob, f"Unigrams + bigrams (AUC {m_best['roc_auc']:.3f})", POS)]:
        fpr, tpr = roc_curve(y[te], p_)
        axes[1].plot(fpr, tpr, color=col, lw=2.2, label=name)
    axes[1].plot([0, 1], [0, 1], color=GRID, lw=1, ls="--", label="Random guess (AUC 0.500)")
    axes[1].set_xlabel("False positive rate"); axes[1].set_ylabel("True positive rate")
    axes[1].legend(frameon=False, labelcolor=TEXT2, loc="lower right"); axes[1].grid(lw=0.5)
    title(axes[1], "ROC curve")
    fig.tight_layout()
    fig.savefig(OUT / "03_confusion_and_roc.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 4: strongest n-grams
    k = 18
    names = [t for t, _ in top_neg[:k]][::-1] + [t for t, _ in top_pos[:k]]
    vals = [v for _, v in top_neg[:k]][::-1] + [v for _, v in top_pos[:k]]
    names, vals = names[::-1], vals[::-1]
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.barh(range(len(vals)), vals, color=[POS if v > 0 else NEG for v in vals], height=0.7)
    ax.set_yticks(range(len(vals)), names); ax.tick_params(axis="y", colors=TEXT, length=0)
    ax.axvline(0, color=MUTED, lw=1); ax.invert_yaxis(); ax.grid(axis="x", lw=0.5)
    ax.set_xlabel("Learned weight (log-odds per unit of TF-IDF)")
    title(ax, f"The {k} strongest positive and negative signals out of {len(terms):,} features")
    fig.tight_layout()
    fig.savefig(OUT / "04_top_ngrams.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 5: training + regularization
    h = model.history
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    axes[0].plot(h["epoch"], h["train_loss"], color=POS, lw=2.2, marker="o", ms=4, label="Training loss")
    axes[0].plot(h["epoch"], h["val_loss"], color=AMBER, lw=2.2, marker="o", ms=4, label="Validation loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Log-loss"); axes[0].legend(frameon=False, labelcolor=TEXT2); axes[0].grid(lw=0.5)
    title(axes[0], "Training with mini-batch Adam (from scratch)")
    xs = [max(r["lambda"], 3e-8) for r in val_curve]
    axes[1].plot(xs, [100 * r["val_accuracy"] for r in val_curve], color=VIOLET, lw=2.2, marker="o", ms=6)
    axes[1].set_xscale("log"); axes[1].set_xlabel("L2 strength lambda (log scale; leftmost point is lambda = 0)")
    axes[1].set_ylabel("Validation accuracy (%)"); axes[1].grid(lw=0.5)
    bx = max(best_lam, 3e-8)
    axes[1].scatter([bx], [100 * max(r["val_accuracy"] for r in val_curve)], s=140, facecolors="none", edgecolors=AQUA, lw=2, zorder=5)
    axes[1].annotate("selected", (bx, 100 * max(r["val_accuracy"] for r in val_curve)), xytext=(10, -18),
                     textcoords="offset points", color=AQUA)
    title(axes[1], "Choosing the regularization on the validation set")
    fig.tight_layout()
    fig.savefig(OUT / "05_training_and_regularization.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 6: confidence + length
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.3, 1]})
    bins = np.linspace(0, 1, 41)
    axes[0].hist(prob[y[te] == 1], bins=bins, color=POS, alpha=0.85, label="Truly positive", edgecolor=BG, lw=1.5)
    axes[0].hist(prob[y[te] == 0], bins=bins, color=NEG, alpha=0.6, label="Truly negative", edgecolor=BG, lw=1.5)
    axes[0].axvline(0.5, color=TEXT2, lw=1, ls="--")
    axes[0].set_xlabel("Predicted probability of a positive review"); axes[0].set_ylabel("Reviews")
    axes[0].legend(frameon=False, labelcolor=TEXT2); axes[0].grid(axis="y", lw=0.5)
    title(axes[0], "How confident the model is")
    lab = [f"{r['from']}-{r['to']}" for r in length_rows]
    accs = [100 * r["accuracy"] for r in length_rows]
    axes[1].bar(lab, accs, color=AQUA, width=0.6)
    for i, a in enumerate(accs):
        axes[1].text(i, a + 0.15, f"{a:.1f}%", ha="center", color=TEXT, fontsize=10)
    axes[1].set_ylim(min(accs) - 3, max(accs) + 1.5); axes[1].set_xlabel("Review length (words), quintiles"); axes[1].set_ylabel("Test accuracy (%)")
    axes[1].grid(axis="y", lw=0.5)
    title(axes[1], "Accuracy by review length")
    fig.tight_layout()
    fig.savefig(OUT / "06_confidence_and_length.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- hardest mistakes (for README and dashboard)
    wrong = np.where(pred != y[te])[0]
    conf_wrong = np.abs(prob[wrong] - 0.5)
    mistakes = [{"true": "positive" if y[te][i] else "negative", "p_positive": round(float(prob[i]), 3),
                 "text": texts[te][i].replace("<br />", " ")[:400]} for i in wrong[np.argsort(-conf_wrong)][:6]]

    # ------------------------------------------------------------- live model for the dashboard
    model_js = {"bias": round(float(model.b), 5), "stop": sorted(STOP_WORDS),
                "vocab": {t: [round(float(vec.idf_[i]), 4), round(float(w[i]), 4)] for i, t in enumerate(terms)}}
    (OUT / "model.js").write_text("window.SENTIMENT_MODEL=" + json.dumps(model_js, separators=(",", ":")) + ";", encoding="utf-8")

    # ------------------------------------------------------------- metrics
    clean = lambda d: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
    metrics = {
        "data": {"reviews": len(raw), "duplicates_removed": len(raw) - len(df), "train": len(tr), "validation": len(va), "test": len(te),
                 "features": int(Xtr.shape[1]), "vectorize_seconds": round(vec_secs, 1)},
        "gradient_check_worst_relative_error": grad_err,
        "selected_lambda": best_lam, "regularization_curve": val_curve, "train_seconds": round(fit_secs, 1),
        "test": {"baseline_majority": 0.5, "scratch_unigrams": clean(m_uni), "scratch_unigrams_bigrams": clean(m_best),
                 "sklearn_unigrams_bigrams": clean(m_sk)},
        "sklearn_check": {"selected_C": sk_C, "vocabulary_overlap": round(vocab_overlap, 4), "prediction_agreement": round(agreement, 4)},
        "accuracy_by_length": length_rows, "top_positive": top_pos, "top_negative": top_neg,
        "most_confident_mistakes": mistakes, "history": h,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("data", "selected_lambda", "test", "sklearn_check")}, indent=1))

    # ------------------------------------------------------------- dashboard
    kpis = {"Test reviews": "10,000", "Test accuracy": f"{100 * m_best['accuracy']:.2f}%", "ROC AUC": f"{m_best['roc_auc']:.3f}",
            "Features": f"{Xtr.shape[1]:,}", "Agreement with scikit-learn": f"{100 * agreement:.1f}%"}
    kpi_html = "".join(f'<div class="kpi"><span>{k}</span><b>{v}</b></div>' for k, v in kpis.items())
    sec = lambda t, n_, f: f'<section><h2>{t}</h2><p class="note">{n_}</p>{f.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})}</section>'
    img = lambda t, n_, src: f'<section><h2>{t}</h2><p class="note">{n_}</p><img src="outputs/{src}" alt="{t}"></section>'
    examples = ["An absolute masterpiece. The acting was brilliant and the ending left me speechless, I would watch it again tomorrow.",
                "What a waste of two hours. The plot made no sense, the dialogue was awful and I wanted to leave halfway through.",
                "I expected to hate it, but honestly it was not bad at all. Not great either, just a decent way to spend an evening.",
                "The director clearly tried something different and the photography is beautiful, but the script is boring and predictable."]
    ex_buttons = "".join(f'<button class="ex" data-text="{e}">Example {i + 1}</button>' for i, e in enumerate(examples))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Movie Review Sentiment</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><script src="outputs/model.js"></script>
<style>:root{{--bg:{BG};--card:{CARD};--text:{TEXT};--text2:{TEXT2};--muted:{MUTED};--line:{GRID}}}*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font-family:Inter,"Segoe UI",Arial,sans-serif}}
header,.kpis,footer{{max-width:1240px;margin:auto;padding:0 32px}}header{{padding-top:48px}}h1{{font-size:40px;margin:0 0 8px}}
header p{{color:var(--text2);max-width:840px;line-height:1.55;margin:0}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin-top:24px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px}}
.kpi span{{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.8px}}.kpi b{{display:block;font-size:24px;margin-top:6px}}
section{{max-width:1240px;margin:28px auto;padding:20px 24px;background:var(--card);border:1px solid var(--line);border-radius:14px}}
section img{{width:100%;border-radius:8px}}h2{{margin:4px 0 6px;font-size:20px}}.note{{color:var(--text2);margin:0 0 10px;font-size:14px}}
textarea{{width:100%;min-height:110px;background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:10px;padding:12px;font:15px/1.5 inherit;resize:vertical}}
.row{{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}}button{{background:var(--bg);color:var(--text2);border:1px solid var(--line);border-radius:8px;padding:7px 12px;cursor:pointer;font:inherit;font-size:13px}}
button:hover{{color:var(--text);border-color:var(--muted)}}.meter{{height:14px;border-radius:7px;background:linear-gradient(90deg,{NEG},#3a3936 50%,{POS});position:relative;margin:14px 0 6px}}
.needle{{position:absolute;top:-5px;width:4px;height:24px;background:var(--text);border-radius:2px;transition:left .25s}}
.verdict{{font-size:22px;font-weight:700}}.words span{{display:inline-block;margin:3px;padding:3px 8px;border-radius:6px;font-size:13px}}
.mist{{border-left:3px solid {RED};padding:6px 12px;margin:10px 0;color:var(--text2);font-size:14px;line-height:1.5}}
footer{{color:var(--muted);font-size:13px;margin:24px auto 48px}}a{{color:#7fb0f0}}
@media(max-width:640px){{header,.kpis,footer{{padding:0 16px}}section{{margin:16px;padding:12px}}h1{{font-size:30px}}}}</style></head><body>
<header><h1>Sentiment Analysis on Movie Reviews</h1><p>Text cleaning, TF-IDF and logistic regression written from scratch in NumPy,
trained on {len(tr):,} IMDB reviews and tested on 10,000 it never saw. The model below runs live in your browser.</p></header><div class="kpis">{kpi_html}</div>
<section><h2>Try the model</h2><p class="note">Write a review in English. The exact model trained in Python scores it here, and every word shows how much it pushed the decision.</p>
<textarea id="rev" placeholder="Type a movie review...">{examples[2]}</textarea><div class="row">{ex_buttons}</div>
<div class="meter"><div class="needle" id="needle"></div></div><div class="row" style="justify-content:space-between;color:var(--muted);font-size:12px;margin:0"><span>Negative</span><span>Positive</span></div>
<p class="verdict" id="verdict"></p><div class="words" id="words"></div></section>
{sec("Reviews in 3D", "Drag to rotate, hover any point to read the review. Red crosses are the model's mistakes: most sit close to the boundary.", fig3d)}
{img("Word clouds", "The vocabulary of praise and criticism, sized by the learned weight.", "02_word_clouds.png")}
{img("Confusion matrix and ROC", "Errors are balanced between both classes.", "03_confusion_and_roc.png")}
{img("Strongest signals", "Bigrams such as negations and ratings carry a lot of the signal.", "04_top_ngrams.png")}
{img("Training and regularization", "Mini-batch Adam and the L2 strength chosen on validation data.", "05_training_and_regularization.png")}
{img("Confidence and review length", "Most reviews are classified with high confidence.", "06_confidence_and_length.png")}
<section><h2>The most confident mistakes</h2><p class="note">Sarcasm, mixed opinions and plot summaries full of negative words fool a bag-of-words model.</p>
{"".join(f'<div class="mist"><b>True: {m["true"]}, model said {100 * m["p_positive"]:.0f}% positive.</b><br>{m["text"]}...</div>' for m in mistakes[:4])}</section>
<footer>Tomás Alarcón &middot; <a href="https://github.com/tomasalarcon10/ml-from-scratch">Source code</a></footer>
<script>
const M=window.SENTIMENT_MODEL,STOP=new Set(M.stop);
function tokenize(t){{t=t.toLowerCase().replace(/<[^>]+>/g," ");const out=[];for(let k of (t.match(/[a-z]+(?:'[a-z]+)?/g)||[])){{
if(k.endsWith("n't")||k==="cannot")k="not";else if(k.includes("'"))k=k.split("'")[0];if(!STOP.has(k)&&k.length>1)out.push(k);}}return out;}}
function score(text){{const tok=tokenize(text),grams=tok.slice();for(let i=0;i+1<tok.length;i++)grams.push(tok[i]+" "+tok[i+1]);
const c={{}};for(const g of grams)if(M.vocab[g])c[g]=(c[g]||0)+1;let n2=0;const v={{}};for(const g in c){{v[g]=(1+Math.log(c[g]))*M.vocab[g][0];n2+=v[g]*v[g];}}
const n=Math.sqrt(n2)||1;let z=M.bias;const contrib=[];for(const g in v){{const x=v[g]/n*M.vocab[g][1];z+=x;contrib.push([g,x]);}}
return {{p:1/(1+Math.exp(-z)),contrib:contrib.sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).slice(0,18)}};}}
function render(){{const r=score(document.getElementById("rev").value);document.getElementById("needle").style.left=`calc(${{(r.p*100).toFixed(1)}}% - 2px)`;
const lab=r.p>=0.5?"Positive":"Negative";document.getElementById("verdict").innerHTML=`${{lab}} <span style="color:var(--muted);font-weight:400;font-size:16px">${{(100*r.p).toFixed(1)}}% probability of a positive review</span>`;
document.getElementById("words").innerHTML=r.contrib.map(([g,x])=>{{const a=Math.min(1,Math.abs(x)/0.6);const col=x>0?`rgba(57,135,229,${{0.25+0.6*a}})`:`rgba(217,89,38,${{0.25+0.6*a}})`;
return `<span style="background:${{col}}">${{g}} ${{x>0?"+":""}}${{x.toFixed(2)}}</span>`}}).join("");}}
document.getElementById("rev").addEventListener("input",render);document.querySelectorAll(".ex").forEach(b=>b.onclick=()=>{{document.getElementById("rev").value=b.dataset.text;render();}});render();
</script></body></html>"""
    (ROOT / "dashboard.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
