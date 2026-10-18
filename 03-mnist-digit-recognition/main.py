"""
MNIST Handwritten Digit Recognition with a neural network library written from scratch in NumPy.
Run:  python main.py      (about 5 minutes on a laptop CPU)
"""

import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go

from src.nn import (load_mnist, Sequential, Dense, Conv2D, MaxPool2, ReLU, Flatten, Dropout,
                    train, gradient_check)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

BG, CARD, TEXT, TEXT2, MUTED, GRID = "#1a1a19", "#222220", "#f0efec", "#c3c2b7", "#8a8980", "#3a3936"
BLUE, ORANGE, AQUA, VIOLET, RED = "#3987e5", "#d95926", "#199e70", "#9085e9", "#e66767"
DIGIT_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767",
                "#5fb8c9", "#b7a46a"]
INK = LinearSegmentedColormap.from_list("ink", [BG, "#9ec5f5"])
BLUES = LinearSegmentedColormap.from_list("b", [CARD, "#1f3b5c", BLUE])

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


def main():
    Xtr_raw, ytr_all, Xte_raw, yte = load_mnist(ROOT / "data")
    Xall = (Xtr_raw.astype(np.float32) / 255.0)[:, None]
    Xte = (Xte_raw.astype(np.float32) / 255.0)[:, None]
    Xtr, ytr, Xval, yval = Xall[:55000], ytr_all[:55000], Xall[55000:], ytr_all[55000:]
    print(f"train {len(Xtr)}, validation {len(Xval)}, test {len(Xte)}")

    rng = np.random.default_rng(42)
    models = {
        "Softmax regression": (Sequential([Flatten(), Dense(784, 10, rng)]), 5, 128, 1e-3),
        "MLP 784-256-128-10": (Sequential([Flatten(), Dense(784, 256, rng), ReLU(), Dropout(0.2, rng),
                                           Dense(256, 128, rng), ReLU(), Dense(128, 10, rng)]), 12, 128, 1e-3),
        "CNN 2 conv + 2 dense": (Sequential([Conv2D(1, 8, 3, rng), ReLU(), MaxPool2(),
                                             Conv2D(8, 16, 4, rng), ReLU(), MaxPool2(), Flatten(),
                                             Dense(400, 128, rng), ReLU(), Dropout(0.25, rng), Dense(128, 10, rng)]), 10, 64, 1e-3),
    }

    # gradient check on the CNN before training (inputs with small noise so no ReLU sits exactly at its kink)
    gc_rng = np.random.default_rng(0)
    probe = Sequential([Conv2D(1, 8, 3, gc_rng), ReLU(), MaxPool2(), Conv2D(8, 16, 4, gc_rng), ReLU(), MaxPool2(),
                        Flatten(), Dense(400, 128, gc_rng), ReLU(), Dense(128, 10, gc_rng)])
    xg = Xall[:4].astype(np.float64) + gc_rng.normal(0, 0.05, (4, 1, 28, 28))
    grad_err = gradient_check(probe, xg, ytr_all[:4], n_checks=15)
    print(f"gradient check, worst relative error: {grad_err:.2e}")

    results, histories = {}, {}
    for name, (model, epochs, batch, lr) in models.items():
        print(f"Training {name} ({model.n_params():,} parameters)")
        t = time.time()
        histories[name] = train(model, Xtr, ytr, Xval, yval, epochs, batch, lr)
        secs = time.time() - t
        acc = float((model.predict(Xte) == yte).mean())
        results[name] = {"parameters": model.n_params(), "epochs": epochs, "train_seconds": round(secs, 1),
                         "test_accuracy": acc, "test_errors": int(round((1 - acc) * len(yte)))}
        print(f"  test accuracy {acc:.4f} in {secs:.0f}s")

    cnn = models["CNN 2 conv + 2 dense"][0]
    logits = cnn.logits(Xte)
    pred = logits.argmax(1)
    prob = np.exp(logits - logits.max(1, keepdims=True)); prob /= prob.sum(1, keepdims=True)
    cm = np.zeros((10, 10), int)
    for t_, p_ in zip(yte, pred):
        cm[t_, p_] += 1
    per_digit = (np.diag(cm) / cm.sum(1)).round(4)
    off = cm.copy(); np.fill_diagonal(off, 0)
    worst_pairs = sorted(((int(off[i, j]), i, j) for i in range(10) for j in range(10) if off[i, j] > 0), reverse=True)[:5]

    # ------------------------------------------------------------- chart 1: 3D embedding of the CNN feature space
    n = 3000
    feats = np.concatenate([cnn.forward(Xte[i:i + 1000], train=False, upto=9) for i in range(0, n, 1000)])
    feats = feats - feats.mean(0)
    _, _, Vt = np.linalg.svd(feats, full_matrices=False)
    emb = feats @ Vt[:3].T
    var_expl = None
    fig3d = go.Figure()
    for d in range(10):
        m = yte[:n] == d
        fig3d.add_trace(go.Scatter3d(x=emb[m, 0], y=emb[m, 1], z=emb[m, 2], mode="markers", name=str(d),
                                     marker=dict(size=2.6, color=DIGIT_COLORS[d], opacity=0.85),
                                     customdata=pred[:n][m],
                                     hovertemplate=f"True digit {d}<br>Predicted %{{customdata}}<extra></extra>"))
        c = emb[m].mean(0)
        fig3d.add_trace(go.Scatter3d(x=[c[0]], y=[c[1]], z=[c[2]], mode="text", text=[f"<b>{d}</b>"],
                                     textfont=dict(size=22, color=TEXT), showlegend=False, hoverinfo="skip"))
    axis = dict(backgroundcolor=BG, gridcolor=GRID, zerolinecolor=GRID, color=TEXT2, showticklabels=False)
    fig3d.update_layout(**plotly_layout(
        title=dict(text="How the CNN sees digits: 3,000 test images in its learned feature space (top 3 principal components)", x=0.02),
        scene=dict(xaxis=dict(title="PC 1", **axis), yaxis=dict(title="PC 2", **axis), zaxis=dict(title="PC 3", **axis),
                   camera=dict(eye=dict(x=1.5, y=-1.5, z=0.9))),
        legend=dict(title=dict(text="Digit"), font=dict(color=TEXT2), itemsizing="constant"), height=780))
    fig3d.write_html(OUT / "01_feature_space_3d.html", include_plotlyjs="cdn")
    try:
        fig3d.write_image(OUT / "01_feature_space_3d.png", width=1600, height=900, scale=2)
    except Exception as exc:  # noqa: BLE001
        print(f"  (static 3D export skipped: {exc.__class__.__name__})")

    # ------------------------------------------------------------- chart 2: learning curves + comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw={"width_ratios": [1.5, 1]})
    cols = [MUTED, VIOLET, BLUE]
    for (name, h), c in zip(histories.items(), cols):
        axes[0].plot(h["epoch"], 100 * (1 - np.array(h["val_acc"])), color=c, lw=2.2, marker="o", ms=4, label=name)
    axes[0].set_yscale("log"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Validation error (%, log scale)")
    axes[0].set_yticks([1, 2, 5, 10]); axes[0].set_yticklabels(["1", "2", "5", "10"])
    axes[0].set_title("Validation error during training", loc="left", fontsize=12.5, fontweight="bold", color=TEXT)
    axes[0].legend(frameon=False, labelcolor=TEXT2); axes[0].grid(lw=0.5)
    names = list(results)
    accs = [100 * results[k]["test_accuracy"] for k in names]
    axes[1].barh(names[::-1], accs[::-1], color=cols[::-1], height=0.55)
    for i, (k, a) in enumerate(zip(names[::-1], accs[::-1])):
        axes[1].text(a - 0.3, i, f"{a:.2f}%", va="center", ha="right", color=BG, fontweight="bold", fontsize=11)
        axes[1].text(88.2, i - 0.36, f"{results[k]['parameters']:,} parameters, {results[k]['train_seconds']:.0f}s",
                     color=TEXT2, fontsize=8.5)
    axes[1].set_xlim(88, 100); axes[1].set_xlabel("Test accuracy (%) on 10,000 unseen digits")
    axes[1].tick_params(axis="y", colors=TEXT)
    axes[1].set_title("Test accuracy", loc="left", fontsize=12.5, fontweight="bold", color=TEXT)
    fig.tight_layout()
    fig.savefig(OUT / "02_training_and_accuracy.png", dpi=180); plt.close(fig)

    # ------------------------------------------------------------- chart 3: confusion matrix
    fig, ax = plt.subplots(figsize=(9, 8))
    norm = cm / cm.sum(1, keepdims=True)
    show = np.where(np.eye(10, dtype=bool), np.nan, norm)
    ax.imshow(show, cmap=LinearSegmentedColormap.from_list("e", [CARD, "#5c2a2a", RED]), vmin=0, vmax=max(0.01, np.nanmax(show)))
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j] if cm[i, j] else "", ha="center", va="center", fontsize=9.5,
                    color=TEXT if i != j else AQUA, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(10)); ax.set_yticks(range(10)); ax.tick_params(colors=TEXT)
    ax.set_xlabel("Predicted digit"); ax.set_ylabel("True digit")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"CNN confusion matrix: {cm.trace():,} of 10,000 correct", loc="left", fontsize=13, fontweight="bold", color=TEXT)
    fig.text(0.02, 0.01, "Diagonal (green) is correct. Red cells are errors; darker means a more frequent confusion.",
             color=TEXT2, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "03_confusion_matrix.png", dpi=180); plt.close(fig)

    # ------------------------------------------------------------- chart 4: what the CNN learned
    sample = Xte[:1]
    fmaps = cnn.forward(sample, train=False, upto=2)[0]
    W1 = cnn.layers[0].params["W"][:, 0]
    fig, axes = plt.subplots(3, 8, figsize=(15, 6.4))
    for k in range(8):
        axes[0, k].imshow(W1[k], cmap=LinearSegmentedColormap.from_list("w", [RED, CARD, BLUE]),
                          vmin=-np.abs(W1).max(), vmax=np.abs(W1).max())
        axes[1, k].imshow(fmaps[k], cmap=INK)
        axes[2, k].imshow(sample[0, 0] if k == 0 else np.zeros((28, 28)), cmap=INK)
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    for k in range(1, 8):
        axes[2, k].set_visible(False)
    axes[0, 0].set_ylabel("Filters 3x3", color=TEXT2); axes[1, 0].set_ylabel("Activations", color=TEXT2)
    axes[2, 0].set_ylabel("Input", color=TEXT2)
    fig.suptitle("Inside the first convolutional layer: 8 learned edge detectors and how they respond to a digit",
                 x=0.02, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "04_learned_filters.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 5: hardest mistakes
    wrong = np.where(pred != yte)[0]
    conf = prob[wrong, pred[wrong]]
    worst = wrong[np.argsort(-conf)][:24]
    fig, axes = plt.subplots(3, 8, figsize=(15, 6.6))
    for a, i in zip(axes.ravel(), worst):
        a.imshow(Xte[i, 0], cmap=INK)
        a.set_title(f"true {yte[i]}  pred {pred[i]}\n{100*prob[i, pred[i]]:.0f}% sure", fontsize=9, color=TEXT2)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    fig.suptitle(f"The 24 most confident mistakes out of {len(wrong)} errors: many are ambiguous even for a person",
                 x=0.02, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "05_hardest_mistakes.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- metrics + dashboard
    metrics = {"gradient_check_worst_relative_error": grad_err, "models": results,
               "cnn_accuracy_by_digit": dict(enumerate(per_digit.tolist())),
               "cnn_most_frequent_confusions": [{"count": c, "true": i, "predicted": j} for c, i, j in worst_pairs],
               "histories": histories}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    best = results["CNN 2 conv + 2 dense"]
    kpis = {"Test images": "10,000", "CNN test accuracy": f"{100*best['test_accuracy']:.2f}%",
            "MLP test accuracy": f"{100*results['MLP 784-256-128-10']['test_accuracy']:.2f}%",
            "Gradient check error": f"{grad_err:.0e}"}
    kpi_html = "".join(f'<div class="kpi"><span>{k}</span><b>{v}</b></div>' for k, v in kpis.items())
    sec = lambda t, n_, f: f'<section><h2>{t}</h2><p class="note">{n_}</p>{f.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})}</section>'
    img = lambda t, n_, src: f'<section><h2>{t}</h2><p class="note">{n_}</p><img src="outputs/{src}" alt="{t}"></section>'
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>MNIST Digit Recognition</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>:root{{--bg:{BG};--card:{CARD};--text:{TEXT};--text2:{TEXT2};--muted:{MUTED};--line:{GRID}}}*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font-family:Inter,"Segoe UI",Arial,sans-serif}}
header,.kpis,footer{{max-width:1240px;margin:auto;padding:0 32px}}header{{padding-top:48px}}h1{{font-size:40px;margin:0 0 8px}}
header p{{color:var(--text2);max-width:820px;line-height:1.55;margin:0}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:24px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px}}
.kpi span{{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.8px}}.kpi b{{display:block;font-size:24px;margin-top:6px}}
section{{max-width:1240px;margin:28px auto;padding:20px 24px;background:var(--card);border:1px solid var(--line);border-radius:14px}}
section img{{width:100%;border-radius:8px}}h2{{margin:4px 0 6px;font-size:20px}}.note{{color:var(--text2);margin:0 0 10px;font-size:14px}}
footer{{color:var(--muted);font-size:13px;margin:24px auto 48px}}a{{color:#7fb0f0}}
@media(max-width:640px){{header,.kpis,footer{{padding:0 16px}}section{{margin:16px;padding:12px}}h1{{font-size:30px}}}}</style></head><body>
<header><h1>MNIST Handwritten Digit Recognition</h1><p>A neural network library written from scratch in NumPy (dense and convolutional layers,
backpropagation, Adam) trained on 55,000 handwritten digits and tested on 10,000 it never saw.</p></header><div class="kpis">{kpi_html}</div>
{sec("The CNN feature space in 3D", "Drag to rotate. Each dot is a test image, positioned by what the network learned; the numbers mark each cluster center.", fig3d)}
{img("Training and accuracy", "Three architectures trained with the same library.", "02_training_and_accuracy.png")}
{img("Confusion matrix", "Where the remaining errors are.", "03_confusion_matrix.png")}
{img("Learned filters", "First-layer convolution kernels and their responses.", "04_learned_filters.png")}
{img("Hardest mistakes", "The errors the network was most confident about.", "05_hardest_mistakes.png")}
<footer>Tomás Alarcón &middot; <a href="https://github.com/tomasalarcon10/ml-from-scratch">Source code</a></footer></body></html>"""
    (ROOT / "dashboard.html").write_text(page, encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "histories"}, indent=1))


if __name__ == "__main__":
    main()
