"""
Neural Network from scratch: forward pass and backpropagation by hand, trained on intertwined spirals and on Iris.
Run:  python main.py      (about two minutes on a laptop CPU)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
from sklearn.neural_network import MLPClassifier

from src.mlp import MLP, gradient_check, make_spirals

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

BG, CARD, TEXT, TEXT2, MUTED, GRID = "#1a1a19", "#222220", "#f0efec", "#c3c2b7", "#8a8980", "#3a3936"
BLUE, ORANGE, AQUA, AMBER, PINK, VIOLET, RED = "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9", "#e66767"
CLASS_COLORS = [BLUE, ORANGE, AQUA]
REGION = ListedColormap(["#1c3047", "#46261a", "#123a2d"])

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


def title(ax, text, size=12.5):
    ax.set_title(text, loc="left", fontsize=size, fontweight="bold", color=TEXT)


def stratified_split(y, test_frac, seed):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        k = int(round(test_frac * len(idx)))
        te += list(idx[:k]); tr += list(idx[k:])
    return np.array(tr), np.array(te)


def main():
    # ------------------------------------------------------------- spirals
    X, y = make_spirals(n_per_class=200, k=3, noise=0.25, seed=0)
    pd.DataFrame({"x1": X[:, 0].round(5), "x2": X[:, 1].round(5), "label": y}).to_csv(ROOT / "data" / "spirals.csv", index=False)
    tr, te = stratified_split(y, 0.25, seed=1)
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    print(f"Spirals: {len(X)} points, 3 classes; train {len(tr)}, test {len(te)}")

    grad_err = gradient_check(MLP([2, 24, 3, 3], "tanh", seed=3), Xtr[:60], ytr[:60], lam=1e-3)
    grad_err_relu = gradient_check(MLP([2, 16, 16, 3], "relu", seed=4, init="he"), Xtr[:60] + 0.01, ytr[:60], lam=1e-3)
    print(f"gradient check: tanh {grad_err:.1e}, relu {grad_err_relu:.1e}")

    epochs = 4000
    snaps = (0, 50, 150, 400, 1000, epochs)
    linear = MLP([2, 3], "tanh", seed=0)
    h_lin, _ = linear.fit(Xtr, ytr, epochs=epochs, lr=0.3, lam=1e-4, X_val=Xte, y_val=yte)
    net = MLP([2, 24, 3, 3], "tanh", seed=0)
    h_net, snapshots = net.fit(Xtr, ytr, epochs=epochs, lr=0.3, lam=1e-4, X_val=Xte, y_val=yte, snapshot_at=snaps)
    acc_lin, acc_net = float((linear.predict(Xte) == yte).mean()), float((net.predict(Xte) == yte).mean())
    print(f"Linear softmax test accuracy {acc_lin:.3f}; network 2-24-3-3 test accuracy {acc_net:.3f} ({net.n_params()} parameters)")

    # ------------------------------------------------------------- learning rate sweep
    lr_runs = {}
    for lr in (0.003, 0.03, 0.3, 3.0):
        m = MLP([2, 24, 3, 3], "tanh", seed=0)
        h, _ = m.fit(Xtr, ytr, epochs=2000, lr=lr, lam=1e-4)
        lr_runs[lr] = h
    # ------------------------------------------------------------- vanishing gradients in a deep network
    deep = {}
    for act in ("sigmoid", "tanh", "relu"):
        m = MLP([2] + [16] * 8 + [3], act, seed=2, init="he" if act == "relu" else "xavier")
        h, _ = m.fit(Xtr, ytr, epochs=3000, lr=0.05, lam=1e-4, momentum=0.9, X_val=Xte, y_val=yte, record_grad_norms=True)
        deep[act] = {"hist": h, "grad0": h["grad_norms"][0], "test_acc": float((m.predict(Xte) == yte).mean())}
        print(f"  deep {act}: first/last layer gradient {h['grad_norms'][0][0]:.2e} / {h['grad_norms'][0][-1]:.2e}, test {deep[act]['test_acc']:.3f}")

    # ------------------------------------------------------------- Iris
    iris = pd.read_csv(ROOT / "data" / "iris.csv")
    species = sorted(iris["species"].unique())
    Xi = iris.iloc[:, :4].to_numpy(float); yi = iris["species"].map({s: i for i, s in enumerate(species)}).to_numpy()
    itr, ite = stratified_split(yi, 0.3, seed=7)
    mu, sd = Xi[itr].mean(0), Xi[itr].std(0)
    Zi = (Xi - mu) / sd
    inet = MLP([4, 8, 3], "tanh", seed=0)
    h_iris, _ = inet.fit(Zi[itr], yi[itr], epochs=1500, lr=0.1, lam=1e-3, X_val=Zi[ite], y_val=yi[ite])
    acc_iris = float((inet.predict(Zi[ite]) == yi[ite]).mean())
    sk = MLPClassifier(hidden_layer_sizes=(8,), activation="tanh", alpha=1e-3, max_iter=5000, random_state=0).fit(Zi[itr], yi[itr])
    acc_sk = float(sk.score(Zi[ite], yi[ite]))
    agree = float((sk.predict(Zi[ite]) == inet.predict(Zi[ite])).mean())
    cm = np.zeros((3, 3), int)
    for t_, p_ in zip(yi[ite], inet.predict(Zi[ite])):
        cm[t_, p_] += 1
    print(f"Iris 4-8-3: test accuracy {acc_iris:.3f}; scikit-learn MLP {acc_sk:.3f}; agreement {agree:.3f}")

    # ------------------------------------------------------------- chart 1: 3D hidden space with a slider over training
    order_cls = [(c, np.where(y == c)[0]) for c in range(3)]
    frames = []
    def hidden_at(ws):
        tmp = MLP([2, 24, 3, 3], "tanh"); tmp.W, tmp.b = ws
        return tmp.hidden(X, 2)
    first = hidden_at(snapshots[0])
    fig3d = go.Figure()
    for c, idx in order_cls:
        fig3d.add_trace(go.Scatter3d(x=first[idx, 0], y=first[idx, 1], z=first[idx, 2], mode="markers", name=f"Spiral {c + 1}",
                                     marker=dict(size=3, color=CLASS_COLORS[c], opacity=0.85),
                                     hovertemplate=f"Spiral {c + 1}<br>h1 %{{x:.2f}}, h2 %{{y:.2f}}, h3 %{{z:.2f}}<extra></extra>"))
    for ep in snaps:
        H = hidden_at(snapshots[ep])
        frames.append(go.Frame(name=str(ep), data=[go.Scatter3d(x=H[idx, 0], y=H[idx, 1], z=H[idx, 2]) for _, idx in order_cls]))
    fig3d.frames = frames
    axis = dict(backgroundcolor=BG, gridcolor=GRID, zerolinecolor=GRID, color=TEXT2, range=[-1.05, 1.05])
    fig3d.update_layout(**plotly_layout(
        title=dict(text="Inside the network: the 3 neurons of the last hidden layer pull the intertwined spirals apart", x=0.02),
        scene=dict(xaxis=dict(title="neuron 1", **axis), yaxis=dict(title="neuron 2", **axis), zaxis=dict(title="neuron 3", **axis),
                   camera=dict(eye=dict(x=1.5, y=-1.5, z=0.9))),
        legend=dict(font=dict(color=TEXT2), x=0.01, y=0.92, itemsizing="constant"), height=780,
        updatemenus=[dict(type="buttons", x=0.02, y=0.06, xanchor="left", showactive=False, bgcolor=CARD, font=dict(color=TEXT),
                          buttons=[dict(label="Play training", method="animate",
                                        args=[None, dict(frame=dict(duration=900, redraw=True), transition=dict(duration=500), fromcurrent=True)])])],
        sliders=[dict(active=0, x=0.15, y=0.06, len=0.8, currentvalue=dict(prefix="Epoch ", font=dict(color=TEXT2)), font=dict(color=TEXT2),
                      steps=[dict(label=str(ep), method="animate", args=[[str(ep)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                             for ep in snaps])]))
    fig3d.write_html(OUT / "01_hidden_space_3d.html", include_plotlyjs="cdn")
    final_fig = go.Figure(data=list(fig3d.data), layout=fig3d.layout)
    final_fig.layout.sliders = []
    final_fig.layout.updatemenus = []
    final_fig.update_layout(title=dict(text="Inside the trained network: the 3 neurons of the last hidden layer pull the intertwined spirals apart"))
    Hf = hidden_at(snapshots[epochs])
    for i, (_, idx) in enumerate(order_cls):
        final_fig.data[i].update(x=Hf[idx, 0], y=Hf[idx, 1], z=Hf[idx, 2])
    try:
        final_fig.write_image(OUT / "01_hidden_space_3d.png", width=1600, height=900, scale=2)
    except Exception as exc:  # noqa: BLE001
        print(f"  (static 3D export skipped: {exc.__class__.__name__})")

    # ------------------------------------------------------------- chart 2: decision boundary while learning
    g = np.linspace(-1.25, 1.25, 300)
    gx, gy = np.meshgrid(g, g)
    grid = np.c_[gx.ravel(), gy.ravel()]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.4))
    panels = [("Linear model (no hidden layer)", linear, None)] + [(f"Epoch {ep}", None, ep) for ep in snaps] + [("Test points, final network", net, "test")]
    loss_at = dict(zip(h_net["epoch"], h_net["train_loss"]))
    for ax, (lab, model, ep) in zip(axes.ravel(), panels):
        if model is None:
            model = MLP([2, 24, 3, 3], "tanh"); model.W, model.b = snapshots[ep]
        ax.imshow(model.predict(grid).reshape(gx.shape), origin="lower", extent=(-1.25, 1.25, -1.25, 1.25), cmap=REGION, vmin=0, vmax=2)
        pts, lbl = (Xte, yte) if ep == "test" else (Xtr, ytr)
        for c in range(3):
            m = lbl == c
            ax.scatter(pts[m, 0], pts[m, 1], s=9 if ep != "test" else 22, color=CLASS_COLORS[c], edgecolor=BG, lw=0.4)
        if ep == "test":
            wrong = model.predict(Xte) != yte
            ax.scatter(Xte[wrong, 0], Xte[wrong, 1], s=90, facecolors="none", edgecolors=TEXT, lw=1.4)
            sub = f"accuracy {acc_net:.1%}"
        elif ep is None:
            sub = f"test accuracy {acc_lin:.1%}"
        else:
            sub = f"loss {loss_at.get(ep, np.nan):.3f}"
        title(ax, f"{lab}: {sub}", 11)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle("Learning to separate three spirals: the decision regions of the 2-24-3-3 network during training",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "02_decision_boundary_evolution.png", dpi=160); plt.close(fig)

    # ------------------------------------------------------------- chart 3: loss curves
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.4))
    axes[0].semilogy(h_net["epoch"], h_net["train_loss"], color=BLUE, lw=2.2, label="Network, training")
    axes[0].semilogy(h_net["epoch"], h_net["val_loss"], color=AMBER, lw=2.2, label="Network, test")
    axes[0].semilogy(h_lin["epoch"], h_lin["train_loss"], color=MUTED, lw=1.6, ls="--", label="Linear model, training")
    axes[0].set_xlabel("Epoch (full-batch gradient descent with momentum)"); axes[0].set_ylabel("Cross-entropy loss (log scale)")
    axes[0].legend(frameon=False, labelcolor=TEXT2); axes[0].grid(lw=0.5)
    title(axes[0], "Loss curve")
    axes[1].plot(h_net["epoch"], 100 * np.array(h_net["train_acc"]), color=BLUE, lw=2.2, label="Network, training")
    axes[1].plot(h_net["epoch"], 100 * np.array(h_net["val_acc"]), color=AMBER, lw=2.2, label="Network, test")
    axes[1].plot(h_lin["epoch"], 100 * np.array(h_lin["val_acc"]), color=MUTED, lw=1.6, ls="--", label="Linear model, test")
    axes[1].axhline(100 / 3, color=GRID, lw=1, ls=":")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)"); axes[1].set_ylim(20, 102)
    axes[1].legend(frameon=False, labelcolor=TEXT2, loc="lower right"); axes[1].grid(lw=0.5)
    title(axes[1], f"Accuracy: {acc_net:.1%} for the network vs {acc_lin:.1%} for a linear model")
    fig.tight_layout()
    fig.savefig(OUT / "03_loss_curves.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 4: learning rate and vanishing gradients
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    for (lr, h), col in zip(lr_runs.items(), [MUTED, VIOLET, BLUE, RED]):
        ep_, ls_ = np.array(h["epoch"]), np.array(h["train_loss"], float)
        axes[0].semilogy(ep_, np.where(np.isfinite(ls_), ls_, np.nan), color=col, lw=2, label=f"learning rate {lr:g}")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training loss (log scale)"); axes[0].legend(frameon=False, labelcolor=TEXT2); axes[0].grid(lw=0.5)
    title(axes[0], "Learning rate: too small crawls, too large diverges")
    for (act, d), col in zip(deep.items(), [PINK, AMBER, AQUA]):
        axes[1].semilogy(range(1, 10), d["grad0"], color=col, lw=2.2, marker="o", label=act)
    axes[1].set_xlabel("Layer (1 = closest to the input)"); axes[1].set_ylabel("Gradient norm at initialization (log scale)")
    axes[1].legend(frameon=False, labelcolor=TEXT2); axes[1].grid(lw=0.5); axes[1].set_xticks(range(1, 10))
    title(axes[1], "Vanishing gradients in an 8-hidden-layer network")
    for (act, d), col in zip(deep.items(), [PINK, AMBER, AQUA]):
        axes[2].semilogy(d["hist"]["epoch"], d["hist"]["train_loss"], color=col, lw=2.2, label=f"{act} (test {d['test_acc']:.0%})")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Training loss (log scale)"); axes[2].legend(frameon=False, labelcolor=TEXT2); axes[2].grid(lw=0.5)
    title(axes[2], "So the deep sigmoid network barely learns")
    fig.tight_layout()
    fig.savefig(OUT / "04_learning_rate_and_vanishing_gradients.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 5: the Iris network itself
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.2), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    layers = [4, 8, 3]
    ypos = [np.linspace(0.85, 0.15, n) if n > 1 else [0.5] for n in layers]
    xpos = [0.12, 0.5, 0.88]
    wmax = max(np.abs(W).max() for W in inet.W)
    for l, W in enumerate(inet.W):
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                w = W[i, j]
                ax.plot([xpos[l], xpos[l + 1]], [ypos[l][i], ypos[l + 1][j]], color=BLUE if w > 0 else ORANGE,
                        lw=0.4 + 3.6 * abs(w) / wmax, alpha=0.25 + 0.7 * abs(w) / wmax, zorder=1)
    names_in = ["sepal length", "sepal width", "petal length", "petal width"]
    for l, n in enumerate(layers):
        for i in range(n):
            ax.scatter(xpos[l], ypos[l][i], s=520, color=CARD, edgecolor=TEXT2 if l != 2 else CLASS_COLORS[i], lw=1.8, zorder=3)
            if l == 0:
                ax.text(xpos[l] - 0.05, ypos[l][i], names_in[i], ha="right", va="center", color=TEXT, fontsize=10)
            if l == 2:
                ax.text(xpos[l] + 0.05, ypos[l][i], species[i], ha="left", va="center", color=CLASS_COLORS[i], fontsize=11, fontweight="bold")
    ax.text(xpos[1], 0.03, "8 tanh neurons", ha="center", color=TEXT2)
    ax.set_xlim(-0.12, 1.12); ax.set_ylim(0, 1); ax.axis("off")
    title(ax, "The trained Iris network 4-8-3: line width = weight size (blue positive, orange negative)")
    axes[1].plot(h_iris["epoch"], h_iris["train_loss"], color=BLUE, lw=2.2, label="Training")
    axes[1].plot(h_iris["epoch"], h_iris["val_loss"], color=AMBER, lw=2.2, label="Test")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Cross-entropy loss"); axes[1].legend(frameon=False, labelcolor=TEXT2); axes[1].grid(lw=0.5)
    axes[1].text(0.97, 0.55, f"Test accuracy {acc_iris:.1%}\n({cm.trace()} of {cm.sum()} flowers)\nscikit-learn MLP: {acc_sk:.1%}",
                 transform=axes[1].transAxes, ha="right", color=TEXT, fontsize=11)
    title(axes[1], "Iris loss curve")
    fig.tight_layout()
    fig.savefig(OUT / "05_iris_network.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- metrics
    metrics = {
        "gradient_check": {"tanh_network": grad_err, "relu_network": grad_err_relu},
        "spirals": {"points": len(X), "train": len(tr), "test": len(te), "architecture": [2, 24, 3, 3], "parameters": net.n_params(),
                    "epochs": epochs, "learning_rate": 0.3, "momentum": 0.9, "l2": 1e-4,
                    "test_accuracy_network": acc_net, "test_accuracy_linear": acc_lin,
                    "final_train_loss": h_net["train_loss"][-1], "final_test_loss": h_net["val_loss"][-1]},
        "learning_rate_sweep_final_loss": {str(lr): (h["train_loss"][-1] if np.isfinite(h["train_loss"][-1]) else "diverged") for lr, h in lr_runs.items()},
        "deep_network_8x16": {act: {"gradient_norm_layer1": d["grad0"][0], "gradient_norm_last_hidden": d["grad0"][-2],
                                    "final_train_loss": d["hist"]["train_loss"][-1], "test_accuracy": d["test_acc"]} for act, d in deep.items()},
        "iris": {"architecture": [4, 8, 3], "test_accuracy": acc_iris, "sklearn_mlp_test_accuracy": acc_sk, "prediction_agreement": agree,
                 "confusion_matrix": cm.tolist(), "test_size": int(len(ite))},
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(json.dumps({k: metrics[k] for k in ("spirals", "learning_rate_sweep_final_loss", "deep_network_8x16")}, indent=1, default=float))

    # ------------------------------------------------------------- dashboard with a live playground
    (OUT / "spirals.js").write_text("window.SPIRALS=" + json.dumps({"X": X.round(4).tolist(), "y": y.tolist(), "train": tr.tolist()}) + ";")
    kpis = {"Spirals test accuracy": f"{100 * acc_net:.1f}%", "Linear model": f"{100 * acc_lin:.1f}%",
            "Iris test accuracy": f"{100 * acc_iris:.1f}%", "Gradient check": f"{max(grad_err, grad_err_relu):.0e}"}
    kpi_html = "".join(f'<div class="kpi"><span>{k}</span><b>{v}</b></div>' for k, v in kpis.items())
    sec = lambda t, n_, f: f'<section><h2>{t}</h2><p class="note">{n_}</p>{f.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False}, auto_play=False)}</section>'
    img = lambda t, n_, src: f'<section><h2>{t}</h2><p class="note">{n_}</p><img src="outputs/{src}" alt="{t}"></section>'
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Neural Network from Scratch</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><script src="outputs/spirals.js"></script>
<style>:root{{--bg:{BG};--card:{CARD};--text:{TEXT};--text2:{TEXT2};--muted:{MUTED};--line:{GRID}}}*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font-family:Inter,"Segoe UI",Arial,sans-serif}}
header,.kpis,footer{{max-width:1240px;margin:auto;padding:0 32px}}header{{padding-top:48px}}h1{{font-size:40px;margin:0 0 8px}}
header p{{color:var(--text2);max-width:860px;line-height:1.55;margin:0}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-top:24px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px}}
.kpi span{{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.8px}}.kpi b{{display:block;font-size:24px;margin-top:6px}}
section{{max-width:1240px;margin:28px auto;padding:20px 24px;background:var(--card);border:1px solid var(--line);border-radius:14px}}
section img{{width:100%;border-radius:8px}}h2{{margin:4px 0 6px;font-size:20px}}.note{{color:var(--text2);margin:0 0 10px;font-size:14px}}
.play{{display:grid;grid-template-columns:300px 1fr;gap:24px;align-items:start}}.ctl label{{display:block;color:var(--text2);font-size:13px;margin:12px 0 4px}}
.ctl input[type=range]{{width:100%;accent-color:{BLUE}}}.ctl select{{width:100%;background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:6px}}
.btns{{display:flex;gap:8px;margin-top:16px}}button{{flex:1;border:0;border-radius:8px;padding:9px;font:inherit;font-weight:600;cursor:pointer;background:{BLUE};color:white}}
button.sec{{background:var(--bg);color:var(--text2);border:1px solid var(--line)}}.viz{{display:grid;grid-template-columns:auto 1fr;gap:20px}}
canvas{{border-radius:10px;border:1px solid var(--line);max-width:100%}}.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:10px}}
.stats div{{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:12px;color:var(--muted)}}.stats b{{display:block;color:var(--text);font-size:18px}}
footer{{color:var(--muted);font-size:13px;margin:24px auto 48px}}a{{color:#7fb0f0}}
@media(max-width:900px){{.play,.viz{{grid-template-columns:1fr}}header,.kpis,footer{{padding:0 16px}}section{{margin:16px;padding:12px}}h1{{font-size:30px}}}}</style></head><body>
<header><h1>Neural Network from Scratch</h1><p>A multilayer perceptron with the forward pass and backpropagation written equation by equation in NumPy.
It learns to separate three intertwined spirals that no straight line can split, and classifies Iris flowers.</p></header>
<div class="kpis">{kpi_html}</div>
<section><h2>Train a network yourself</h2><p class="note">The same equations, rewritten in JavaScript, train a 2-H-3 network on the training spirals in your browser. Colors are the regions the network assigns to each spiral; the ringed points are the test set.</p>
<div class="play"><div class="ctl"><label>Hidden neurons: <b id="h_o"></b></label><input type="range" id="h" min="2" max="40" value="20">
<label>Activation</label><select id="act"><option>tanh</option><option>relu</option><option>sigmoid</option></select>
<label>Learning rate: <b id="lr_o"></b></label><input type="range" id="lr" min="-3" max="0.7" step="0.1" value="-0.5">
<div class="btns"><button id="run">Train</button><button class="sec" id="reset">Reset</button></div>
<div class="stats"><div>Epoch<b id="ep">0</b></div><div>Loss<b id="ls">-</b></div><div>Test acc.<b id="ac">-</b></div></div></div>
<div class="viz"><canvas id="cv" width="420" height="420"></canvas><canvas id="lc" width="420" height="420"></canvas></div></div></section>
{sec("Inside the network, in 3D", "Press Play or move the slider: the three neurons of the last hidden layer start as a tangle and end with each spiral in its own corner, where a straight cut can separate them.", fig3d)}
{img("Decision boundary during training", "From a linear model to the trained network.", "02_decision_boundary_evolution.png")}
{img("Loss curve", "Training and test loss of the network trained by hand.", "03_loss_curves.png")}
{img("Learning rate and vanishing gradients", "Two classic ways training fails.", "04_learning_rate_and_vanishing_gradients.png")}
{img("Iris", "The same code on a real dataset.", "05_iris_network.png")}
<footer>Tomás Alarcón &middot; <a href="https://github.com/tomasalarcon10/ml-from-scratch">Source code</a></footer>
<script>
const D=window.SPIRALS,trainSet=new Set(D.train),XT=[],YT=[],XV=[],YV=[];D.X.forEach((x,i)=>{{(trainSet.has(i)?XT:XV).push(x);(trainSet.has(i)?YT:YV).push(D.y[i])}});
const COLS=["{BLUE}","{ORANGE}","{AQUA}"],REG=[[28,48,71],[70,38,26],[18,58,45]];let net,ep,running=false,losses=[];
const act={{tanh:[Math.tanh,(z,a)=>1-a*a],relu:[z=>Math.max(0,z),(z,a)=>z>0?1:0],sigmoid:[z=>1/(1+Math.exp(-z)),(z,a)=>a*(1-a)]}};
function randn(){{let u=0,v=0;while(!u)u=Math.random();while(!v)v=Math.random();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)}}
function init(){{const H=+document.getElementById("h").value,a=document.getElementById("act").value,s=a==="relu"?Math.sqrt(2/2):Math.sqrt(1/2),s2=a==="relu"?Math.sqrt(2/H):Math.sqrt(1/H);
net={{H,a,W1:[...Array(2)].map(()=>[...Array(H)].map(()=>randn()*s)),b1:Array(H).fill(0),W2:[...Array(H)].map(()=>[...Array(3)].map(()=>randn()*s2)),b2:[0,0,0],
v:{{W1:[...Array(2)].map(()=>Array(H).fill(0)),b1:Array(H).fill(0),W2:[...Array(H)].map(()=>Array(3).fill(0)),b2:[0,0,0]}}}};ep=0;losses=[];draw()}}
function fwd(x){{const g=act[net.a][0],z=[],h=[];for(let j=0;j<net.H;j++){{const v=x[0]*net.W1[0][j]+x[1]*net.W1[1][j]+net.b1[j];z.push(v);h.push(g(v))}}
const o=[0,1,2].map(k=>{{let s=net.b2[k];for(let j=0;j<net.H;j++)s+=h[j]*net.W2[j][k];return s}});const m=Math.max(...o),e=o.map(v=>Math.exp(v-m)),t=e[0]+e[1]+e[2];return {{z,h,p:e.map(v=>v/t)}}}}
function step(){{const lr=Math.pow(10,+document.getElementById("lr").value),n=XT.length,H=net.H,dg=act[net.a][1],gW1=[Array(H).fill(0),Array(H).fill(0)],gb1=Array(H).fill(0),gW2=[...Array(H)].map(()=>[0,0,0]),gb2=[0,0,0];let L=0;
for(let i=0;i<n;i++){{const x=XT[i],{{z,h,p}}=fwd(x);L-=Math.log(p[YT[i]]+1e-12);const dz=p.map((v,k)=>(v-(k===YT[i]?1:0))/n);
for(let k=0;k<3;k++){{gb2[k]+=dz[k];for(let j=0;j<H;j++)gW2[j][k]+=h[j]*dz[k]}}
for(let j=0;j<H;j++){{const dh=dz[0]*net.W2[j][0]+dz[1]*net.W2[j][1]+dz[2]*net.W2[j][2],d=dh*dg(z[j],h[j]);gb1[j]+=d;gW1[0][j]+=x[0]*d;gW1[1][j]+=x[1]*d}}}}
const upd=(P,G,V)=>{{for(let i=0;i<P.length;i++){{if(Array.isArray(P[i]))upd(P[i],G[i],V[i]);else{{V[i]=0.9*V[i]-lr*G[i];P[i]+=V[i]}}}}}};
upd(net.W1,gW1,net.v.W1);upd(net.b1,gb1,net.v.b1);upd(net.W2,gW2,net.v.W2);upd(net.b2,gb2,net.v.b2);ep++;return L/n}}
const cv=document.getElementById("cv"),cx=cv.getContext("2d"),lc=document.getElementById("lc"),lx=lc.getContext("2d"),R=1.25,S=420;
function draw(){{const N=84,cell=S/N,img=cx.createImageData(S,S);for(let a=0;a<N;a++)for(let b=0;b<N;b++){{const x=-R+2*R*(b+0.5)/N,y=R-2*R*(a+0.5)/N,p=fwd([x,y]).p,k=p.indexOf(Math.max(...p)),c=REG[k],sh=0.55+0.45*p[k];
for(let u=Math.floor(a*cell);u<Math.floor((a+1)*cell);u++)for(let w=Math.floor(b*cell);w<Math.floor((b+1)*cell);w++){{const o=4*(u*S+w);img.data[o]=c[0]*sh+10;img.data[o+1]=c[1]*sh+10;img.data[o+2]=c[2]*sh+10;img.data[o+3]=255}}}}
cx.putImageData(img,0,0);const P=(x)=>[(x[0]+R)/(2*R)*S,(R-x[1])/(2*R)*S];
XT.forEach((x,i)=>{{const [u,v]=P(x);cx.fillStyle=COLS[YT[i]];cx.beginPath();cx.arc(u,v,2.6,0,7);cx.fill()}});
let ok=0;XV.forEach((x,i)=>{{const [u,v]=P(x),p=fwd(x).p,k=p.indexOf(Math.max(...p));if(k===YV[i])ok++;cx.strokeStyle=k===YV[i]?"#f0efec":"#e66767";cx.lineWidth=1.6;cx.fillStyle=COLS[YV[i]];cx.beginPath();cx.arc(u,v,4,0,7);cx.fill();cx.stroke()}});
document.getElementById("ep").textContent=ep;document.getElementById("ls").textContent=losses.length?losses[losses.length-1].toFixed(3):"-";document.getElementById("ac").textContent=(100*ok/XV.length).toFixed(1)+"%";
lx.fillStyle="{BG}";lx.fillRect(0,0,S,S);lx.strokeStyle="{GRID}";lx.lineWidth=1;for(let i=1;i<5;i++){{lx.beginPath();lx.moveTo(40,i*S/5);lx.lineTo(S-10,i*S/5);lx.stroke()}}
lx.fillStyle="{MUTED}";lx.font="12px Inter, Arial";lx.fillText("training loss (log scale) vs epoch",48,20);
if(losses.length>1){{const lo=Math.log10(Math.max(1e-3,Math.min(...losses))),hi=Math.log10(Math.max(...losses));lx.strokeStyle="{BLUE}";lx.lineWidth=2.4;lx.beginPath();
losses.forEach((v,i)=>{{const x=40+(S-55)*i/(losses.length-1),y=30+(S-60)*(hi-Math.log10(Math.max(v,1e-3)))/Math.max(1e-9,hi-lo);i?lx.lineTo(x,y):lx.moveTo(x,y)}});lx.stroke()}}}}
function loop(){{if(!running)return;for(let i=0;i<10;i++){{const L=step();if(i===9)losses.push(L);if(!isFinite(L)){{running=false;document.getElementById("run").textContent="Diverged";break}}}}draw();if(ep<6000)requestAnimationFrame(loop);else{{running=false;document.getElementById("run").textContent="Train"}}}}
document.getElementById("run").onclick=e=>{{running=!running;e.target.textContent=running?"Pause":"Train";if(running)loop()}};
document.getElementById("reset").onclick=()=>{{running=false;document.getElementById("run").textContent="Train";init()}};
const lab=()=>{{document.getElementById("h_o").textContent=document.getElementById("h").value;document.getElementById("lr_o").textContent=Math.pow(10,+document.getElementById("lr").value).toPrecision(2)}};
["h","act"].forEach(id=>document.getElementById(id).onchange=()=>{{lab();running=false;document.getElementById("run").textContent="Train";init()}});document.getElementById("lr").oninput=lab;lab();init();
</script></body></html>"""
    (ROOT / "dashboard.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
