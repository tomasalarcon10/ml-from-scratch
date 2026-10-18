"""
Iris Flower Classification
Run:  python main.py
k-NN implemented from scratch, decision trees, repeated stratified cross-validation,
decision boundaries, 3D species clusters and an interactive dashboard.
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

from src.knn import (KNNClassifier, StandardScaler, classification_metrics, stratified_kfold,
                     cross_val_accuracy)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

BG, CARD, TEXT, TEXT2, MUTED, GRID = "#1a1a19", "#222220", "#f0efec", "#c3c2b7", "#8a8980", "#3a3936"
SPECIES = ["setosa", "versicolor", "virginica"]
COLORS = {"setosa": "#3987e5", "versicolor": "#d95926", "virginica": "#199e70"}
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
LABEL = {f: f.replace("_", " ").capitalize() + " (cm)" for f in FEATURES}

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


class MajorityBaseline:
    def fit(self, X, y):
        vals, counts = np.unique(y, return_counts=True)
        self.c_ = vals[np.argmax(counts)]
        return self

    def predict(self, X):
        return np.full(len(X), self.c_)


def out_of_fold(make_model, X, y, scale=True):
    pred = np.empty(len(y), dtype=object)
    for tr, te in stratified_kfold(y, 5, seed=0):
        Xtr, Xte = X[tr], X[te]
        if scale:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        pred[te] = make_model().fit(Xtr, y[tr]).predict(Xte)
    return pred.astype(str)


def main():
    df = pd.read_csv(ROOT / "data" / "iris.csv")
    X, y = df[FEATURES].values, df["species"].values

    # ------------------------------------------------------------- choose k and depth
    ks = list(range(1, 32, 2))
    knn_curve = [cross_val_accuracy(lambda k=k: KNNClassifier(k), X, y) for k in ks]
    sk_curve = [cross_val_accuracy(lambda k=k: KNeighborsClassifier(k), X, y)[0] for k in ks]
    best_k = ks[int(np.argmax([m for m, _ in knn_curve]))]
    depths = [1, 2, 3, 4, 5, None]
    tree_curve = [cross_val_accuracy(lambda d=d: DecisionTreeClassifier(max_depth=d, random_state=0), X, y, scale=False)
                  for d in depths]
    best_depth = 3  # simplest tree within one point of the best score; deeper trees do not generalize better

    results = {
        "Majority class baseline": cross_val_accuracy(MajorityBaseline, X, y, scale=False),
        "k-NN, k=1 (scratch)": cross_val_accuracy(lambda: KNNClassifier(1), X, y),
        f"k-NN, k={best_k} (scratch)": cross_val_accuracy(lambda: KNNClassifier(best_k), X, y),
        f"scikit-learn k-NN, k={best_k}": cross_val_accuracy(lambda: KNeighborsClassifier(best_k), X, y),
        "k-NN on petals only, k=" + str(best_k): cross_val_accuracy(lambda: KNNClassifier(best_k), X[:, 2:], y),
        "Decision tree, depth 2": tree_curve[1],
        "Decision tree, depth 3": tree_curve[2],
        "Decision tree, unlimited": tree_curve[-1],
    }
    for k_, v in results.items():
        print(f"  {k_:34s} {100*v[0]:.1f}% +/- {100*v[1]:.1f}")

    oof_knn = out_of_fold(lambda: KNNClassifier(best_k), X, y)
    oof_tree = out_of_fold(lambda: DecisionTreeClassifier(max_depth=best_depth, random_state=0), X, y, scale=False)
    m_knn = classification_metrics(y, oof_knn, SPECIES)
    m_tree = classification_metrics(y, oof_tree, SPECIES)

    sc = StandardScaler().fit(X)
    agree = float((KNNClassifier(best_k).fit(sc.transform(X), y).predict(sc.transform(X))
                   == KNeighborsClassifier(best_k).fit(sc.transform(X), y).predict(sc.transform(X))).mean())

    # ------------------------------------------------------------- chart 1: 3D clusters (interactive)
    f3 = ["petal_length", "petal_width", "sepal_length"]
    fig3d = go.Figure()
    for s in SPECIES:
        d = df[df["species"] == s]
        fig3d.add_trace(go.Mesh3d(x=d[f3[0]], y=d[f3[1]], z=d[f3[2]], alphahull=0, opacity=0.13,
                                  color=COLORS[s], hoverinfo="skip", showlegend=False))
        fig3d.add_trace(go.Scatter3d(x=d[f3[0]], y=d[f3[1]], z=d[f3[2]], mode="markers", name=s.capitalize(),
                                     marker=dict(size=4.5, color=COLORS[s], line=dict(width=0.6, color=BG)),
                                     hovertemplate=f"<b>{s}</b><br>Petal %{{x}} x %{{y}} cm<br>Sepal length %{{z}} cm<extra></extra>"))
    wrong = df[oof_knn != y]
    fig3d.add_trace(go.Scatter3d(x=wrong[f3[0]], y=wrong[f3[1]], z=wrong[f3[2]], mode="markers",
                                 name="Misclassified by k-NN", marker=dict(size=9, color="rgba(0,0,0,0)",
                                                                          line=dict(width=3, color=TEXT)),
                                 hovertemplate="Misclassified<extra></extra>"))
    axis = dict(backgroundcolor=BG, gridcolor=GRID, zerolinecolor=GRID, color=TEXT2)
    fig3d.update_layout(**plotly_layout(
        title=dict(text="The three species in measurement space (convex hulls per species)", x=0.02),
        scene=dict(xaxis=dict(title="Petal length (cm)", **axis), yaxis=dict(title="Petal width (cm)", **axis),
                   zaxis=dict(title="Sepal length (cm)", **axis), camera=dict(eye=dict(x=1.6, y=-1.5, z=0.8))),
        legend=dict(font=dict(color=TEXT2), x=0.02, y=0.95), height=760))
    fig3d.write_html(OUT / "01_species_3d.html", include_plotlyjs="cdn")
    try:
        fig3d.write_image(OUT / "01_species_3d.png", width=1600, height=900, scale=2)
    except Exception as exc:  # noqa: BLE001
        print(f"  (static 3D export skipped: {exc.__class__.__name__})")

    # ------------------------------------------------------------- chart 2: decision boundaries
    Xp = df[["petal_length", "petal_width"]].values
    xx, yy = np.meshgrid(np.linspace(0.5, 7.3, 300), np.linspace(-0.1, 2.8, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    code = {s: i for i, s in enumerate(SPECIES)}
    region_cmap = ListedColormap(["#1d3350", "#4a2a1c", "#173a2f"])
    sc2 = StandardScaler().fit(Xp)
    panels = [
        ("k-NN, k=1: memorizes every point", KNNClassifier(1).fit(sc2.transform(Xp), y), True),
        (f"k-NN, k={best_k}: smoother, more robust", KNNClassifier(best_k).fit(sc2.transform(Xp), y), True),
        ("Decision tree, depth 3: axis-aligned rules", DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xp, y), False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), sharey=True)
    for ax, (title, model, scaled) in zip(axes, panels):
        z = model.predict(sc2.transform(grid) if scaled else grid)
        ax.contourf(xx, yy, np.vectorize(code.get)(z).reshape(xx.shape), levels=[-0.5, 0.5, 1.5, 2.5],
                    cmap=region_cmap, alpha=1.0)
        for s in SPECIES:
            d = df[df["species"] == s]
            ax.scatter(d["petal_length"], d["petal_width"], s=26, color=COLORS[s], edgecolor=BG, linewidth=0.6,
                       label=s.capitalize())
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=TEXT)
        ax.set_xlabel("Petal length (cm)")
    axes[0].set_ylabel("Petal width (cm)")
    axes[0].legend(frameon=False, labelcolor=TEXT2, loc="upper left")
    fig.suptitle("How each model splits the flower space", x=0.02, ha="left", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "02_decision_boundaries.png", dpi=180); plt.close(fig)

    # ------------------------------------------------------------- chart 3: choosing k and depth
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw={"width_ratios": [1.6, 1]})
    m = np.array([a for a, _ in knn_curve]) * 100; s_ = np.array([b for _, b in knn_curve]) * 100
    axes[0].fill_between(ks, m - s_, m + s_, color="#3987e5", alpha=0.15, lw=0)
    axes[0].plot(ks, m, color="#3987e5", lw=2.2, marker="o", ms=5, label="k-NN from scratch")
    axes[0].plot(ks, np.array(sk_curve) * 100, color="#9085e9", lw=0, marker="x", ms=8, label="scikit-learn (same folds)")
    axes[0].axvline(best_k, color=TEXT2, ls="--", lw=1)
    axes[0].text(best_k + 0.6, m.min() - 2, f"best k = {best_k}", color=TEXT2, fontsize=9.5)
    axes[0].set_xlabel("Number of neighbors k"); axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("k-NN: accuracy by number of neighbors", loc="left", fontsize=12.5, fontweight="bold", color=TEXT)
    axes[0].legend(frameon=False, labelcolor=TEXT2); axes[0].grid(lw=0.5)
    labels_d = ["1", "2", "3", "4", "5", "None"]
    tm = [100 * a for a, _ in tree_curve]
    axes[1].bar(labels_d, tm, color=["#5c5b56" if d != 3 else "#d95926" for d in depths], width=0.6)
    for i, v in enumerate(tm):
        axes[1].text(i, v + 1, f"{v:.1f}", ha="center", color=TEXT2, fontsize=9.5)
    axes[1].set_ylim(60, 102); axes[1].set_xlabel("Maximum tree depth"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Decision tree: accuracy by depth", loc="left", fontsize=12.5, fontweight="bold", color=TEXT)
    axes[1].set_axisbelow(True); axes[1].grid(axis="y", lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "03_model_selection.png", dpi=180); plt.close(fig)

    # ------------------------------------------------------------- chart 4: confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, (title, met) in zip(axes, [(f"k-NN, k={best_k}", m_knn), ("Decision tree, depth 3", m_tree)]):
        cm = met["confusion"]
        ax.imshow(cm, cmap=ListedColormap(["#222220", "#1f3b5c", "#2a5a96", "#3987e5"]), vmin=0, vmax=50)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=15, fontweight="bold",
                        color=TEXT if cm[i, j] > 0 else MUTED)
        ax.set_xticks(range(3), [s.capitalize() for s in SPECIES]); ax.set_yticks(range(3), [s.capitalize() for s in SPECIES])
        ax.tick_params(colors=TEXT)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{title}: accuracy {100*met['accuracy']:.1f}%, macro F1 {met['macro_f1']:.3f}",
                     loc="left", fontsize=12, fontweight="bold", color=TEXT)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle("Out-of-fold predictions: every flower is predicted by a model that never saw it",
                 x=0.02, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "04_confusion_matrices.png", dpi=180); plt.close(fig)

    # ------------------------------------------------------------- chart 5: the tree itself
    tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(df[FEATURES], y)
    fig, ax = plt.subplots(figsize=(16, 8))
    art = plot_tree(tree, feature_names=[LABEL[f] for f in FEATURES], class_names=[s.capitalize() for s in tree.classes_],
                    filled=False, rounded=True, impurity=True, fontsize=10, ax=ax)
    for t in art:
        t.set_color(TEXT)
        if t.get_bbox_patch():
            label = t.get_text().split("class = ")[-1].strip().lower()
            t.get_bbox_patch().set(facecolor=CARD, edgecolor=COLORS.get(label, GRID), linewidth=1.8)
        if t.arrow_patch is not None:
            t.arrow_patch.set_color(MUTED)
    ax.set_title("The depth-3 tree: three questions about the petals classify the flower", loc="left",
                 fontsize=14, fontweight="bold", color=TEXT)
    fig.savefig(OUT / "05_decision_tree.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    importance = dict(zip(FEATURES, tree.feature_importances_.round(3).tolist()))

    # ------------------------------------------------------------- chart 6: feature distributions (interactive)
    fig_d = go.Figure()
    for i, f in enumerate(FEATURES):
        for s in SPECIES:
            fig_d.add_trace(go.Violin(x=[LABEL[f]] * 50, y=df.loc[df["species"] == s, f], name=s.capitalize(),
                                      legendgroup=s, showlegend=i == 0, line_color=COLORS[s], fillcolor=COLORS[s],
                                      opacity=0.55, box_visible=True, meanline_visible=True, points=False, scalegroup=f))
    fig_d.update_layout(**plotly_layout(title=dict(text="Petals separate the species, sepals overlap", x=0.02),
                                        violinmode="group", xaxis=dict(color=TEXT), yaxis=dict(title="cm", gridcolor=GRID, color=TEXT2),
                                        legend=dict(font=dict(color=TEXT2)), height=560))
    fig_d.write_html(OUT / "06_feature_distributions.html", include_plotlyjs="cdn")

    # ------------------------------------------------------------- metrics and dashboard
    metrics = {
        "best_k": best_k, "best_depth": best_depth, "knn_vs_sklearn_prediction_agreement": agree,
        "cv_accuracy": {k_: {"mean": v[0], "std": v[1]} for k_, v in results.items()},
        "out_of_fold": {name: {"accuracy": mt["accuracy"], "macro_f1": mt["macro_f1"],
                               "recall_by_species": dict(zip(SPECIES, mt["recall"].round(3).tolist())),
                               "confusion": mt["confusion"].tolist()}
                        for name, mt in [("knn", m_knn), ("tree_depth3", m_tree)]},
        "tree_feature_importance": importance,
        "tree_rules": export_text(tree, feature_names=FEATURES),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    kpis = {"Flowers": "150", f"k-NN accuracy (k={best_k})": f"{100*results[f'k-NN, k={best_k} (scratch)'][0]:.1f}%",
            "Tree accuracy (depth 3)": f"{100*tree_curve[2][0]:.1f}%", "Agreement with scikit-learn": f"{100*agree:.0f}%"}
    kpi_html = "".join(f'<div class="kpi"><span>{k_}</span><b>{v}</b></div>' for k_, v in kpis.items())
    sec = lambda t, n, f: f'<section><h2>{t}</h2><p class="note">{n}</p>{f.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})}</section>'
    img = lambda t, n, src: f'<section><h2>{t}</h2><p class="note">{n}</p><img src="outputs/{src}" alt="{t}"></section>'
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Iris Classification</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
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
<header><h1>Iris Flower Classification</h1><p>k-nearest neighbors implemented from scratch and decision trees, evaluated with
repeated stratified cross-validation on the 150 flowers of Fisher's Iris dataset.</p></header><div class="kpis">{kpi_html}</div>
{sec("Species in 3D", "Drag to rotate. Translucent shapes are the convex hull of each species; white rings mark the flowers k-NN gets wrong.", fig3d)}
{img("Decision boundaries", "Same two petal measurements, three models.", "02_decision_boundaries.png")}
{img("Model selection", "Repeated stratified 5-fold cross-validation (50 test folds per point).", "03_model_selection.png")}
{img("Confusion matrices", "Out-of-fold predictions.", "04_confusion_matrices.png")}
{img("The decision tree", "Every split, its Gini impurity and the samples that reach it.", "05_decision_tree.png")}
{sec("Feature distributions", "Hover for quartiles.", fig_d)}
<footer>Tomás Alarcón &middot; <a href="https://github.com/tomasalarcon10/ml-from-scratch">Source code</a></footer></body></html>"""
    (ROOT / "dashboard.html").write_text(page, encoding="utf-8")
    print(json.dumps({k_: v for k_, v in metrics.items() if k_ != "tree_rules"}, indent=1)[:1500])


if __name__ == "__main__":
    main()
