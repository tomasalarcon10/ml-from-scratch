"""
Decision Tree from scratch: how a tree chooses each question (Gini, entropy, information gain) on the Palmer penguins.
Run:  python main.py      (under a minute on a laptop CPU)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import FancyBboxPatch
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier

from src.tree import DecisionTree, gini, entropy, scan_feature, cross_val_accuracy, stratified_kfold

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

BG, CARD, TEXT, TEXT2, MUTED, GRID = "#1a1a19", "#222220", "#f0efec", "#c3c2b7", "#8a8980", "#3a3936"
SPECIES_COLORS = ["#d95926", "#9085e9", "#199e70"]  # Adelie, Chinstrap, Gentoo
BLUE, AMBER = "#3987e5", "#c98500"
FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
LABELS = {"bill_length_mm": "Bill length (mm)", "bill_depth_mm": "Bill depth (mm)",
          "flipper_length_mm": "Flipper length (mm)", "body_mass_g": "Body mass (g)"}

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


def main():
    df = pd.read_csv(ROOT / "data" / "penguins.csv").dropna(subset=FEATURES).reset_index(drop=True)
    species = sorted(df["species"].unique())
    X = df[FEATURES].to_numpy(float)
    y = df["species"].map({s: i for i, s in enumerate(species)}).to_numpy()
    folds = stratified_kfold(y, k=4, seed=42)
    te, tr = folds[0], np.concatenate(folds[1:])
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    print(f"{len(df)} penguins ({dict(zip(species, np.bincount(y)))}); train {len(tr)}, test {len(te)}")

    # ------------------------------------------------------------- correctness against scikit-learn
    checks = {}
    for crit in ("gini", "entropy"):
        mine = DecisionTree(crit).fit(Xtr, ytr, n_classes=3)
        sk = DecisionTreeClassifier(criterion=crit, random_state=0).fit(Xtr, ytr)
        a_mine, _ = mine.pruning_path()
        a_sk = sk.cost_complexity_pruning_path(Xtr, ytr).ccp_alphas
        checks[crit] = {"prediction_agreement_test": float((mine.predict(Xte) == sk.predict(Xte)).mean()),
                        "same_depth_and_leaves": bool(mine.depth() == sk.get_depth() and mine.n_leaves() == sk.get_n_leaves()),
                        "same_root_split": bool(mine.root_.feature == sk.tree_.feature[0] and abs(mine.root_.threshold - sk.tree_.threshold[0]) < 1e-9),
                        "pruning_alphas_max_abs_diff": float(np.max(np.abs(a_mine - a_sk))) if len(a_mine) == len(a_sk) else None}
        print(f"  {crit}: {checks[crit]}")

    # ------------------------------------------------------------- model selection by cross-validation on the training set
    depths = [1, 2, 3, 4, 5, 6, None]
    cv_depth = {crit: [cross_val_accuracy(lambda d=d: DecisionTree(crit, max_depth=d), Xtr, ytr, k=5, repeats=10) for d in depths]
                for crit in ("gini", "entropy")}
    full = DecisionTree("gini").fit(Xtr, ytr, n_classes=3)
    alphas, impurities = full.pruning_path()
    leaves_path = [full.prune(a).n_leaves() for a in alphas]

    def cv_alpha(a):
        scores = []
        for r in range(10):
            fs = stratified_kfold(ytr, 5, seed=r)
            for i in range(5):
                vt = fs[i]; tt = np.concatenate([fs[j] for j in range(5) if j != i])
                m = DecisionTree("gini").fit(Xtr[tt], ytr[tt], n_classes=3).prune(a)
                scores.append((m.predict(Xtr[vt]) == ytr[vt]).mean())
        return float(np.mean(scores)), float(np.std(scores))
    cv_prune = [cv_alpha(a) for a in alphas]
    best_mean = max(m for m, _ in cv_prune)
    # one-standard-error rule: the simplest tree whose CV accuracy is within one standard error of the best
    se = cv_prune[int(np.argmax([m for m, _ in cv_prune]))][1] / np.sqrt(50)
    k_best = max(i for i, (m, _) in enumerate(cv_prune) if m >= best_mean - se)
    alpha_star = float(alphas[k_best])
    final = full.prune(alpha_star)
    print(f"Pruning: alpha {alpha_star:.4f} -> {final.n_leaves()} leaves, depth {final.depth()} (full tree {full.n_leaves()} leaves)")

    def test_acc(m):
        return float((m.predict(Xte) == yte).mean())
    models = {
        "Full tree (Gini)": full, "Full tree (entropy)": DecisionTree("entropy").fit(Xtr, ytr, n_classes=3),
        "Depth 1 (one question)": DecisionTree("gini", max_depth=1).fit(Xtr, ytr, n_classes=3),
        "Depth 2": DecisionTree("gini", max_depth=2).fit(Xtr, ytr, n_classes=3),
        f"Pruned tree (alpha {alpha_star:.3f})": final,
    }
    results = {n: {"test_accuracy": test_acc(m), "train_accuracy": float((m.predict(Xtr) == ytr).mean()),
                   "leaves": m.n_leaves(), "depth": m.depth()} for n, m in models.items()}
    for n, r in results.items():
        print(f"  {n:28s} train {r['train_accuracy']:.3f}  test {r['test_accuracy']:.3f}  leaves {r['leaves']}")
    pred = final.predict(Xte)
    cm = np.zeros((3, 3), int)
    for t_, p_ in zip(yte, pred):
        cm[t_, p_] += 1
    importances = final.feature_importances()

    # ------------------------------------------------------------- chart 1: 3D penguins with the split planes
    axes3 = [0, 2, 1]  # bill length (x), flipper length (y), bill depth (z)
    lo, hi = X.min(0) - np.array([1, 0.5, 3, 150]), X.max(0) + np.array([1, 0.5, 3, 150])
    fig3d = go.Figure()
    for k, s in enumerate(species):
        m = y == k
        fig3d.add_trace(go.Scatter3d(
            x=X[m, 0], y=X[m, 2], z=X[m, 1], mode="markers", name=s,
            marker=dict(size=4, color=SPECIES_COLORS[k], opacity=0.9, line=dict(width=0.5, color=BG)),
            customdata=X[m][:, [3]], hovertemplate=f"<b>{s}</b><br>Bill %{{x:.1f}} x %{{z:.1f}} mm<br>Flipper %{{y:.0f}} mm<br>Mass %{{customdata[0]:.0f}} g<extra></extra>"))

    def planes(node, box, order=[0]):
        if node.is_leaf:
            return
        f, t = node.feature, node.threshold
        if f in axes3:
            others = [a for a in axes3 if a != f]
            grid = {f: np.full((2, 2), t)}
            g0, g1 = np.meshgrid([box[others[0]][0], box[others[0]][1]], [box[others[1]][0], box[others[1]][1]])
            grid[others[0]], grid[others[1]] = g0, g1
            order[0] += 1
            fig3d.add_trace(go.Surface(x=grid[0], y=grid[2], z=grid[1], showscale=False, opacity=0.22,
                                       colorscale=[[0, TEXT2], [1, TEXT2]], showlegend=order[0] == 1,
                                       name="Split planes of the pruned tree", legendgroup="planes",
                                       hovertemplate=f"Question {node.id}: {LABELS[FEATURES[f]]} <= {t:.1f}<extra></extra>"))
        left = {**box, f: (box[f][0], t)}; right = {**box, f: (t, box[f][1])}
        planes(node.left, left); planes(node.right, right)
    planes(final.root_, {j: (lo[j], hi[j]) for j in range(4)})
    axis = dict(backgroundcolor=BG, gridcolor=GRID, zerolinecolor=GRID, color=TEXT2)
    fig3d.update_layout(**plotly_layout(
        title=dict(text="342 penguins and the questions the tree asks: every plane is one split of the pruned tree", x=0.02),
        scene=dict(xaxis=dict(title="Bill length (mm)", range=[lo[0], hi[0]], **axis),
                   yaxis=dict(title="Flipper length (mm)", range=[lo[2], hi[2]], **axis),
                   zaxis=dict(title="Bill depth (mm)", range=[lo[1], hi[1]], **axis),
                   camera=dict(eye=dict(x=1.6, y=-1.5, z=0.75))),
        legend=dict(font=dict(color=TEXT2), x=0.01, y=0.92, itemsizing="constant"), height=760))
    fig3d.write_html(OUT / "01_penguins_split_planes_3d.html", include_plotlyjs="cdn")
    try:
        fig3d.write_image(OUT / "01_penguins_split_planes_3d.png", width=1600, height=900, scale=2)
    except Exception as exc:  # noqa: BLE001
        print(f"  (static 3D export skipped: {exc.__class__.__name__})")

    # ------------------------------------------------------------- chart 2: how the first question is chosen
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.2), sharex="col", gridspec_kw={"height_ratios": [1.5, 1]})
    root_scan = {}
    for j, f in enumerate(FEATURES):
        for crit, ls, col in (("gini", "-", BLUE), ("entropy", "-", AMBER)):
            thr, gain = scan_feature(Xtr[:, j], ytr, 3, crit)
            root_scan[(f, crit)] = (thr, gain)
            ax = axes[0, j]
            ax.plot(thr, gain / (gini([1, 1, 1]) if crit == "gini" else entropy([1, 1, 1])), color=col, lw=2, ls=ls,
                    label=f"{crit.capitalize()} gain")
        thr, gain = root_scan[(f, "gini")]
        k = int(np.argmax(gain))
        axes[0, j].scatter([thr[k]], [gain[k] / gini([1, 1, 1])], color=BLUE, s=50, zorder=5)
        axes[0, j].annotate(f"best: <= {thr[k]:.1f}\nGini gain {gain[k]:.3f}", (thr[k], gain[k] / gini([1, 1, 1])),
                            xytext=(8, -2), textcoords="offset points", color=TEXT, fontsize=9)
        axes[0, j].grid(lw=0.5); axes[0, j].set_ylim(0, 0.75)
        title(axes[0, j], LABELS[f], 11.5)
        for k_, s in enumerate(species):
            v = Xtr[ytr == k_, j]
            axes[1, j].hist(v, bins=24, range=(X[:, j].min(), X[:, j].max()), color=SPECIES_COLORS[k_], alpha=0.75, label=s)
        axes[1, j].set_xlabel(LABELS[f]); axes[1, j].grid(axis="y", lw=0.5)
    axes[0, 0].set_ylabel("Information gain\n(share of the root impurity removed)")
    axes[1, 0].set_ylabel("Penguins")
    axes[0, 0].legend(frameon=False, labelcolor=TEXT2, loc="upper left")
    axes[1, 0].legend(frameon=False, labelcolor=TEXT2, fontsize=8.5)
    fig.suptitle("How the tree chooses its first question: every threshold of every feature is scored, and the largest gain wins",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "02_choosing_the_first_question.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 3: impurity functions
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    p = np.linspace(0, 1, 401)
    two = np.c_[p, 1 - p]
    axes[0].plot(p, entropy(two), color=AMBER, lw=2.4, label="Entropy (bits)")
    axes[0].plot(p, gini(two), color=BLUE, lw=2.4, label="Gini")
    axes[0].plot(p, 1 - np.maximum(p, 1 - p), color=MUTED, lw=1.6, ls="--", label="Misclassification rate")
    axes[0].set_xlabel("Share of class A in the node (two classes)"); axes[0].set_ylabel("Impurity")
    axes[0].legend(frameon=False, labelcolor=TEXT2); axes[0].grid(lw=0.5)
    title(axes[0], "Impurity is zero for a pure node and maximal at 50/50")
    n = 60
    pts = np.array([(i, j, n - i - j) for i in range(n + 1) for j in range(n + 1 - i)], float) / n
    cx = pts[:, 1] + 0.5 * pts[:, 2]; cy = np.sqrt(3) / 2 * pts[:, 2]
    tri = mtri.Triangulation(cx, cy)
    for ax, f, name, cmap in [(axes[1], gini, "Gini", LinearSegmentedColormap.from_list("g", [CARD, "#1f3b5c", BLUE, "#b9d6f8"])),
                              (axes[2], entropy, "Entropy", LinearSegmentedColormap.from_list("e", [CARD, "#4a3200", AMBER, "#f3d28c"]))]:
        cs = ax.tricontourf(tri, f(pts), levels=18, cmap=cmap)
        ax.tricontour(tri, f(pts), levels=6, colors=BG, linewidths=0.6)
        ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], color=MUTED, lw=1)
        for (x_, y_), s, c, ha in [((0, -0.06), species[0], SPECIES_COLORS[0], "center"), ((1, -0.06), species[1], SPECIES_COLORS[1], "center"),
                                   ((0.5, np.sqrt(3) / 2 + 0.04), species[2], SPECIES_COLORS[2], "center")]:
            ax.text(x_, y_, f"100% {s}", color=c, ha=ha, fontsize=10, fontweight="bold")
        r = final.root_.counts / final.root_.counts.sum()
        ax.scatter([r[1] + 0.5 * r[2]], [np.sqrt(3) / 2 * r[2]], color=TEXT, s=40, zorder=5)
        ax.annotate(f"root node: {name.lower()} {f(final.root_.counts):.3f}", (r[1] + 0.5 * r[2], np.sqrt(3) / 2 * r[2]),
                    xytext=(10, 6), textcoords="offset points", color=TEXT, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec="none", alpha=0.85))
        cb = fig.colorbar(cs, ax=ax, fraction=0.04, pad=0.02); cb.outline.set_visible(False); cb.ax.tick_params(colors=MUTED)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{name} for every mix of three species", loc="left", fontsize=12.5, fontweight="bold", color=TEXT, pad=26)
    fig.tight_layout()
    fig.savefig(OUT / "03_impurity_functions.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 4: the pruned tree itself
    def layout(node, x0, x1, pos):
        if node.is_leaf:
            pos[node.id] = ((x0 + x1) / 2, node.depth); return
        left_leaves = sum(n_.is_leaf for n_ in final.nodes(node.left))
        right_leaves = sum(n_.is_leaf for n_ in final.nodes(node.right))
        mid = x0 + (x1 - x0) * left_leaves / (left_leaves + right_leaves)
        layout(node.left, x0, mid, pos); layout(node.right, mid, x1, pos)
        pos[node.id] = ((pos[node.left.id][0] + pos[node.right.id][0]) / 2, node.depth)
    pos = {}
    layout(final.root_, 0, 1, pos)
    D = final.depth()
    fig, ax = plt.subplots(figsize=(17, 2.6 * (D + 1)))
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(D + 0.6, -0.6); ax.axis("off")
    w, h = min(0.16, 0.9 / final.n_leaves()), 0.62
    for node in final.nodes():
        x_, d_ = pos[node.id]
        if not node.is_leaf:
            for child, lab in ((node.left, "yes"), (node.right, "no")):
                cx_, cd_ = pos[child.id]
                ax.plot([x_, cx_], [d_ + h / 2, cd_ - h / 2], color=GRID, lw=1.5, zorder=1)
                ax.text((x_ + cx_) / 2, (d_ + cd_) / 2, lab, color=MUTED, fontsize=9, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec="none"))
        maj = int(node.counts.argmax())
        ax.add_patch(FancyBboxPatch((x_ - w / 2, d_ - h / 2), w, h, boxstyle="round,pad=0.005,rounding_size=0.02",
                                    fc=CARD, ec=SPECIES_COLORS[maj] if node.is_leaf else GRID, lw=2 if node.is_leaf else 1, zorder=2))
        head = f"{LABELS[FEATURES[node.feature]].replace(' (', ' (').split(' (')[0]} <= {node.threshold:.1f}?" if not node.is_leaf else species[maj]
        ax.text(x_, d_ - h / 2 + 0.11, head, ha="center", va="center", fontsize=10.5, fontweight="bold",
                color=TEXT if not node.is_leaf else SPECIES_COLORS[maj], zorder=3)
        ax.text(x_, d_ - h / 2 + 0.25, f"{node.n} penguins  |  Gini {node.impurity:.3f}", ha="center", va="center", fontsize=8.5, color=TEXT2, zorder=3)
        if not node.is_leaf:
            ax.text(x_, d_ - h / 2 + 0.37, f"gain {node.gain:.3f}", ha="center", va="center", fontsize=8.5, color=BLUE, zorder=3)
        bx = x_ - w / 2 + 0.008; bw = w - 0.016
        acc = 0
        for k, c in enumerate(node.counts):
            if c:
                ax.add_patch(plt.Rectangle((bx + bw * acc / node.n, d_ + h / 2 - 0.13), bw * c / node.n, 0.08, color=SPECIES_COLORS[k], zorder=3))
            acc += c
    for k, s in enumerate(species):
        ax.text(0.0 + 0.1 * k, D + 0.5, f"■ {s}", color=SPECIES_COLORS[k], fontsize=10.5)
    title(ax, f"The pruned tree: {final.n_leaves()} leaves, {results[f'Pruned tree (alpha {alpha_star:.3f})']['test_accuracy']:.1%} accuracy on unseen penguins "
              f"(the bar in each box is its species mix)", 14)
    fig.tight_layout()
    fig.savefig(OUT / "04_the_tree.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 5: decision boundaries as depth grows
    j1, j2 = 0, 2
    gx, gy = np.meshgrid(np.linspace(X[:, j1].min() - 2, X[:, j1].max() + 2, 400), np.linspace(X[:, j2].min() - 5, X[:, j2].max() + 5, 400))
    grid2 = np.c_[gx.ravel(), gy.ravel()]
    region = ListedColormap(["#4a2414", "#2e2a4f", "#0f3a2c"])
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, d in zip(axes, [1, 2, 3, None]):
        m2 = DecisionTree("gini", max_depth=d).fit(Xtr[:, [j1, j2]], ytr, n_classes=3)
        ax.imshow(m2.predict(grid2).reshape(gx.shape), origin="lower", extent=(gx.min(), gx.max(), gy.min(), gy.max()),
                  cmap=region, aspect="auto", vmin=0, vmax=2, interpolation="nearest")
        for k in range(3):
            mk = ytr == k
            ax.scatter(Xtr[mk, j1], Xtr[mk, j2], s=13, color=SPECIES_COLORS[k], edgecolor=BG, lw=0.4)
        acc2 = (m2.predict(Xte[:, [j1, j2]]) == yte).mean()
        title(ax, f"{'Unlimited depth' if d is None else f'Depth {d}'}: {m2.n_leaves()} leaves, test {acc2:.1%}", 11.5)
        ax.set_xlabel(LABELS[FEATURES[j1]])
    axes[0].set_ylabel(LABELS[FEATURES[j2]])
    fig.suptitle("Every question cuts the space with a line parallel to an axis; deeper trees carve ever smaller boxes around a few penguins",
                 x=0.01, ha="left", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "05_decision_boundaries.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 6: model selection and pruning
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    xs = np.arange(len(depths)); lab = [str(d) if d else "full" for d in depths]
    for crit, col in (("gini", BLUE), ("entropy", AMBER)):
        mm = np.array([m for m, _ in cv_depth[crit]]); ss = np.array([s for _, s in cv_depth[crit]])
        axes[0].plot(xs, 100 * mm, color=col, lw=2.2, marker="o", label=crit.capitalize())
        axes[0].fill_between(xs, 100 * (mm - ss), 100 * (mm + ss), color=col, alpha=0.12)
    axes[0].set_xticks(xs, lab); axes[0].set_xlabel("Maximum depth"); axes[0].set_ylabel("Cross-validated accuracy (%)")
    axes[0].legend(frameon=False, labelcolor=TEXT2); axes[0].grid(lw=0.5)
    title(axes[0], "Gini or entropy: almost the same tree")
    axes[1].step(alphas, leaves_path, where="post", color=BLUE, lw=2.2)
    axes[1].axvline(alpha_star, color=AMBER, ls="--", lw=1.4)
    axes[1].set_xlabel("Complexity penalty alpha"); axes[1].set_ylabel("Number of leaves"); axes[1].grid(lw=0.5)
    title(axes[1], "Cost-complexity pruning path")
    mm = np.array([m for m, _ in cv_prune])
    axes[2].plot(range(len(alphas)), 100 * mm, color=BLUE, lw=2.2, marker="o")
    axes[2].axhline(100 * (best_mean - se), color=MUTED, ls=":", lw=1)
    axes[2].scatter([k_best], [100 * mm[k_best]], s=140, facecolors="none", edgecolors=AMBER, lw=2, zorder=5)
    axes[2].annotate(f"selected: {final.n_leaves()} leaves\n(one-standard-error rule)", (k_best, 100 * mm[k_best]),
                     xytext=(-120, -50), textcoords="offset points", color=AMBER, arrowprops=dict(arrowstyle="-", color=AMBER))
    axes[2].set_xticks(range(len(alphas)), [str(l_) for l_ in leaves_path]); axes[2].set_xlabel("Leaves after pruning")
    axes[2].set_ylabel("Cross-validated accuracy (%)"); axes[2].grid(lw=0.5)
    title(axes[2], "Smaller trees generalize as well")
    fig.tight_layout()
    fig.savefig(OUT / "06_model_selection_and_pruning.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 7: confusion + importance
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1.2]})
    axes[0].imshow(np.eye(3), cmap=LinearSegmentedColormap.from_list("c", [CARD, "#183a2f"]), vmin=0, vmax=1)
    for i in range(3):
        for k in range(3):
            axes[0].text(k, i, cm[i, k], ha="center", va="center", fontsize=20, fontweight="bold",
                         color="#199e70" if i == k else ("#e66767" if cm[i, k] else MUTED))
    axes[0].set_xticks(range(3), species); axes[0].set_yticks(range(3), species); axes[0].tick_params(colors=TEXT, length=0)
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    for sp in axes[0].spines.values():
        sp.set_visible(False)
    title(axes[0], f"Pruned tree on {len(te)} unseen penguins: {cm.trace()} correct")
    order = np.argsort(importances)
    axes[1].barh([LABELS[FEATURES[i]] for i in order], 100 * importances[order], color=BLUE, height=0.55)
    for i, v in enumerate(importances[order]):
        axes[1].text(100 * v + 1, i, f"{100 * v:.0f}%", va="center", color=TEXT)
    axes[1].tick_params(axis="y", colors=TEXT, length=0); axes[1].grid(axis="x", lw=0.5); axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Share of the total impurity removed (%)")
    title(axes[1], "Which measurements the tree relies on")
    fig.tight_layout()
    fig.savefig(OUT / "07_confusion_and_importance.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- metrics
    root = final.root_
    metrics = {
        "data": {"penguins": len(df), "species_counts": dict(zip(species, np.bincount(y).tolist())), "train": len(tr), "test": len(te),
                 "features": FEATURES},
        "sklearn_checks": checks,
        "root_question": {"feature": FEATURES[root.feature], "threshold": root.threshold, "gini_gain": root.gain,
                          "best_threshold_per_feature_gini": {f: {"threshold": float(root_scan[(f, 'gini')][0][np.argmax(root_scan[(f, 'gini')][1])]),
                                                                  "gain": float(root_scan[(f, 'gini')][1].max())} for f in FEATURES}},
        "cv_accuracy_by_depth": {c: {str(d): {"mean": round(m, 4), "std": round(s, 4)} for d, (m, s) in zip(depths, v)} for c, v in cv_depth.items()},
        "pruning": {"alphas": alphas.round(5).tolist(), "leaves": leaves_path, "cv_accuracy": [round(m, 4) for m, _ in cv_prune],
                    "selected_alpha": alpha_star, "rule": "one standard error"},
        "models": results, "final_confusion_matrix": cm.tolist(),
        "final_feature_importance": dict(zip(FEATURES, importances.round(4).tolist())),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ------------------------------------------------------------- dashboard
    live = {"features": FEATURES, "labels": [LABELS[f] for f in FEATURES], "species": species, "colors": SPECIES_COLORS,
            "tree": final.to_dict(), "X": Xtr.round(1).tolist(), "y": ytr.tolist(),
            "ranges": [[float(X[:, j].min()), float(X[:, j].max())] for j in range(4)]}
    (OUT / "tree.js").write_text("window.TREE_DATA=" + json.dumps(live, separators=(",", ":")) + ";", encoding="utf-8")
    final_name = f"Pruned tree (alpha {alpha_star:.3f})"
    kpis = {"Test accuracy (pruned)": f"{100 * results[final_name]['test_accuracy']:.1f}%", "Leaves": f"{final.n_leaves()} (full tree {full.n_leaves()})",
            "Agreement with scikit-learn": f"{100 * checks['gini']['prediction_agreement_test']:.0f}%",
            "First question": f"{LABELS[FEATURES[root.feature]].split(' (')[0]} <= {root.threshold:.1f}"}
    kpi_html = "".join(f'<div class="kpi"><span>{k}</span><b>{v}</b></div>' for k, v in kpis.items())
    sec = lambda t, n_, f: f'<section><h2>{t}</h2><p class="note">{n_}</p>{f.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})}</section>'
    img = lambda t, n_, src: f'<section><h2>{t}</h2><p class="note">{n_}</p><img src="outputs/{src}" alt="{t}"></section>'
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Decision Tree from Scratch</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><script src="outputs/tree.js"></script>
<style>:root{{--bg:{BG};--card:{CARD};--text:{TEXT};--text2:{TEXT2};--muted:{MUTED};--line:{GRID}}}*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font-family:Inter,"Segoe UI",Arial,sans-serif}}
header,.kpis,footer{{max-width:1240px;margin:auto;padding:0 32px}}header{{padding-top:48px}}h1{{font-size:40px;margin:0 0 8px}}
header p{{color:var(--text2);max-width:860px;line-height:1.55;margin:0}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-top:24px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px}}
.kpi span{{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.8px}}.kpi b{{display:block;font-size:22px;margin-top:6px}}
section{{max-width:1240px;margin:28px auto;padding:20px 24px;background:var(--card);border:1px solid var(--line);border-radius:14px}}
section img{{width:100%;border-radius:8px}}h2{{margin:4px 0 6px;font-size:20px}}.note{{color:var(--text2);margin:0 0 10px;font-size:14px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}label{{display:grid;grid-template-columns:140px 1fr 70px;align-items:center;gap:10px;color:var(--text2);font-size:13px;margin:8px 0}}
input[type=range]{{width:100%;accent-color:{BLUE}}}output{{text-align:right;color:var(--text);font-variant-numeric:tabular-nums}}
select{{background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:6px}}
.side{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}}.box{{background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:12px}}
.box h3{{margin:0 0 8px;font-size:14px;color:var(--text2);font-weight:600}}.stack{{display:flex;height:14px;border-radius:7px;overflow:hidden;margin:6px 0}}
.metric{{display:flex;justify-content:space-between;font-size:13px;color:var(--text2);margin:3px 0}}.metric b{{color:var(--text)}}.big{{font-size:26px;font-weight:800}}
.tree ul{{list-style:none;margin:0;padding-left:22px;border-left:1px dashed var(--line)}}.tree>ul{{padding-left:0;border:0}}
.tree li{{margin:6px 0}}.nd{{display:inline-block;padding:6px 10px;border-radius:8px;border:1px solid var(--line);background:var(--bg);font-size:13px;color:var(--text2);transition:all .15s}}
.nd.on{{border-color:{BLUE};color:var(--text);box-shadow:0 0 0 2px {BLUE}55}}.nd.leaf.on{{font-weight:700}}.edge{{color:var(--muted);font-size:11px;margin-right:6px}}
footer{{color:var(--muted);font-size:13px;margin:24px auto 48px}}a{{color:#7fb0f0}}
@media(max-width:760px){{.grid2,.side{{grid-template-columns:1fr}}header,.kpis,footer{{padding:0 16px}}section{{margin:16px;padding:12px}}h1{{font-size:30px}}label{{grid-template-columns:100px 1fr 60px}}}}</style></head><body>
<header><h1>Decision Tree from Scratch</h1><p>How a decision tree chooses each question. A CART classifier written in NumPy (Gini, entropy, information gain,
cost-complexity pruning) identifies three penguin species from four body measurements and matches scikit-learn prediction for prediction.</p></header>
<div class="kpis">{kpi_html}</div>
<section><h2>Be the tree: choose the first question</h2><p class="note">Pick a measurement and move the threshold. The {len(tr)} training penguins are split in two; the gain shows how much purer the two groups are. The tree tries every option and keeps the best one.</p>
<label>Measurement<select id="feat"></select><span></span></label><label>Threshold<input type="range" id="thr"><output id="thr_o"></output></label>
<div class="side"><div class="box"><h3>Yes (&le; threshold)</h3><div class="stack" id="lstack"></div><div id="lstat"></div></div>
<div class="box"><h3>No (&gt; threshold)</h3><div class="stack" id="rstack"></div><div id="rstat"></div></div></div>
<div class="side"><div class="box"><div class="metric"><span>Gini gain</span><b id="gg"></b></div><div class="metric"><span>Information gain (entropy, bits)</span><b id="eg"></b></div></div>
<div class="box"><div class="metric"><span>Best threshold for this measurement</span><b id="best"></b></div><button id="snap" style="margin-top:6px;background:var(--card);color:var(--text2);border:1px solid var(--line);border-radius:6px;padding:6px 10px;cursor:pointer">Jump to the best threshold</button></div></div></section>
<section><h2>Follow a penguin through the tree</h2><p class="note">Set the measurements of a penguin: the questions it answers light up and the leaf gives the species.</p>
<div class="grid2"><div id="sliders"></div><div><div class="note">Prediction</div><div class="big" id="pred"></div><div class="stack" id="pstack" style="height:18px"></div><div class="note" id="pnote"></div></div></div>
<div class="tree" id="tree" style="margin-top:14px"></div></section>
{sec("Penguins and split planes in 3D", "Drag to rotate. Each translucent plane is one question of the pruned tree, drawn only inside the region where it applies.", fig3d)}
{img("Choosing the first question", "Gini and entropy gain for every possible threshold of every measurement.", "02_choosing_the_first_question.png")}
{img("Impurity functions", "What Gini and entropy measure, for two classes and for every mix of the three species.", "03_impurity_functions.png")}
{img("The tree", "The final pruned tree as learned by the code.", "04_the_tree.png")}
{img("Decision boundaries", "Trained on two measurements only, so the cuts can be drawn.", "05_decision_boundaries.png")}
{img("Model selection and pruning", "Cross-validation on the training set chooses how big the tree should be.", "06_model_selection_and_pruning.png")}
{img("Results", "Errors and feature importance of the final tree.", "07_confusion_and_importance.png")}
<footer>Tomás Alarcón &middot; <a href="https://github.com/tomasalarcon10/ml-from-scratch">Source code</a></footer>
<script>
const T=window.TREE_DATA,K=T.species.length;
const gini=c=>{{const n=c.reduce((a,b)=>a+b,0);return n?1-c.reduce((a,b)=>a+(b/n)**2,0):0}};
const ent=c=>{{const n=c.reduce((a,b)=>a+b,0);return n?-c.reduce((a,b)=>a+(b?b/n*Math.log2(b/n):0),0):0}};
const stack=(el,c)=>{{const n=c.reduce((a,b)=>a+b,0)||1;el.innerHTML=c.map((v,k)=>`<div style="width:${{100*v/n}}%;background:${{T.colors[k]}}"></div>`).join("")}};
const stat=c=>`${{c.reduce((a,b)=>a+b,0)}} penguins<br>`+c.map((v,k)=>`<span style="color:${{T.colors[k]}}">${{T.species[k]}} ${{v}}</span>`).join(" &middot; ")+`<br>Gini ${{gini(c).toFixed(3)}} &middot; entropy ${{ent(c).toFixed(3)}}`;
const feat=document.getElementById("feat"),thr=document.getElementById("thr");
T.labels.forEach((l,i)=>feat.innerHTML+=`<option value="${{i}}">${{l}}</option>`);
function split(j,t){{const L=Array(K).fill(0),R=Array(K).fill(0);T.X.forEach((x,i)=>(x[j]<=t?L:R)[T.y[i]]++);return [L,R]}}
function gains(j,t){{const [L,R]=split(j,t),P=L.map((v,k)=>v+R[k]),n=T.X.length,nl=L.reduce((a,b)=>a+b,0),nr=n-nl;
return [gini(P)-nl/n*gini(L)-nr/n*gini(R),ent(P)-nl/n*ent(L)-nr/n*ent(R),L,R]}}
function bestThr(j){{const v=[...new Set(T.X.map(x=>x[j]))].sort((a,b)=>a-b);let b=[-1,0];for(let i=0;i+1<v.length;i++){{const t=(v[i]+v[i+1])/2,g=gains(j,t)[0];if(g>b[0])b=[g,t]}}return b}}
function setFeat(){{const j=+feat.value,[lo,hi]=T.ranges[j],step=j===3?25:0.5;thr.min=lo;thr.max=hi;thr.step=step;thr.value=(lo+hi)/2;const b=bestThr(j);
document.getElementById("best").textContent=`${{b[1].toFixed(1)}} (Gini gain ${{b[0].toFixed(3)}})`;document.getElementById("snap").onclick=()=>{{thr.value=b[1];upd()}};upd()}}
function upd(){{const j=+feat.value,t=+thr.value,[g,e,L,R]=gains(j,t);document.getElementById("thr_o").textContent=t.toFixed(1);
stack(document.getElementById("lstack"),L);stack(document.getElementById("rstack"),R);document.getElementById("lstat").innerHTML=stat(L);document.getElementById("rstat").innerHTML=stat(R);
document.getElementById("gg").textContent=g.toFixed(3);document.getElementById("eg").textContent=e.toFixed(3)}}
feat.onchange=setFeat;thr.oninput=upd;feat.value=2;setFeat();
const sl=document.getElementById("sliders"),defaults=[45,15,210,4800];
T.labels.forEach((l,j)=>{{const [lo,hi]=T.ranges[j];sl.innerHTML+=`<label>${{l}}<input type="range" id="s${{j}}" min="${{lo}}" max="${{hi}}" step="${{j===3?25:0.5}}" value="${{defaults[j]}}"><output id="s${{j}}_o"></output></label>`}});
function render(n,edge){{const leaf=!n.left,maj=n.counts.indexOf(Math.max(...n.counts));
const txt=leaf?`<span style="color:${{T.colors[maj]}}">${{T.species[maj]}}</span> &middot; ${{n.counts.reduce((a,b)=>a+b,0)}} penguins`:`${{T.labels[n.feature]}} &le; ${{n.threshold}}?`;
return `<li>${{edge?`<span class="edge">${{edge}}</span>`:""}}<span class="nd${{leaf?" leaf":""}}" id="n${{n.id}}">${{txt}}</span>${{leaf?"":`<ul>${{render(n.left,"yes")}}${{render(n.right,"no")}}</ul>`}}</li>`}}
document.getElementById("tree").innerHTML=`<ul>${{render(T.tree,"")}}</ul>`;
function trace(){{const x=T.labels.map((_,j)=>{{const v=+document.getElementById("s"+j).value;document.getElementById("s"+j+"_o").textContent=v;return v}});
document.querySelectorAll(".nd").forEach(e=>e.classList.remove("on"));let n=T.tree;while(true){{document.getElementById("n"+n.id).classList.add("on");if(!n.left)break;n=x[n.feature]<=n.threshold?n.left:n.right}}
const tot=n.counts.reduce((a,b)=>a+b,0),maj=n.counts.indexOf(Math.max(...n.counts));const p=document.getElementById("pred");p.textContent=T.species[maj];p.style.color=T.colors[maj];
stack(document.getElementById("pstack"),n.counts);document.getElementById("pnote").textContent=`The leaf holds ${{tot}} training penguins: ${{n.counts.map((v,k)=>`${{v}} ${{T.species[k]}}`).join(", ")}}.`}}
document.querySelectorAll("#sliders input").forEach(e=>e.oninput=trace);trace();
</script></body></html>"""
    (ROOT / "dashboard.html").write_text(page, encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("root_question", "models")}, indent=1))


if __name__ == "__main__":
    main()
