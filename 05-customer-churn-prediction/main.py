"""
Customer Churn Prediction: LightGBM, class imbalance (SMOTE from scratch), explainability and a profit-driven decision.
Run:  python main.py      (under a minute on a laptop CPU)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.imbalance import stratified_split, random_oversample, smote
from src.evaluation import (pr_curve, average_precision, at_threshold, gains, profit_curve,
                            permutation_importance)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

BG, CARD, TEXT, TEXT2, MUTED, GRID = "#1a1a19", "#222220", "#f0efec", "#c3c2b7", "#8a8980", "#3a3936"
BLUE, ORANGE, AQUA, AMBER, PINK, VIOLET, RED = "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9", "#e66767"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": TEXT,
    "axes.labelcolor": TEXT2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
    "grid.color": GRID, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})

# Business assumptions for the retention campaign (editable live in the dashboard)
VALUE_SAVED = 1000    # value of keeping a customer who was about to leave (USD)
CONTACT_COST = 50     # cost of the retention offer per contacted customer (USD)
SUCCESS_RATE = 0.30   # share of contacted churners who stay thanks to the offer

LGB_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_estimators=2000, verbose=-1, random_state=7)


def plotly_layout(**kw):
    base = dict(paper_bgcolor=BG, plot_bgcolor=BG, font=dict(family="Inter, Segoe UI, Arial", color=TEXT, size=13),
                hoverlabel=dict(bgcolor=CARD, bordercolor=GRID, font=dict(color=TEXT)), margin=dict(l=20, r=20, t=70, b=20))
    base.update(kw)
    return base


def title(ax, text):
    ax.set_title(text, loc="left", fontsize=12.5, fontweight="bold", color=TEXT)


def build_features(df):
    X = pd.DataFrame({
        "CreditScore": df["CreditScore"], "Germany": (df["Geography"] == "Germany").astype(int),
        "Spain": (df["Geography"] == "Spain").astype(int), "Male": (df["Gender"] == "Male").astype(int),
        "Age": df["Age"], "Tenure": df["Tenure"], "Balance": df["Balance"], "NumOfProducts": df["NumOfProducts"],
        "HasCrCard": df["HasCrCard"], "IsActiveMember": df["IsActiveMember"], "EstimatedSalary": df["EstimatedSalary"],
    })
    X["ZeroBalance"] = (X["Balance"] == 0).astype(int)
    X["BalanceToSalary"] = X["Balance"] / X["EstimatedSalary"]
    return X


def main():
    df = pd.read_csv(ROOT / "data" / "bank_churn.csv")
    X = build_features(df)
    cols = X.columns.tolist()
    Xa, y = X.to_numpy(float), df["Exited"].to_numpy()
    tr, va, te = stratified_split(y, (0.6, 0.2, 0.2), seed=42)
    print(f"{len(df):,} customers, churn rate {y.mean():.1%}; train {len(tr)}, validation {len(va)}, test {len(te)}")

    def fit_lgb(Xt, yt, **kw):
        m = lgb.LGBMClassifier(**{**LGB_PARAMS, **kw})
        m.fit(Xt, yt, eval_set=[(Xa[va], y[va])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        return m

    # ------------------------------------------------------------- imbalance strategies
    binary_cols = [cols.index(c) for c in ["Germany", "Spain", "Male", "HasCrCard", "IsActiveMember", "ZeroBalance",
                                            "NumOfProducts", "Tenure"]]
    X_over, y_over = random_oversample(Xa[tr], y[tr])
    X_smote, y_smote = smote(Xa[tr], y[tr], k=5, binary_cols=binary_cols)
    spw = (y[tr] == 0).sum() / (y[tr] == 1).sum()
    mu, sd = Xa[tr].mean(0), Xa[tr].std(0)
    logit = LogisticRegression(class_weight="balanced", max_iter=2000).fit((Xa[tr] - mu) / sd, y[tr])

    models = {
        "Logistic regression (balanced)": (lambda Z: logit.predict_proba((Z - mu) / sd)[:, 1], None),
        "LightGBM, no rebalancing": fit_lgb(Xa[tr], y[tr]),
        "LightGBM + class weights": fit_lgb(Xa[tr], y[tr], scale_pos_weight=spw),
        "LightGBM + random oversampling": fit_lgb(X_over, y_over),
        "LightGBM + SMOTE (from scratch)": fit_lgb(X_smote, y_smote),
    }
    results, probs = {}, {}
    for name, m in models.items():
        predict = m[0] if isinstance(m, tuple) else (lambda Z, m=m: m.predict_proba(Z)[:, 1])
        p_va, p_te = predict(Xa[va]), predict(Xa[te])
        probs[name] = (p_va, p_te)
        results[name] = {"val_roc_auc": roc_auc_score(y[va], p_va), "test_roc_auc": roc_auc_score(y[te], p_te),
                         "test_average_precision": average_precision(y[te], p_te),
                         "at_0_5": at_threshold(y[te], p_te, 0.5),
                         "trees": None if isinstance(m, tuple) else int(m.best_iteration_)}
        print(f"  {name:34s} AUC {results[name]['test_roc_auc']:.3f}  AP {results[name]['test_average_precision']:.3f}  "
              f"recall@0.5 {results[name]['at_0_5']['recall']:.2f}  precision@0.5 {results[name]['at_0_5']['precision']:.2f}")

    # ------------------------------------------------------------- business rule: pick model and threshold by profit on validation
    def profit_at(yv, pv, t):
        sel = pv >= t
        return float(((yv == 1) & sel).sum() * SUCCESS_RATE * VALUE_SAVED - sel.sum() * CONTACT_COST)
    grid = np.round(np.arange(0.02, 0.96, 0.01), 2)
    for name, (pv, pt) in probs.items():
        vp = [profit_at(y[va], pv, t) for t in grid]
        t_opt = float(grid[int(np.argmax(vp))])
        results[name].update({"val_best_threshold": t_opt, "val_profit": max(vp), "test_profit": profit_at(y[te], pt, t_opt),
                              "mean_predicted_churn": float(pt.mean())})
        print(f"  {name:34s} val-optimal threshold {t_opt:.2f}  val profit {max(vp):>9,.0f}  test profit {results[name]['test_profit']:>9,.0f}"
              f"  mean predicted churn {pt.mean():.1%}")
    lgb_names = [n for n in models if n.startswith("LightGBM")]
    # rebalancing distorts probabilities; among models within 2% of the best validation profit,
    # keep the one whose average predicted churn is closest to the real churn rate (best calibrated)
    top_profit = max(results[n]["val_profit"] for n in lgb_names)
    contenders = [n for n in lgb_names if results[n]["val_profit"] >= 0.98 * top_profit]
    best_name = min(contenders, key=lambda n: abs(probs[n][0].mean() - y[tr].mean()))
    best = models[best_name]
    p_va, p_te = probs[best_name]
    t_star = results[best_name]["val_best_threshold"]
    print(f"Within 2% of the best validation profit: {contenders}; most calibrated: {best_name}")
    breakeven = CONTACT_COST / (SUCCESS_RATE * VALUE_SAVED)
    op = at_threshold(y[te], p_te, t_star)
    op_05 = at_threshold(y[te], p_te, 0.5)
    profit_star, profit_05 = profit_at(y[te], p_te, t_star), profit_at(y[te], p_te, 0.5)
    profit_all = float(y[te].sum() * SUCCESS_RATE * VALUE_SAVED - len(te) * CONTACT_COST)
    profit_random_same = float(op["tp"] + op["fp"]) * (y[te].mean() * SUCCESS_RATE * VALUE_SAVED - CONTACT_COST)
    share, captured = gains(y[te], p_te)
    top20 = float(np.interp(0.2, share, captured))
    print(f"Profit-optimal threshold {t_star} (break-even probability {breakeven:.3f}); test profit {profit_star:,.0f} "
          f"vs {profit_05:,.0f} at 0.5 and {profit_all:,.0f} contacting everyone")

    # ------------------------------------------------------------- explainability
    booster = best.booster_
    contrib = booster.predict(Xa[te], pred_contrib=True)[:, :-1]
    shap_mean = np.abs(contrib).mean(0)
    perm_mean, perm_std = permutation_importance(lambda Z: best.predict_proba(Z)[:, 1], Xa[te], y[te], roc_auc_score,
                                                 n_repeats=10)
    gain = booster.feature_importance("gain")
    gain = gain / gain.sum()

    # ------------------------------------------------------------- chart 1: 3D churn risk surface
    ages = np.arange(18, 81, 2)
    bals = np.linspace(0, 220_000, 34)
    ref = Xa[va][:800]
    def surface(active):
        Zs = np.zeros((len(bals), len(ages)))
        for i, b in enumerate(bals):
            block = np.repeat(ref, len(ages), axis=0)
            block[:, cols.index("Age")] = np.tile(ages, len(ref))
            block[:, cols.index("Balance")] = b
            block[:, cols.index("ZeroBalance")] = float(b == 0)
            block[:, cols.index("BalanceToSalary")] = b / block[:, cols.index("EstimatedSalary")]
            block[:, cols.index("IsActiveMember")] = active
            Zs[i] = best.predict_proba(block)[:, 1].reshape(len(ref), len(ages)).mean(0)
        return Zs
    Z_in, Z_act = surface(0), surface(1)
    fig3d = go.Figure()
    for Zs, nm, scale, op_ in [(Z_in, "Inactive members", [[0, "#3b1f12"], [0.5, ORANGE], [1, "#f6c3a6"]], 0.72),
                               (Z_act, "Active members", [[0, "#10233b"], [0.5, BLUE], [1, "#b9d6f8"]], 1.0)]:
        fig3d.add_trace(go.Surface(x=ages, y=bals / 1000, z=100 * Zs, name=nm, colorscale=scale, showscale=False, opacity=op_,
                                   cmin=0, cmax=100 * max(Z_in.max(), Z_act.max()), showlegend=True,
                                   contours=dict(z=dict(show=True, usecolormap=True, project_z=False, width=1)),
                                   hovertemplate=f"<b>{nm}</b><br>Age %{{x}}<br>Balance %{{y:.0f}}k<br>Churn probability %{{z:.1f}}%<extra></extra>"))
    axis = dict(backgroundcolor=BG, gridcolor=GRID, zerolinecolor=GRID, color=TEXT2)
    fig3d.update_layout(**plotly_layout(
        title=dict(text="Churn risk landscape learned by LightGBM: age x balance, active (blue) vs inactive (orange) members", x=0.02),
        scene=dict(xaxis=dict(title="Age", **axis), yaxis=dict(title="Balance (thousand USD)", **axis),
                   zaxis=dict(title="Churn probability (%)", **axis), aspectratio=dict(x=1.3, y=1, z=0.75),
                   camera=dict(eye=dict(x=1.75, y=-1.25, z=0.45))),
        legend=dict(font=dict(color=TEXT2), x=0.01, y=0.92), height=760))
    fig3d.write_html(OUT / "01_churn_risk_3d.html", include_plotlyjs="cdn")
    try:
        fig3d.write_image(OUT / "01_churn_risk_3d.png", width=1600, height=900, scale=2)
    except Exception as exc:  # noqa: BLE001
        print(f"  (static 3D export skipped: {exc.__class__.__name__})")

    # ------------------------------------------------------------- chart 2: who churns (segments)
    seg = df.assign(AgeBand=pd.cut(df["Age"], [17, 30, 40, 50, 60, 100], labels=["18-30", "31-40", "41-50", "51-60", "61+"]))
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1.4, 1.1, 1.1, 0.8]})
    for ax, (col, lab) in zip(axes, [("AgeBand", "Age"), ("NumOfProducts", "Number of products"),
                                     ("Geography", "Country"), ("IsActiveMember", "Active member")]):
        g = seg.groupby(col, observed=True)["Exited"].agg(["mean", "size"])
        names = [("No" if i == 0 else "Yes") if col == "IsActiveMember" else str(i) for i in g.index]
        bars = ax.bar(names, 100 * g["mean"], color=[RED if v > 0.3 else AMBER if v > y.mean() else AQUA for v in g["mean"]], width=0.62)
        for b_, (v, n) in zip(bars, zip(g["mean"], g["size"])):
            ax.text(b_.get_x() + b_.get_width() / 2, 100 * v + 1.5, f"{100 * v:.0f}%", ha="center", color=TEXT, fontsize=10)
            ax.text(b_.get_x() + b_.get_width() / 2, -7, f"n={n:,}", ha="center", color=MUTED, fontsize=8)
        ax.axhline(100 * y.mean(), color=TEXT2, lw=1, ls="--")
        ax.set_ylim(-10, 110 if col == "NumOfProducts" else 60); ax.set_xlabel(lab); ax.grid(axis="y", lw=0.5)
        ax.tick_params(axis="x", colors=TEXT, length=0)
    axes[0].set_ylabel("Churn rate (%)")
    fig.suptitle(f"Who leaves the bank: churn rate by segment (dashed line = average {100 * y.mean():.1f}%)", x=0.01, ha="left",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "02_who_churns.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 3: imbalance strategies
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})
    pal = [MUTED, BLUE, AMBER, PINK, VIOLET]
    for (name, (_, pt)), c in zip(probs.items(), pal):
        pr, rc, _ = pr_curve(y[te], pt)
        axes[0].plot(rc, pr, color=c, lw=2.4 if name == best_name else 1.6, label=f"{name} (AP {results[name]['test_average_precision']:.3f})")
    axes[0].axhline(y[te].mean(), color=GRID, ls="--", lw=1, label=f"Random ({y[te].mean():.3f})")
    axes[0].set_xlabel("Recall (share of churners found)"); axes[0].set_ylabel("Precision (share of alerts that are real churners)")
    axes[0].legend(frameon=False, labelcolor=TEXT2, fontsize=8.5, loc="upper right"); axes[0].grid(lw=0.5)
    title(axes[0], "Precision-recall curves on the test set")
    names = list(results)[::-1]
    yy = np.arange(len(names))
    rec = [100 * results[n]["at_0_5"]["recall"] for n in names]
    pre = [100 * results[n]["at_0_5"]["precision"] for n in names]
    axes[1].barh(yy + 0.2, rec, height=0.38, color=BLUE, label="Recall at threshold 0.5")
    axes[1].barh(yy - 0.2, pre, height=0.38, color=AMBER, label="Precision at threshold 0.5")
    for i, (r, p_) in enumerate(zip(rec, pre)):
        axes[1].text(r + 1, i + 0.2, f"{r:.0f}%", va="center", color=TEXT, fontsize=9)
        axes[1].text(p_ + 1, i - 0.2, f"{p_:.0f}%", va="center", color=TEXT, fontsize=9)
    axes[1].set_yticks(yy, [f"{n}\nAUC {results[n]['test_roc_auc']:.3f}" for n in names], fontsize=9)
    axes[1].tick_params(axis="y", colors=TEXT, length=0); axes[1].set_xlim(0, 105); axes[1].grid(axis="x", lw=0.5)
    axes[1].legend(frameon=False, labelcolor=TEXT2, loc="lower right", fontsize=9)
    title(axes[1], "Rebalancing moves the trade-off, not the ranking")
    fig.tight_layout()
    fig.savefig(OUT / "03_imbalance_strategies.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 4: importance + SHAP
    order = np.argsort(shap_mean)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), gridspec_kw={"width_ratios": [1, 1.35]})
    yy = np.arange(len(cols))
    axes[0].barh(yy + 0.2, perm_mean[order], xerr=perm_std[order], height=0.38, color=BLUE, ecolor=MUTED, label="Permutation (drop in AUC)")
    axes[0].barh(yy - 0.2, gain[order] * perm_mean.max() / gain.max(), height=0.38, color=MUTED, label="LightGBM gain (rescaled)")
    axes[0].set_yticks(yy, [cols[i] for i in order]); axes[0].tick_params(axis="y", colors=TEXT, length=0)
    axes[0].legend(frameon=False, labelcolor=TEXT2, loc="lower right"); axes[0].grid(axis="x", lw=0.5)
    axes[0].set_xlabel("Importance")
    title(axes[0], "Which variables matter")
    top = order[-9:]
    cmap = LinearSegmentedColormap.from_list("v", [BLUE, "#8a8980", ORANGE])
    rng = np.random.default_rng(0)
    for row, j in enumerate(top):
        v = Xa[te][:, j]
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        ranks = np.clip((v - lo) / max(hi - lo, 1e-12), 0, 1)
        axes[1].scatter(contrib[:, j], row + rng.normal(0, 0.12, len(v)), c=ranks, cmap=cmap, s=6, alpha=0.75, linewidths=0)
    axes[1].axvline(0, color=MUTED, lw=1)
    axes[1].set_yticks(range(len(top)), [cols[i] for i in top]); axes[1].tick_params(axis="y", colors=TEXT, length=0)
    axes[1].set_xlabel("SHAP value: effect on the churn log-odds (right = more likely to leave)"); axes[1].grid(axis="x", lw=0.5)
    title(axes[1], "SHAP: how each value pushes each customer (orange = high value, blue = low)")
    fig.tight_layout()
    fig.savefig(OUT / "04_feature_importance_shap.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 5: business decision
    k, prof = profit_curve(y[te], p_te, VALUE_SAVED, CONTACT_COST, SUCCESS_RATE)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6))
    axes[0].plot(100 * share, 100 * captured, color=BLUE, lw=2.4, label="LightGBM")
    axes[0].plot([0, 100], [0, 100], color=GRID, ls="--", lw=1, label="Random selection")
    axes[0].scatter([20], [100 * top20], color=AQUA, s=60, zorder=5)
    axes[0].annotate(f"Top 20% riskiest customers\ncontain {100 * top20:.0f}% of all churners", (20, 100 * top20),
                     xytext=(28, 100 * top20 - 22), color=AQUA, arrowprops=dict(arrowstyle="-", color=AQUA))
    axes[0].set_xlabel("Customers contacted, ordered by predicted risk (%)"); axes[0].set_ylabel("Churners reached (%)")
    axes[0].legend(frameon=False, labelcolor=TEXT2, loc="lower right"); axes[0].grid(lw=0.5)
    title(axes[0], "Cumulative gains on the test set")
    axes[1].plot(100 * k / len(te), prof / 1000, color=AQUA, lw=2.4)
    n_star = op["tp"] + op["fp"]; n_05 = op_05["tp"] + op_05["fp"]
    axes[1].scatter([100 * n_star / len(te)], [profit_star / 1000], color=AQUA, s=70, zorder=5)
    axes[1].annotate(f"Profit-optimal threshold {t_star:.2f}\ncontact {n_star} customers: {profit_star / 1000:,.1f}k USD",
                     (100 * n_star / len(te), profit_star / 1000), xytext=(44, profit_star / 1000 * 0.22), color=AQUA,
                     arrowprops=dict(arrowstyle="-", color=AQUA))
    axes[1].scatter([100 * n_05 / len(te)], [profit_05 / 1000], color=AMBER, s=60, zorder=5)
    axes[1].annotate(f"Default threshold 0.5: {profit_05 / 1000:,.1f}k", (100 * n_05 / len(te), profit_05 / 1000),
                     xytext=(14, profit_05 / 1000 * 0.45), color=AMBER, arrowprops=dict(arrowstyle="-", color=AMBER))
    axes[1].axhline(0, color=MUTED, lw=1)
    axes[1].set_xlabel("Customers contacted, ordered by predicted risk (%)"); axes[1].set_ylabel("Campaign profit (thousand USD)")
    axes[1].grid(lw=0.5)
    title(axes[1], f"Retention campaign profit ({len(te):,} test customers)")
    fig.text(0.01, 0.01, f"Assumptions: saving a churner is worth {VALUE_SAVED:,} USD, the offer costs {CONTACT_COST} USD per customer and works "
             f"for {SUCCESS_RATE:.0%} of contacted churners. All three can be changed live in the dashboard.", color=TEXT2, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "05_business_decision.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 6: confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, o, lab in [(axes[0], op_05, "Default threshold 0.5"), (axes[1], op, f"Profit-optimal threshold {t_star:.2f}")]:
        cm = np.array([[o["tn"], o["fp"]], [o["fn"], o["tp"]]])
        ax.imshow(np.eye(2), cmap=LinearSegmentedColormap.from_list("c", ["#3d2420", "#183a2f"]), vmin=0, vmax=1)
        names_ = [["Stays, not contacted", "Stays, contacted (wasted offer)"], ["Leaves, missed", "Leaves, contacted"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i - 0.08, f"{cm[i, j]:,}", ha="center", va="center", fontsize=24, fontweight="bold", color=AQUA if i == j else RED)
                ax.text(j, i + 0.22, names_[i][j], ha="center", va="center", fontsize=9, color=TEXT2)
        ax.set_xticks([0, 1], ["Predicted stays", "Predicted leaves"]); ax.set_yticks([0, 1], ["Stays", "Leaves"])
        ax.tick_params(colors=TEXT, length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        title(ax, f"{lab}: recall {100 * o['recall']:.0f}%, precision {100 * o['precision']:.0f}%")
    fig.tight_layout()
    fig.savefig(OUT / "06_confusion_matrices.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- live model + data for the dashboard
    dump = booster.dump_model()
    def compact(node):
        if "leaf_value" in node:
            return round(node["leaf_value"], 6)
        return [node["split_feature"], round(node["threshold"], 6), int(node.get("default_left", True)),
                compact(node["left_child"]), compact(node["right_child"])]
    trees = [compact(t["tree_structure"]) for t in dump["tree_info"]]
    live = {"features": cols, "trees": trees, "threshold": t_star,
            "test": {"p": np.round(p_te, 4).tolist(), "y": y[te].tolist()},
            "defaults": {"value": VALUE_SAVED, "cost": CONTACT_COST, "success": SUCCESS_RATE}}
    (OUT / "model.js").write_text("window.CHURN_MODEL=" + json.dumps(live, separators=(",", ":")) + ";", encoding="utf-8")

    # ------------------------------------------------------------- metrics
    rnd = lambda d: {k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in d.items()}
    metrics = {
        "data": {"customers": len(df), "churn_rate": round(float(y.mean()), 4), "train": len(tr), "validation": len(va), "test": len(te),
                 "smote_training_rows": int(len(y_smote))},
        "models": {n: {"val_roc_auc": round(r["val_roc_auc"], 4), "test_roc_auc": round(r["test_roc_auc"], 4),
                       "test_average_precision": round(r["test_average_precision"], 4), "trees": r["trees"],
                       "val_optimal_threshold": r["val_best_threshold"], "val_profit": r["val_profit"], "test_profit": r["test_profit"],
                       "mean_predicted_churn_test": round(r["mean_predicted_churn"], 4),
                       "at_threshold_0_5": rnd(r["at_0_5"])} for n, r in results.items()},
        "selected_model": best_name, "selection_rule": "highest validation profit within 2%, then best calibrated", "contenders": contenders,
        "business": {"assumptions": {"value_saved_usd": VALUE_SAVED, "contact_cost_usd": CONTACT_COST, "success_rate": SUCCESS_RATE},
                     "break_even_probability": round(breakeven, 4), "profit_optimal_threshold": t_star,
                     "test_at_optimal": rnd(op), "test_at_0_5": rnd(op_05), "test_profit_optimal": profit_star,
                     "test_profit_at_0_5": profit_05, "test_profit_contact_everyone": profit_all,
                     "test_profit_random_same_size": round(profit_random_same, 1), "top_20pct_captures": round(top20, 4)},
        "importance": {c: {"mean_abs_shap": round(float(s), 4), "permutation_auc_drop": round(float(pm), 4), "gain_share": round(float(g), 4)}
                       for c, s, pm, g in zip(cols, shap_mean, perm_mean, gain)},
        "segments": {"churn_by_products": df.groupby("NumOfProducts")["Exited"].mean().round(4).to_dict(),
                     "churn_by_country": df.groupby("Geography")["Exited"].mean().round(4).to_dict(),
                     "churn_by_active": df.groupby("IsActiveMember")["Exited"].mean().round(4).to_dict()},
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k_: metrics[k_] for k_ in ("selected_model", "business")}, indent=1))

    # ------------------------------------------------------------- dashboard
    kpis = {"Test ROC AUC": f"{results[best_name]['test_roc_auc']:.3f}",
            "Churners found (optimal)": f"{100 * op['recall']:.0f}%",
            "Campaign profit (test)": f"{profit_star / 1000:,.1f}k USD",
            "Top 20% captures": f"{100 * top20:.0f}% of churners"}
    kpi_html = "".join(f'<div class="kpi"><span>{k_}</span><b>{v}</b></div>' for k_, v in kpis.items())
    sec = lambda t, n_, f: f'<section><h2>{t}</h2><p class="note">{n_}</p>{f.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})}</section>'
    img = lambda t, n_, src: f'<section><h2>{t}</h2><p class="note">{n_}</p><img src="outputs/{src}" alt="{t}"></section>'
    sliders = [("Age", 18, 92, 1, 45), ("CreditScore", 350, 850, 5, 650), ("Balance", 0, 250000, 1000, 120000),
               ("EstimatedSalary", 1000, 200000, 1000, 100000), ("Tenure", 0, 10, 1, 4), ("NumOfProducts", 1, 4, 1, 1)]
    slider_html = "".join(f'<label>{n}<input type="range" id="{n}" min="{a}" max="{b}" step="{s}" value="{v}"><output id="{n}_o"></output></label>'
                          for n, a, b, s, v in sliders)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Customer Churn Prediction</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><script src="outputs/model.js"></script>
<style>:root{{--bg:{BG};--card:{CARD};--text:{TEXT};--text2:{TEXT2};--muted:{MUTED};--line:{GRID}}}*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font-family:Inter,"Segoe UI",Arial,sans-serif}}
header,.kpis,footer{{max-width:1240px;margin:auto;padding:0 32px}}header{{padding-top:48px}}h1{{font-size:40px;margin:0 0 8px}}
header p{{color:var(--text2);max-width:860px;line-height:1.55;margin:0}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:24px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px}}
.kpi span{{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.8px}}.kpi b{{display:block;font-size:24px;margin-top:6px}}
section{{max-width:1240px;margin:28px auto;padding:20px 24px;background:var(--card);border:1px solid var(--line);border-radius:14px}}
section img{{width:100%;border-radius:8px}}h2{{margin:4px 0 6px;font-size:20px}}.note{{color:var(--text2);margin:0 0 10px;font-size:14px}}
.grid2{{display:grid;grid-template-columns:1.1fr 1fr;gap:24px}}label{{display:grid;grid-template-columns:130px 1fr 80px;align-items:center;gap:10px;color:var(--text2);font-size:13px;margin:7px 0}}
input[type=range]{{width:100%;accent-color:{BLUE}}}output{{color:var(--text);text-align:right;font-variant-numeric:tabular-nums}}
select{{background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:5px}}.toggles{{display:flex;flex-wrap:wrap;gap:14px;margin:12px 0;color:var(--text2);font-size:13px}}
.risk{{font-size:56px;font-weight:800;margin:6px 0}}.band{{display:inline-block;padding:4px 12px;border-radius:999px;font-size:13px;font-weight:600}}
.bar{{height:12px;border-radius:6px;background:linear-gradient(90deg,{AQUA},{AMBER},{RED});position:relative;margin:16px 0 4px}}.pin{{position:absolute;top:-6px;width:4px;height:24px;background:var(--text);border-radius:2px}}
footer{{color:var(--muted);font-size:13px;margin:24px auto 48px}}a{{color:#7fb0f0}}
@media(max-width:760px){{.grid2{{grid-template-columns:1fr}}header,.kpis,footer{{padding:0 16px}}section{{margin:16px;padding:12px}}h1{{font-size:30px}}label{{grid-template-columns:100px 1fr 64px}}}}</style></head><body>
<header><h1>Customer Churn Prediction</h1><p>Which bank customers are about to leave, why, and who should receive a retention offer.
LightGBM trained on {len(tr):,} customers, four ways of handling class imbalance compared (including SMOTE written from scratch), decisions explained with SHAP and the alert threshold chosen by profit, not by default.</p></header>
<div class="kpis">{kpi_html}</div>
<section><h2>Churn risk calculator</h2><p class="note">The exact LightGBM model ({len(trees)} trees) runs in your browser. Move the sliders to build a customer profile.</p>
<div class="grid2"><div>{slider_html}
<div class="toggles"><span>Country <select id="Country"><option>France</option><option selected>Germany</option><option>Spain</option></select></span>
<span>Gender <select id="Gender"><option>Female</option><option>Male</option></select></span>
<span><input type="checkbox" id="IsActiveMember"> Active member</span><span><input type="checkbox" id="HasCrCard" checked> Has credit card</span></div></div>
<div><div class="note">Probability that this customer leaves</div><div class="risk" id="risk"></div><span class="band" id="band"></span>
<div class="bar"><div class="pin" id="pin"></div></div><div style="display:flex;justify-content:space-between;color:var(--muted);font-size:12px"><span>0%</span><span>100%</span></div>
<p class="note" id="advice" style="margin-top:16px"></p></div></div></section>
<section><h2>Campaign simulator</h2><p class="note">Change the business assumptions: the profit curve on the {len(te):,} test customers and the best number of customers to contact update instantly.</p>
<label>Value of saving a churner (USD)<input type="range" id="v" min="200" max="5000" step="100" value="{VALUE_SAVED}"><output id="v_o"></output></label>
<label>Cost per contact (USD)<input type="range" id="c" min="5" max="300" step="5" value="{CONTACT_COST}"><output id="c_o"></output></label>
<label>Offer success rate<input type="range" id="s" min="0.05" max="0.8" step="0.05" value="{SUCCESS_RATE}"><output id="s_o"></output></label>
<div id="profit" style="height:420px"></div><p class="note" id="profit_txt"></p></section>
{sec("Churn risk landscape in 3D", "Drag to rotate. Average churn probability predicted for every combination of age and balance; inactive members (orange) sit far above active ones.", fig3d)}
{img("Who churns", "Raw churn rates by segment before any model.", "02_who_churns.png")}
{img("Handling the imbalance", "Only 20% of customers leave. Class weights, oversampling and SMOTE are compared on the same test set.", "03_imbalance_strategies.png")}
{img("Feature importance and SHAP", "Three ways to explain the model agree on the drivers: age, number of products, activity and Germany.", "04_feature_importance_shap.png")}
{img("From model to decision", "Gains and profit of a retention campaign that contacts customers in order of risk.", "05_business_decision.png")}
{img("Confusion matrices", "The profit-optimal threshold finds many more churners at the cost of more offers.", "06_confusion_matrices.png")}
<footer>Tomás Alarcón &middot; <a href="https://github.com/tomasalarcon10/ml-from-scratch">Source code</a></footer>
<script>
const M=window.CHURN_MODEL,F=M.features;
function walk(n,x){{while(Array.isArray(n)){{const v=x[n[0]];n=(isNaN(v)?n[2]:v<=n[1])?n[3]:n[4];}}return n;}}
function predict(x){{let z=0;for(const t of M.trees)z+=walk(t,x);return 1/(1+Math.exp(-z));}}
function profile(){{const g=id=>+document.getElementById(id).value,ctry=document.getElementById("Country").value,b=g("Balance"),s=g("EstimatedSalary");
const o={{CreditScore:g("CreditScore"),Germany:+(ctry==="Germany"),Spain:+(ctry==="Spain"),Male:+(document.getElementById("Gender").value==="Male"),Age:g("Age"),
Tenure:g("Tenure"),Balance:b,NumOfProducts:g("NumOfProducts"),HasCrCard:+document.getElementById("HasCrCard").checked,IsActiveMember:+document.getElementById("IsActiveMember").checked,
EstimatedSalary:s,ZeroBalance:+(b===0),BalanceToSalary:b/s}};return F.map(f=>o[f]);}}
function calc(){{for(const id of ["Age","CreditScore","Balance","EstimatedSalary","Tenure","NumOfProducts"]){{const v=+document.getElementById(id).value;
document.getElementById(id+"_o").textContent=(id==="Balance"||id==="EstimatedSalary")?v.toLocaleString("en-US"):v;}}
const p=predict(profile()),pct=(100*p).toFixed(1);document.getElementById("risk").textContent=pct+"%";document.getElementById("pin").style.left=`calc(${{pct}}% - 2px)`;
const band=document.getElementById("band");const hi=p>=0.5,mid=p>=M.threshold;band.textContent=hi?"High risk":mid?"At risk":"Low risk";
band.style.background=hi?"{RED}33":mid?"{AMBER}33":"{AQUA}33";band.style.color=hi?"{RED}":mid?"{AMBER}":"{AQUA}";
document.getElementById("advice").textContent=mid?`Above the profit-optimal threshold of ${{(100*M.threshold).toFixed(0)}}%: include this customer in the retention campaign.`:`Below the profit-optimal threshold of ${{(100*M.threshold).toFixed(0)}}%: an offer would most likely be wasted.`;}}
document.querySelectorAll("input,select").forEach(e=>e.addEventListener("input",()=>{{calc();sim();}}));
const idx=M.test.p.map((p,i)=>i).sort((a,b)=>M.test.p[b]-M.test.p[a]);
function sim(){{const v=+document.getElementById("v").value,c=+document.getElementById("c").value,s=+document.getElementById("s").value;
document.getElementById("v_o").textContent=v.toLocaleString("en-US");document.getElementById("c_o").textContent=c;document.getElementById("s_o").textContent=Math.round(100*s)+"%";
const xs=[0],ys=[0];let caught=0;idx.forEach((i,k)=>{{caught+=M.test.y[i];xs.push(k+1);ys.push(caught*s*v-(k+1)*c);}});
let best=0;ys.forEach((y,k)=>{{if(y>ys[best])best=k;}});const everyone=ys[ys.length-1];
Plotly.react("profit",[{{x:xs,y:ys,mode:"lines",line:{{color:"{AQUA}",width:3}},hovertemplate:"Contact top %{{x}} customers<br>Profit %{{y:,.0f}} USD<extra></extra>"}},
{{x:[best],y:[ys[best]],mode:"markers",marker:{{size:12,color:"{AQUA}",line:{{color:"{TEXT}",width:2}}}},hoverinfo:"skip"}}],
{{paper_bgcolor:"{CARD}",plot_bgcolor:"{CARD}",font:{{color:"{TEXT2}",family:"Inter, Segoe UI, Arial"}},showlegend:false,margin:{{l:70,r:20,t:20,b:50}},
xaxis:{{title:{{text:"Customers contacted, riskiest first"}},gridcolor:"{GRID}",zeroline:false}},yaxis:{{title:{{text:"Campaign profit (USD)"}},gridcolor:"{GRID}",zerolinecolor:"{MUTED}"}}}},{{displaylogo:false,responsive:true}});
document.getElementById("profit_txt").innerHTML=`Best plan in hindsight on these test customers: contact the <b>${{best}}</b> riskiest customers (${{(100*best/idx.length).toFixed(0)}}%) for a profit of <b>${{Math.round(ys[best]).toLocaleString("en-US")}} USD</b>. Contacting everyone would give ${{Math.round(everyone).toLocaleString("en-US")}} USD.`;}}
calc();if(window.Plotly)sim();
</script></body></html>"""
    (ROOT / "dashboard.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
