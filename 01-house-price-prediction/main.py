"""
House Price Prediction (Ames, Iowa)
Run:  python main.py
Cleans the data, engineers features, trains linear models implemented from scratch,
benchmarks them against scikit-learn and builds every chart plus the dashboard.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from src.features import load_raw, clean, engineer, design_matrix
from src.models import (StandardScalerScratch, LinearRegressionNormalEquation, RidgeClosedForm,
                        LinearRegressionGD, rmse, r2, kfold_indices, cross_validate)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

BG, CARD, TEXT, TEXT2, MUTED, GRID = "#1a1a19", "#222220", "#f0efec", "#c3c2b7", "#8a8980", "#3a3936"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET, RED = "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9", "#e66767"

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


def sklearn_cv(model, X, y, k=5):
    scores = []
    for tr, te in kfold_indices(len(X), k):
        sc = StandardScalerScratch().fit(X[tr])
        m = model.fit(sc.transform(X[tr]), y[tr])
        scores.append(rmse(y[te], m.predict(sc.transform(X[te]))))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    # ------------------------------------------------------------------ data
    raw = load_raw()
    df, clean_log = clean(raw)
    fe = engineer(df)
    X, y, skewed = design_matrix(fe)
    names, Xv = X.columns.to_numpy(), X.values
    print(f"Rows {len(Xv)}, features {Xv.shape[1]}")

    # ------------------------------------------------------------------ alpha selection (Ridge, from scratch)
    alphas = np.logspace(-1, 3.5, 25)
    curve = [cross_validate(lambda a=a: RidgeClosedForm(a), Xv, y) for a in alphas]
    best_alpha = float(alphas[int(np.argmin([c[0] for c in curve]))])

    # ------------------------------------------------------------------ model comparison (same 5 folds for every model)
    base_cols = ["Gr Liv Area"]
    results = {
        "Baseline: living area only": cross_validate(LinearRegressionNormalEquation, np.log(fe[base_cols].values.astype(float)), y),
        "OLS, normal equation (scratch)": cross_validate(LinearRegressionNormalEquation, Xv, y),
        f"Ridge closed form, alpha={best_alpha:.0f} (scratch)": cross_validate(lambda: RidgeClosedForm(best_alpha), Xv, y),
        "Ridge, gradient descent (scratch)": cross_validate(lambda: LinearRegressionGD(0.01, 3000, best_alpha), Xv, y),
        "scikit-learn LinearRegression": sklearn_cv(LinearRegression(), Xv, y),
        f"scikit-learn Ridge, alpha={best_alpha:.0f}": sklearn_cv(Ridge(alpha=best_alpha), Xv, y),
        "scikit-learn Lasso, alpha=0.0005": sklearn_cv(Lasso(alpha=0.0005, max_iter=50000), Xv, y),
    }
    for k, v in results.items():
        print(f"  {k:48s} RMSE(log) {v[0]:.4f} +/- {v[1]:.4f}")

    # ------------------------------------------------------------------ holdout evaluation
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(Xv))
    cut = int(0.8 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    sc = StandardScalerScratch().fit(Xv[tr])
    ridge = RidgeClosedForm(best_alpha).fit(sc.transform(Xv[tr]), y[tr])
    gd = LinearRegressionGD(0.01, 3000, best_alpha).fit(sc.transform(Xv[tr]), y[tr])
    pred_log = ridge.predict(sc.transform(Xv[te]))
    price, pred = np.exp(y[te]), np.exp(pred_log)
    hold = {
        "rmse_log": rmse(y[te], pred_log), "r2_log": r2(y[te], pred_log),
        "mae_usd": float(np.mean(np.abs(price - pred))),
        "mape_pct": float(100 * np.mean(np.abs(price - pred) / price)),
        "median_ape_pct": float(100 * np.median(np.abs(price - pred) / price)),
        "gd_vs_closed_form_max_coef_diff": float(np.max(np.abs(gd.coef_ - ridge.coef_))),
    }
    print(json.dumps(hold, indent=2))

    # ------------------------------------------------------------------ chart 1: model comparison
    fig, ax = plt.subplots(figsize=(12, 5.6))
    labels = list(results)[::-1]
    vals = [results[l][0] for l in labels]
    errs = [results[l][1] for l in labels]
    colors = [MUTED if "Baseline" in l else (BLUE if "scratch" in l else VIOLET) for l in labels]
    ax.barh(labels, vals, xerr=errs, color=colors, height=0.62, error_kw=dict(ecolor=TEXT2, lw=1, capsize=3))
    for i, v in enumerate(vals):
        ax.text(v + max(errs) + 0.004, i, f"{v:.3f}", va="center", color=TEXT2, fontsize=9.5)
    ax.set_xlabel("5-fold cross-validated RMSE on log(price)   (lower is better)")
    ax.tick_params(axis="y", colors=TEXT, labelsize=10)
    ax.set_axisbelow(True); ax.grid(axis="x", lw=0.5)
    fig.suptitle("Models built from scratch match scikit-learn", x=0.02, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.02, 0.9, "Blue: implemented with NumPy. Violet: scikit-learn on the same folds. "
                        "Gray: one-variable baseline. Bars show mean, whiskers one standard deviation.", color=TEXT2, fontsize=9.5)
    fig.subplots_adjust(left=0.3, right=0.97, top=0.84, bottom=0.12)
    fig.savefig(OUT / "02_model_comparison.png", dpi=180); plt.close(fig)

    # ------------------------------------------------------------------ chart 2: regularization curve + gradient descent convergence
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    m = np.array([c[0] for c in curve]); s = np.array([c[1] for c in curve])
    axes[0].fill_between(alphas, m - s, m + s, color=BLUE, alpha=0.18, lw=0)
    axes[0].plot(alphas, m, color=BLUE, lw=2.2, marker="o", ms=4)
    axes[0].axvline(best_alpha, color=TEXT2, ls="--", lw=1)
    axes[0].text(best_alpha * 1.15, m.max(), f"best alpha = {best_alpha:.0f}", color=TEXT2, fontsize=9.5, va="top")
    axes[0].set_xscale("log"); axes[0].set_xlabel("Ridge penalty alpha (log scale)"); axes[0].set_ylabel("CV RMSE on log(price)")
    axes[0].set_title("Choosing the penalty with cross-validation", loc="left", fontsize=12.5, fontweight="bold", color=TEXT)
    axes[0].grid(lw=0.5)
    axes[1].plot(np.arange(1, len(gd.loss_history_) + 1), gd.loss_history_, color=ORANGE, lw=2.2, label="Gradient descent")
    closed_mse = float(np.mean((ridge.predict(sc.transform(Xv[tr])) - y[tr]) ** 2))
    axes[1].axhline(closed_mse, color=AQUA, ls="--", lw=1.6, label="Closed-form solution")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("Epoch (log scale)"); axes[1].set_ylabel("Training MSE (log scale)")
    axes[1].set_title("Gradient descent converges to the closed form", loc="left", fontsize=12.5, fontweight="bold", color=TEXT)
    axes[1].legend(frameon=False, labelcolor=TEXT2); axes[1].grid(lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "03_regularization_and_convergence.png", dpi=180); plt.close(fig)

    # ------------------------------------------------------------------ chart 3: predicted vs actual + residuals
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    lim = [0, max(price.max(), pred.max()) * 1.05 / 1000]
    axes[0].scatter(price / 1000, pred / 1000, s=14, color=BLUE, alpha=0.65, edgecolor=BG, linewidth=0.4)
    axes[0].plot(lim, lim, color=TEXT2, lw=1, ls="--")
    axes[0].set_xlim(lim); axes[0].set_ylim(lim)
    axes[0].set_xlabel("Actual price (thousand US$)"); axes[0].set_ylabel("Predicted price (thousand US$)")
    axes[0].set_title(f"Holdout: R² {hold['r2_log']:.3f} on log price, median error {hold['median_ape_pct']:.1f}%",
                      loc="left", fontsize=12, fontweight="bold", color=TEXT)
    axes[0].grid(lw=0.5)
    resid = y[te] - pred_log
    axes[1].hist(resid, bins=40, color=VIOLET, alpha=0.85)
    axes[1].axvline(0, color=TEXT2, lw=1)
    axes[1].set_xlabel("Residual on log(price)   (0.10 is roughly a 10% error)"); axes[1].set_ylabel("Houses")
    axes[1].set_title("Residuals are centered and roughly symmetric", loc="left", fontsize=12, fontweight="bold", color=TEXT)
    axes[1].grid(axis="y", lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "04_predicted_vs_actual.png", dpi=180); plt.close(fig)

    # ------------------------------------------------------------------ chart 4: what drives the price
    full_sc = StandardScalerScratch().fit(Xv)
    full = RidgeClosedForm(best_alpha).fit(full_sc.transform(Xv), y)
    coef = pd.Series(full.coef_, index=names)
    top = pd.concat([coef.nlargest(12), coef.nsmallest(8)]).sort_values()
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.barh(top.index, 100 * top.values, color=[BLUE if v > 0 else RED for v in top.values], height=0.66)
    ax.axvline(0, color=TEXT2, lw=0.8)
    ax.tick_params(axis="y", colors=TEXT, labelsize=9.5)
    ax.set_xlabel("Approximate % change in price for a one-standard-deviation increase")
    ax.set_axisbelow(True); ax.grid(axis="x", lw=0.5)
    fig.suptitle("What moves the price of a house", x=0.02, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.02, 0.92, "Standardized Ridge coefficients on log(price). Blue raises the price, red lowers it.",
             color=TEXT2, fontsize=9.5)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.87, bottom=0.08)
    fig.savefig(OUT / "05_price_drivers.png", dpi=180); plt.close(fig)

    # ------------------------------------------------------------------ chart 5: 3D price surface (interactive)
    sf, q, p = fe["Total SF"].values, fe["Overall Qual"].values.astype(float), fe["SalePrice"].values
    A = np.c_[np.ones(len(sf)), np.log(sf), q]
    beta = np.linalg.lstsq(A, np.log(p), rcond=None)[0]
    gsf = np.linspace(np.percentile(sf, 1), np.percentile(sf, 99.5), 40)
    gq = np.linspace(1, 10, 40)
    GS, GQ = np.meshgrid(gsf, gq)
    Z = np.exp(beta[0] + beta[1] * np.log(GS) + beta[2] * GQ) / 1000
    nb_med = fe.groupby("Neighborhood")["SalePrice"].median()
    fig3d = go.Figure()
    fig3d.add_trace(go.Surface(x=GS, y=GQ, z=Z, opacity=0.55, showscale=False,
                               colorscale=[[0, "#1f3b5c"], [1, BLUE]], name="Model surface",
                               hovertemplate="Total SF %{x:,.0f}<br>Quality %{y:.1f}<br>Model: $%{z:,.0f}k<extra></extra>"))
    fig3d.add_trace(go.Scatter3d(
        x=sf, y=q + np.random.default_rng(0).uniform(-0.25, 0.25, len(q)), z=p / 1000, mode="markers",
        marker=dict(size=2.6, color=np.log(fe["Neighborhood"].map(nb_med)), colorscale=[[0, AQUA], [0.5, YELLOW], [1, ORANGE]],
                    opacity=0.85, colorbar=dict(title=dict(text="Neighborhood<br>median price", font=dict(color=TEXT2)),
                                                tickvals=np.log([120000, 180000, 300000]), ticktext=["$120k", "$180k", "$300k"],
                                                tickfont=dict(color=TEXT2), len=0.55, thickness=12)),
        customdata=np.stack([fe["Neighborhood"], fe["Year Built"]], axis=1), name="Houses",
        hovertemplate="<b>%{customdata[0]}</b>, built %{customdata[1]}<br>Total SF %{x:,.0f}<br>Quality %{y:.0f}<br>Price $%{z:,.0f}k<extra></extra>",
    ))
    axis = dict(backgroundcolor=BG, gridcolor=GRID, zerolinecolor=GRID, color=TEXT2)
    fig3d.update_layout(**plotly_layout(
        title=dict(text="Price as a function of size and quality: 2,925 sales and a fitted surface", x=0.02),
        scene=dict(xaxis=dict(title="Total square feet", **axis), yaxis=dict(title="Overall quality (1-10)", **axis),
                   zaxis=dict(title="Price (thousand US$)", **axis), camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
                   aspectratio=dict(x=1.4, y=1.1, z=0.85)),
        showlegend=False, height=760))
    fig3d.write_html(OUT / "01_price_surface_3d.html", include_plotlyjs="cdn")
    try:
        fig3d.write_image(OUT / "01_price_surface_3d.png", width=1600, height=900, scale=2)
    except Exception as exc:  # noqa: BLE001
        print(f"  (static 3D export skipped: {exc.__class__.__name__})")

    # ------------------------------------------------------------------ chart 6: neighborhoods (interactive)
    nb = fe.groupby("Neighborhood").agg(median=("SalePrice", "median"), n=("SalePrice", "size"),
                                        sf=("Total SF", "median")).sort_values("median")
    fig_nb = go.Figure(go.Bar(x=nb["median"] / 1000, y=nb.index, orientation="h", marker=dict(color=BLUE),
                              customdata=np.stack([nb["n"], nb["sf"]], axis=1),
                              hovertemplate="<b>%{y}</b><br>Median price $%{x:,.0f}k<br>Sales %{customdata[0]}<br>Median size %{customdata[1]:,.0f} sq ft<extra></extra>"))
    fig_nb.update_layout(**plotly_layout(title=dict(text="Median sale price by neighborhood (thousand US$)", x=0.02),
                                         xaxis=dict(gridcolor=GRID, color=TEXT2), yaxis=dict(color=TEXT), height=820))
    fig_nb.write_html(OUT / "06_neighborhoods.html", include_plotlyjs="cdn")

    # ------------------------------------------------------------------ metrics + dashboard
    metrics = {"cleaning": clean_log, "features": int(Xv.shape[1]), "skewed_log_transformed": len(skewed),
               "best_alpha": best_alpha, "cv": {k: {"rmse_log_mean": v[0], "rmse_log_std": v[1]} for k, v in results.items()},
               "holdout": hold}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    kpis = {"Sales analyzed": f"{len(Xv):,}", "Features engineered": f"{Xv.shape[1]}",
            "Holdout R² (log price)": f"{hold['r2_log']:.3f}", "Median price error": f"{hold['median_ape_pct']:.1f}%"}
    kpi_html = "".join(f'<div class="kpi"><span>{k}</span><b>{v}</b></div>' for k, v in kpis.items())
    sec = lambda t, n, f: f'<section><h2>{t}</h2><p class="note">{n}</p>{f.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})}</section>'
    img = lambda t, n, src: f'<section><h2>{t}</h2><p class="note">{n}</p><img src="outputs/{src}" alt="{t}"></section>'
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>House Price Prediction</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
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
<header><h1>House Price Prediction</h1><p>Linear regression built from scratch on 2,925 home sales in Ames, Iowa: data cleaning, feature engineering,
Ridge regularization and a benchmark against scikit-learn.</p></header><div class="kpis">{kpi_html}</div>
{sec("Price surface (3D)", "Drag to rotate. Each dot is a sale colored by how expensive its neighborhood is; the surface is a two-variable log-linear fit.", fig3d)}
{img("Model comparison", "Five-fold cross-validation, same folds for every model.", "02_model_comparison.png")}
{img("Regularization and convergence", "Left: choosing alpha. Right: gradient descent reaching the closed-form solution.", "03_regularization_and_convergence.png")}
{img("Holdout performance", "Ridge trained on 80% of the sales, evaluated on the remaining 20%.", "04_predicted_vs_actual.png")}
{img("Price drivers", "Standardized Ridge coefficients.", "05_price_drivers.png")}
{sec("Neighborhoods", "Hover a bar for sales count and median size.", fig_nb)}
<footer>Tomás Alarcón &middot; <a href="https://github.com/tomasalarcon10/ml-from-scratch">Source code</a></footer></body></html>"""
    (ROOT / "dashboard.html").write_text(page, encoding="utf-8")
    print("Done. Open dashboard.html")


if __name__ == "__main__":
    main()
