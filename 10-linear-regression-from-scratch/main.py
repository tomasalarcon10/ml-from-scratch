"""
Linear Regression from scratch: the normal equation and gradient descent reach the same line; the learning rate decides how.
Run:  python main.py      (under a minute on a laptop CPU)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import stats
from sklearn.linear_model import LinearRegression

from src.linreg import (add_intercept, normal_equation, simple_closed_form, loss, hessian, gradient_descent,
                        minibatch_gd, inference)

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


def plotly_layout(**kw):
    base = dict(paper_bgcolor=BG, plot_bgcolor=BG, font=dict(family="Inter, Segoe UI, Arial", color=TEXT, size=13),
                hoverlabel=dict(bgcolor=CARD, bordercolor=GRID, font=dict(color=TEXT)), margin=dict(l=20, r=20, t=70, b=20))
    base.update(kw)
    return base


def title(ax, text, size=12.5):
    ax.set_title(text, loc="left", fontsize=size, fontweight="bold", color=TEXT)


def main():
    df = pd.read_csv(ROOT / "data" / "ames_houses.csv")
    df = df[df["living_area_sqft"] <= 4000].dropna().reset_index(drop=True)  # De Cock (2011) recommends removing the 5 houses above 4,000 sq ft
    x = df["living_area_sqft"].to_numpy() / 1000          # thousand square feet
    y = df["sale_price_usd"].to_numpy() / 1000            # thousand USD
    X = add_intercept(x)
    n = len(y)
    print(f"{n:,} homes; price (k USD) vs living area (thousand sq ft)")

    # ------------------------------------------------------------- the same line, four ways
    b_normal = normal_equation(X, y)
    b_simple = simple_closed_form(x, y)
    b_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]
    sk = LinearRegression().fit(x[:, None], y)
    b_sk = np.array([sk.intercept_, sk.coef_[0]])
    H = hessian(X)
    eig = np.linalg.eigvalsh(H)
    lr_max = 2 / eig.max()
    lr_opt = 2 / (eig.min() + eig.max())
    kappa = eig.max() / eig.min()
    b_gd, path_gd, loss_gd = gradient_descent(X, y, lr=lr_opt, iters=20_000, tol=1e-12)
    J_star = loss(X, y, b_normal)
    print(f"normal equation: price = {b_normal[0]:.3f} + {b_normal[1]:.3f} * area;  GD ({len(path_gd) - 1} steps) "
          f"diff {np.abs(b_gd - b_normal).max():.1e}; Hessian eigenvalues {eig.round(4)}, condition {kappa:.1f}, lr limit {lr_max:.4f}")

    # ------------------------------------------------------------- learning rate experiments
    lr_list = [0.01, 0.1, 0.5, 0.58, 0.6]
    runs = {lr: gradient_descent(X, y, lr, 400) for lr in lr_list}
    scan = np.linspace(0.005, 0.62, 200)
    scan_loss = []
    for lr in scan:
        _, _, ls = gradient_descent(X, y, lr, 300)
        scan_loss.append(ls[-1] - J_star if np.isfinite(ls[-1]) and len(ls) == 301 else np.inf)
    scan_loss = np.array(scan_loss)
    steps_to = {}
    for lr in (0.01, 0.1, 0.3, lr_opt, 0.58):
        _, p_, ls_ = gradient_descent(X, y, lr, 200_000, tol=1e-9)
        steps_to[round(float(lr), 4)] = len(p_) - 1
    print("steps to converge:", steps_to)

    # ------------------------------------------------------------- feature scaling
    Xs_raw = add_intercept(df["living_area_sqft"].to_numpy())
    kappa_raw = float(np.linalg.cond(hessian(Xs_raw)))
    z = (x - x.mean()) / x.std()
    Xz = add_intercept(z)
    eig_z = np.linalg.eigvalsh(hessian(Xz))
    bz, path_z, loss_z = gradient_descent(Xz, y, lr=1.0, iters=50, tol=1e-12)
    b_back = np.array([bz[0] - bz[1] * x.mean() / x.std(), bz[1] / x.std()])
    _, path_raw, loss_raw = gradient_descent(Xs_raw, y, lr=1.9 / np.linalg.eigvalsh(hessian(Xs_raw)).max(), iters=2000)
    print(f"condition numbers: raw sq ft {kappa_raw:.2e}, thousand sq ft {kappa:.1f}, standardized {eig_z.max() / eig_z.min():.2f}; "
          f"standardized GD converges in {len(path_z) - 1} step(s)")

    # ------------------------------------------------------------- stochastic and mini-batch
    sgd_runs = {
        f"Batch (all {n:,} homes per step)": gradient_descent(X, y, lr_opt, 30)[2],
        "Mini-batch of 32": minibatch_gd(X, y, lr=0.1, epochs=30, batch=32, seed=1),
        "Stochastic (1 home per step)": minibatch_gd(X, y, lr=0.02, epochs=30, batch=1, seed=1),
        "Stochastic with decaying rate": minibatch_gd(X, y, lr=0.05, epochs=30, batch=1, seed=1, decay=1e-3),
    }
    sgd_final = {k: (v[-1] if isinstance(v, np.ndarray) else v[1][-1]) for k, v in sgd_runs.items()}
    sgd_runs = {k: (v if isinstance(v, np.ndarray) and v.ndim == 1 else v[1]) for k, v in sgd_runs.items()}
    b_sgd, _ = minibatch_gd(X, y, lr=0.05, epochs=30, batch=1, seed=1, decay=1e-3)

    # ------------------------------------------------------------- inference
    inf = inference(X, y, b_normal)
    resid = y - X @ b_normal
    bp_r = np.corrcoef(X @ b_normal, resid ** 2)[0, 1]

    # ------------------------------------------------------------- multiple regression: same machinery in matrix form
    feats = ["living_area_sqft", "overall_quality", "year_built", "garage_cars", "basement_sqft"]
    F = df[feats].to_numpy(float)
    mu, sd = F.mean(0), F.std(0)
    XF = add_intercept((F - mu) / sd)
    bF = normal_equation(XF, y)
    eigF = np.linalg.eigvalsh(hessian(XF))
    bF_gd, pF, _ = gradient_descent(XF, y, lr=1.9 / eigF.max(), iters=50_000, tol=1e-10)
    skF = LinearRegression().fit((F - mu) / sd, y)
    infF = inference(XF, y, bF)
    print(f"multiple regression R2 {infF['r2']:.3f} vs simple {inf['r2']:.3f}; GD vs normal eq {np.abs(bF_gd - bF).max():.1e}; vs sklearn {np.abs(np.r_[skF.intercept_, skF.coef_] - bF).max():.1e}")

    # ------------------------------------------------------------- chart 1: 3D loss surface with descent paths
    b0g = np.linspace(-45, 115, 140)
    b1g = np.linspace(40, 175, 140)
    B0, B1 = np.meshgrid(b0g, b1g)
    r = y[None, None, :] - (B0[..., None] + B1[..., None] * x[None, None, :])
    Jg = (r ** 2).mean(-1) / 2 / 1000  # in millions of (k USD)^2 / 1000 for readable axis
    fig3d = go.Figure(go.Surface(x=B0, y=B1, z=Jg, colorscale=[[0, "#10233b"], [0.15, "#1f3b5c"], [0.5, BLUE], [1, "#d6e6fb"]],
                                 opacity=0.88, showscale=False, name="Loss surface",
                                 contours=dict(z=dict(show=True, usecolormap=True, project_z=True, width=1)),
                                 hovertemplate="intercept %{x:.0f}<br>slope %{y:.0f}<br>loss %{z:.1f}k<extra></extra>"))
    evals, evecs = np.linalg.eigh(H)
    v_min, v_max = evecs[:, 0] * np.sign(evecs[0, 0]), evecs[:, 1] * np.sign(evecs[1, 1])
    start = b_normal + 25 * v_max + 95 * v_min  # off the valley floor, so the shape of the path is visible
    for lr, col in [(0.05, AMBER), (0.4, AQUA), (0.58, PINK)]:
        _, p_, l_ = gradient_descent(X, y, lr, 150 if lr != 0.05 else 600, beta0=start)
        keep = (p_[:, 0] > b0g[0]) & (p_[:, 0] < b0g[-1]) & (p_[:, 1] > b1g[0]) & (p_[:, 1] < b1g[-1])
        p_, l_ = p_[keep], l_[keep]
        fig3d.add_trace(go.Scatter3d(x=p_[:, 0], y=p_[:, 1], z=l_ / 1000 + 0.15, mode="lines+markers", name=f"Gradient descent, learning rate {lr}",
                                     line=dict(color=col, width=5), marker=dict(size=2.2, color=col),
                                     hovertemplate=f"lr {lr}<br>intercept %{{x:.1f}}<br>slope %{{y:.1f}}<br>loss %{{z:.2f}}k<extra></extra>"))
    fig3d.add_trace(go.Scatter3d(x=[b_normal[0]], y=[b_normal[1]], z=[J_star / 1000 + 0.15], mode="markers", name="Normal equation (exact minimum)",
                                 marker=dict(size=7, color=TEXT, symbol="diamond", line=dict(color=BG, width=1))))
    axis = dict(backgroundcolor=BG, gridcolor=GRID, zerolinecolor=GRID, color=TEXT2)
    fig3d.update_layout(**plotly_layout(
        title=dict(text="The loss is a bowl with one bottom: every learning rate that converges ends at the normal-equation solution", x=0.02),
        scene=dict(xaxis=dict(title="Intercept (k USD)", **axis), yaxis=dict(title="Slope (k USD per 1,000 sq ft)", **axis),
                   zaxis=dict(title="Loss J (thousands)", **axis), aspectratio=dict(x=1.2, y=1.2, z=0.75),
                   camera=dict(eye=dict(x=1.15, y=-1.35, z=1.45))),
        legend=dict(font=dict(color=TEXT2), x=0.01, y=0.92), height=780))
    fig3d.write_html(OUT / "01_loss_surface_3d.html", include_plotlyjs="cdn")
    try:
        fig3d.write_image(OUT / "01_loss_surface_3d.png", width=1600, height=900, scale=2)
    except Exception as exc:  # noqa: BLE001
        print(f"  (static 3D export skipped: {exc.__class__.__name__})")

    # ------------------------------------------------------------- chart 2: same line
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), gridspec_kw={"width_ratios": [1.35, 1]})
    axes[0].scatter(x, y, s=6, color=MUTED, alpha=0.45, lw=0)
    xs = np.linspace(x.min(), x.max(), 50)
    _, p_show, _ = gradient_descent(X, y, 0.1, 400, beta0=[0, 0])
    for k_, a in [(1, 0.25), (3, 0.35), (8, 0.45), (20, 0.6), (60, 0.75)]:
        axes[0].plot(xs, p_show[k_, 0] + p_show[k_, 1] * xs, color=AMBER, lw=1.2, alpha=a)
        axes[0].text(xs[-1] + 0.03, p_show[k_, 0] + p_show[k_, 1] * xs[-1], f"step {k_}", color=AMBER, fontsize=8, va="center", alpha=a + 0.2)
    axes[0].plot(xs, b_normal[0] + b_normal[1] * xs, color=BLUE, lw=3.2, label="Normal equation")
    axes[0].plot(xs, b_gd[0] + b_gd[1] * xs, color=PINK, lw=1.4, ls="--", label=f"Gradient descent ({len(path_gd) - 1} steps)")
    se_fit = np.sqrt(np.einsum("ij,jk,ik->i", add_intercept(xs), inf["cov"], add_intercept(xs)))
    yfit = b_normal[0] + b_normal[1] * xs
    axes[0].fill_between(xs, yfit - inf["t_crit"] * se_fit, yfit + inf["t_crit"] * se_fit, color=BLUE, alpha=0.25, label="95% confidence band")
    axes[0].set_xlabel("Living area (thousand sq ft)"); axes[0].set_ylabel("Sale price (thousand USD)")
    axes[0].legend(frameon=False, labelcolor=TEXT2, loc="upper left"); axes[0].grid(lw=0.5)
    title(axes[0], f"price = {b_normal[0]:.1f} + {b_normal[1]:.1f} x area   (R² = {inf['r2']:.3f})")
    methods = ["Normal equation\n(Gaussian elimination)", "cov / var formula", "NumPy lstsq", "scikit-learn", f"Gradient descent", "Stochastic GD"]
    coefs = [b_normal, b_simple, b_lstsq, b_sk, b_gd, b_sgd]
    diffs = [max(np.abs(c - b_normal).max(), 1e-16) for c in coefs]
    axes[1].barh(methods[::-1], diffs[::-1], color=[VIOLET, BLUE, MUTED, MUTED, MUTED, BLUE][::-1], height=0.55)
    axes[1].set_xscale("log"); axes[1].tick_params(axis="y", colors=TEXT, length=0)
    for i, d in enumerate(diffs[::-1]):
        axes[1].text(d * 1.6, i, f"{d:.0e}" if d > 1e-16 else "identical", va="center", color=TEXT, fontsize=9.5)
    axes[1].set_xlabel("Largest difference from the normal-equation coefficients (log scale)"); axes[1].grid(axis="x", lw=0.5)
    title(axes[1], "Six methods, one line")
    fig.tight_layout()
    fig.savefig(OUT / "02_same_line.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 3: learning rate
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6))
    for (lr, (_, _, ls_)), col in zip(runs.items(), [MUTED, VIOLET, AQUA, AMBER, RED]):
        gap = np.maximum(ls_ - J_star, 1e-12)
        axes[0].semilogy(range(len(gap)), gap, color=col, lw=2, label=f"learning rate {lr}")
    axes[0].set_ylim(1e-10, 1e7); axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Loss above the minimum (log scale)")
    axes[0].legend(frameon=False, labelcolor=TEXT2, loc="upper right"); axes[0].grid(lw=0.5)
    title(axes[0], "Too small crawls, near the limit oscillates, above it explodes")
    ok = np.isfinite(scan_loss)
    axes[1].semilogy(scan[ok], np.maximum(scan_loss[ok], 1e-12), color=BLUE, lw=2.2)
    axes[1].axvline(lr_max, color=RED, ls="--", lw=1.4)
    axes[1].axvline(lr_opt, color=AQUA, ls=":", lw=1.4)
    axes[1].text(lr_max, 1e-1, f"  limit 2/λ_max = {lr_max:.3f}\n  (diverges beyond)", color=RED, fontsize=9.5)
    axes[1].text(lr_opt, 1e-8, f"optimal 2/(λ_min+λ_max) = {lr_opt:.3f}  ", color=AQUA, fontsize=9.5, ha="right")
    axes[1].set_xlabel("Learning rate"); axes[1].set_ylabel("Loss above the minimum after 300 iterations (log scale)"); axes[1].grid(lw=0.5)
    title(axes[1], "The theory predicts the exact breaking point")
    fig.tight_layout()
    fig.savefig(OUT / "03_learning_rate.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 4: contours and feature scaling
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))
    cs_levels = np.quantile(Jg * 1000, np.linspace(0.002, 0.6, 18))
    axes[0].contour(B0, B1, Jg * 1000, levels=cs_levels, colors=GRID, linewidths=0.9)
    for lr, col in [(0.05, AMBER), (0.4, AQUA), (0.58, PINK)]:
        _, p_, _ = gradient_descent(X, y, lr, 80 if lr != 0.05 else 600, beta0=start)
        axes[0].plot(p_[:, 0], p_[:, 1], color=col, lw=1.6, marker="o", ms=2.5, label=f"learning rate {lr}")
    axes[0].scatter(*b_normal, color=TEXT, marker="D", s=50, zorder=5, label="minimum")
    axes[0].set_xlim(-45, 115); axes[0].set_ylim(40, 175)
    axes[0].set_xlabel("Intercept"); axes[0].set_ylabel("Slope"); axes[0].legend(frameon=False, labelcolor=TEXT2, loc="upper right")
    title(axes[0], f"Area in thousand sq ft: elongated valley (condition number {kappa:.0f}), zig-zag near the limit")
    zg0, zg1 = np.meshgrid(np.linspace(bz[0] - 90, bz[0] + 90, 120), np.linspace(bz[1] - 90, bz[1] + 90, 120))
    rz = y[None, None, :] - (zg0[..., None] + zg1[..., None] * z[None, None, :])
    Jz = (rz ** 2).mean(-1) / 2
    axes[1].contour(zg0, zg1, Jz, levels=np.quantile(Jz, np.linspace(0.002, 0.6, 18)), colors=GRID, linewidths=0.9)
    for lr, col in [(0.3, AMBER), (1.0, AQUA)]:
        _, p_, _ = gradient_descent(Xz, y, lr, 30, beta0=[bz[0] - 70, bz[1] + 45] if lr != 1.0 else [bz[0] + 55, bz[1] - 70])
        axes[1].plot(p_[:, 0], p_[:, 1], color=col, lw=1.8, marker="o", ms=3, label=f"learning rate {lr}" + (" (one step)" if lr == 1.0 else ""))
    axes[1].scatter(*bz, color=TEXT, marker="D", s=50, zorder=5)
    axes[1].set_aspect("equal"); axes[1].set_xlabel("Intercept"); axes[1].set_ylabel("Slope per standard deviation of area")
    axes[1].legend(frameon=False, labelcolor=TEXT2, loc="upper right")
    title(axes[1], "Standardized area: a round bowl (condition number 1)")
    fig.text(0.01, 0.01, f"With area in raw square feet the condition number is {kappa_raw:.1e}: gradient descent would need millions of steps. Scaling features is not cosmetic.",
             color=TEXT2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "04_contours_and_scaling.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 5: stochastic vs batch + residuals
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), gridspec_kw={"width_ratios": [1.2, 1, 0.8]})
    for (name, ls_), col in zip(sgd_runs.items(), [BLUE, AQUA, AMBER, PINK]):
        axes[0].semilogy(range(len(ls_)), np.maximum(ls_ - J_star, 1e-6), color=col, lw=2, label=name)
    axes[0].set_xlabel("Epoch (one pass over the data)"); axes[0].set_ylabel("Loss above the minimum (log scale)")
    axes[0].legend(frameon=False, labelcolor=TEXT2); axes[0].grid(lw=0.5)
    title(axes[0], "Per pass over the data, stochastic steps get closer but never settle")
    fitted = X @ b_normal
    axes[1].scatter(fitted, resid, s=6, color=MUTED, alpha=0.5, lw=0)
    axes[1].axhline(0, color=BLUE, lw=1.4)
    axes[1].set_xlabel("Fitted price (k USD)"); axes[1].set_ylabel("Residual (k USD)"); axes[1].grid(lw=0.5)
    title(axes[1], "Residuals fan out: errors grow with price")
    axes[2].hist(resid, bins=60, color=BLUE, alpha=0.85, density=True)
    rr = np.linspace(resid.min(), resid.max(), 300)
    axes[2].plot(rr, stats.norm.pdf(rr, 0, inf["sigma"]), color=AMBER, lw=2, label="Normal")
    axes[2].set_xlabel("Residual (k USD)"); axes[2].legend(frameon=False, labelcolor=TEXT2); axes[2].grid(axis="y", lw=0.5)
    title(axes[2], "Heavier tails than normal")
    fig.tight_layout()
    fig.savefig(OUT / "05_sgd_and_residuals.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- metrics
    metrics = {
        "data": {"homes": n, "removed": "5 homes above 4,000 sq ft (De Cock, 2011)", "x": "living area, thousand sq ft", "y": "sale price, thousand USD"},
        "coefficients": {"normal_equation": b_normal.tolist(), "cov_var_formula": b_simple.tolist(), "numpy_lstsq": b_lstsq.tolist(),
                         "sklearn": b_sk.tolist(), "gradient_descent": b_gd.tolist(), "stochastic_gd": b_sgd.tolist()},
        "max_difference_vs_normal_equation": {"cov_var": float(np.abs(b_simple - b_normal).max()), "lstsq": float(np.abs(b_lstsq - b_normal).max()),
                                              "sklearn": float(np.abs(b_sk - b_normal).max()), "gradient_descent": float(np.abs(b_gd - b_normal).max()),
                                              "stochastic_gd": float(np.abs(b_sgd - b_normal).max())},
        "gradient_descent_steps_to_converge": len(path_gd) - 1,
        "hessian": {"eigenvalues": eig.tolist(), "condition_number": kappa, "lr_limit": lr_max, "lr_optimal": lr_opt,
                    "condition_raw_sqft": kappa_raw, "condition_standardized": float(eig_z.max() / eig_z.min()),
                    "standardized_steps_at_lr_1": len(path_z) - 1},
        "learning_rate_runs_final_gap": {str(lr): (float(r_[2][-1] - J_star) if np.isfinite(r_[2][-1]) else "diverged") for lr, r_ in runs.items()},
        "steps_to_tolerance": steps_to,
        "inference": {"intercept": [b_normal[0], inf["se"][0], inf["ci_low"][0], inf["ci_high"][0]],
                      "slope": [b_normal[1], inf["se"][1], inf["ci_low"][1], inf["ci_high"][1]],
                      "slope_t": float(inf["t"][1]), "r2": inf["r2"], "residual_sd_k_usd": inf["sigma"],
                      "corr_fitted_vs_squared_residual": float(bp_r), "residual_kurtosis_excess": float(stats.kurtosis(resid))},
        "multiple_regression": {"features": feats, "r2": infF["r2"], "adj_r2": infF["adj_r2"], "coef_per_sd": dict(zip(["intercept"] + feats, bF.round(3).tolist())),
                                "gd_vs_normal_max_diff": float(np.abs(bF_gd - bF).max()),
                                "sklearn_vs_normal_max_diff": float(np.abs(np.r_[skF.intercept_, skF.coef_] - bF).max())},
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(json.dumps({k: metrics[k] for k in ("max_difference_vs_normal_equation", "hessian", "learning_rate_runs_final_gap", "inference", "multiple_regression")}, indent=1, default=float))

    # ------------------------------------------------------------- dashboard with live gradient descent
    (OUT / "houses.js").write_text("window.HOUSES=" + json.dumps({"x": x.round(4).tolist(), "y": y.round(2).tolist(), "b": b_normal.tolist(),
                                                                  "lrmax": lr_max, "lropt": lr_opt, "start": start.round(3).tolist()}) + ";")
    kpis = {"Homes": f"{n:,}", "Slope": f"{b_normal[1]:.1f}k USD per 1,000 sq ft", "GD vs normal equation": f"{np.abs(b_gd - b_normal).max():.0e}",
            "Learning-rate limit": f"{lr_max:.3f}"}
    kpi_html = "".join(f'<div class="kpi"><span>{k}</span><b>{v}</b></div>' for k, v in kpis.items())
    sec = lambda t, n_, f: f'<section><h2>{t}</h2><p class="note">{n_}</p>{f.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})}</section>'
    img = lambda t, n_, src: f'<section><h2>{t}</h2><p class="note">{n_}</p><img src="outputs/{src}" alt="{t}"></section>'
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Linear Regression from Scratch</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><script src="outputs/houses.js"></script>
<style>:root{{--bg:{BG};--card:{CARD};--text:{TEXT};--text2:{TEXT2};--muted:{MUTED};--line:{GRID}}}*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font-family:Inter,"Segoe UI",Arial,sans-serif}}
header,.kpis,footer{{max-width:1240px;margin:auto;padding:0 32px}}header{{padding-top:48px}}h1{{font-size:40px;margin:0 0 8px}}
header p{{color:var(--text2);max-width:860px;line-height:1.55;margin:0}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-top:24px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px}}
.kpi span{{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.8px}}.kpi b{{display:block;font-size:20px;margin-top:6px}}
section{{max-width:1240px;margin:28px auto;padding:20px 24px;background:var(--card);border:1px solid var(--line);border-radius:14px}}
section img{{width:100%;border-radius:8px}}h2{{margin:4px 0 6px;font-size:20px}}.note{{color:var(--text2);margin:0 0 10px;font-size:14px}}
.controls{{display:flex;flex-wrap:wrap;gap:18px;align-items:center;color:var(--text2);font-size:14px;margin:8px 0 12px}}input[type=range]{{width:240px;accent-color:{BLUE};vertical-align:middle}}
button{{border:0;border-radius:8px;padding:8px 16px;font:inherit;font-weight:600;cursor:pointer;background:{BLUE};color:white}}button.sec{{background:var(--bg);color:var(--text2);border:1px solid var(--line)}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.readout{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:10px}}
.readout div{{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:12px;color:var(--muted)}}.readout b{{display:block;color:var(--text);font-size:17px}}
footer{{color:var(--muted);font-size:13px;margin:24px auto 48px}}a{{color:#7fb0f0}}
@media(max-width:800px){{.grid2,.readout{{grid-template-columns:1fr 1fr}}header,.kpis,footer{{padding:0 16px}}section{{margin:16px;padding:12px}}h1{{font-size:30px}}}}</style></head><body>
<header><h1>Linear Regression from Scratch</h1><p>Least squares two ways on {n:,} home sales: the normal equation, solved by Gaussian elimination written by hand, and gradient descent.
Both reach the same line; the learning rate decides whether gradient descent gets there, how fast, or not at all.</p></header>
<div class="kpis">{kpi_html}</div>
<section><h2>Run gradient descent yourself</h2><p class="note">Pick a learning rate and step through gradient descent. Left: the path on the contour map of the loss. Right: the line it currently draws. The white diamond is the normal-equation answer. Above {lr_max:.3f} it diverges.</p>
<div class="controls"><span>Learning rate <input type="range" id="lr" min="0.005" max="0.65" step="0.005" value="0.3"> <b id="lr_o"></b></span>
<button id="play">Run</button><button class="sec" id="step">+1 step</button><button class="sec" id="reset">Reset</button></div>
<div class="grid2"><div id="contour" style="height:420px"></div><div id="line" style="height:420px"></div></div>
<div class="readout"><div>Iteration<b id="it">0</b></div><div>Intercept<b id="b0">-</b></div><div>Slope<b id="b1">-</b></div><div>Distance to the exact answer<b id="dist">-</b></div></div></section>
{sec("The loss surface in 3D", "Drag to rotate. Three learning rates descending the same bowl from the same starting point.", fig3d)}
{img("Same line", "Six methods agree on the coefficients.", "02_same_line.png")}
{img("Learning rate", "From crawling to exploding, and the theoretical limit that separates them.", "03_learning_rate.png")}
{img("Contours and feature scaling", "Why a badly scaled feature makes gradient descent slow.", "04_contours_and_scaling.png")}
{img("Stochastic gradient descent and residuals", "Cheaper steps, and what the residuals say about the model.", "05_sgd_and_residuals.png")}
<footer>Tomás Alarcón &middot; <a href="https://github.com/tomasalarcon10/ml-from-scratch">Source code</a></footer>
<script>
const H=window.HOUSES,n=H.x.length;let b=[...H.start],path=[[...H.start]],it=0,timer=null;
function grad(b){{let g0=0,g1=0;for(let i=0;i<n;i++){{const r=b[0]+b[1]*H.x[i]-H.y[i];g0+=r;g1+=r*H.x[i]}}return [g0/n,g1/n]}}
function J(b0,b1){{let s=0;for(let i=0;i<n;i+=3){{const r=b0+b1*H.x[i]-H.y[i];s+=r*r}}return s/(2*Math.ceil(n/3))}}
const g0=[],g1=[],Z=[];for(let i=0;i<70;i++)g0.push(-120+260*i/69);for(let j=0;j<70;j++)g1.push(40+160*j/69);for(const s of g1)Z.push(g0.map(a=>Math.log10(J(a,s))));
const lay=(t,xa,ya)=>({{title:{{text:t,font:{{size:14,color:"{TEXT}"}},x:0.02}},paper_bgcolor:"{CARD}",plot_bgcolor:"{CARD}",font:{{color:"{TEXT2}",family:"Inter, Segoe UI, Arial"}},
margin:{{l:55,r:10,t:40,b:45}},xaxis:{{title:{{text:xa}},gridcolor:"{GRID}",zeroline:false}},yaxis:{{title:{{text:ya}},gridcolor:"{GRID}",zeroline:false}},showlegend:false}});
function draw(){{const d=Math.hypot(b[0]-H.b[0],b[1]-H.b[1]);document.getElementById("it").textContent=it;document.getElementById("b0").textContent=isFinite(b[0])?b[0].toFixed(2):"diverged";
document.getElementById("b1").textContent=isFinite(b[1])?b[1].toFixed(2):"diverged";document.getElementById("dist").textContent=isFinite(d)?(d<1e-3?d.toExponential(1):d.toFixed(3)):"infinite";
const vis=path.filter(p=>p[0]>-120&&p[0]<140&&p[1]>40&&p[1]<200);
Plotly.react("contour",[{{x:g0,y:g1,z:Z,type:"contour",colorscale:[[0,"#10233b"],[0.5,"#1f3b5c"],[1,"#3987e5"]],showscale:false,contours:{{coloring:"heatmap"}},line:{{color:"{GRID}",width:0.6}},hoverinfo:"skip"}},
{{x:vis.map(p=>p[0]),y:vis.map(p=>p[1]),mode:"lines+markers",line:{{color:"{AMBER}",width:2}},marker:{{size:4,color:"{AMBER}"}},hovertemplate:"intercept %{{x:.1f}}<br>slope %{{y:.1f}}<extra></extra>"}},
{{x:[H.b[0]],y:[H.b[1]],mode:"markers",marker:{{symbol:"diamond",size:12,color:"{TEXT}"}},hoverinfo:"skip"}}],lay("Path on the loss contours","Intercept","Slope"),{{displaylogo:false,responsive:true}});
const xs=[0.3,4],ok=isFinite(b[0])&&isFinite(b[1]);
Plotly.react("line",[{{x:H.x,y:H.y,mode:"markers",marker:{{size:3,color:"{MUTED}",opacity:0.5}},hoverinfo:"skip"}},
{{x:xs,y:xs.map(v=>H.b[0]+H.b[1]*v),mode:"lines",line:{{color:"{BLUE}",width:3}},hoverinfo:"skip"}},
{{x:xs,y:ok?xs.map(v=>b[0]+b[1]*v):[null,null],mode:"lines",line:{{color:"{AMBER}",width:2.5,dash:"dash"}},hoverinfo:"skip"}}],
{{...lay("Blue: normal equation  ·  dashed: current gradient descent line","Living area (thousand sq ft)","Price (thousand USD)"),yaxis:{{range:[0,620],gridcolor:"{GRID}",title:{{text:"Price (thousand USD)"}}}}}},{{displaylogo:false,responsive:true}})}}
function step(){{const lr=+document.getElementById("lr").value,g=grad(b);b=[b[0]-lr*g[0],b[1]-lr*g[1]];path.push([...b]);it++}}
function reset(){{clearInterval(timer);timer=null;document.getElementById("play").textContent="Run";b=[...H.start];path=[[...H.start]];it=0;draw()}}
document.getElementById("lr").oninput=e=>{{const v=+e.target.value;document.getElementById("lr_o").textContent=v.toFixed(3)+(v>H.lrmax?" (above the limit)":"");reset()}};
document.getElementById("step").onclick=()=>{{step();draw()}};document.getElementById("reset").onclick=reset;
document.getElementById("play").onclick=e=>{{if(timer){{clearInterval(timer);timer=null;e.target.textContent="Run";return}}e.target.textContent="Pause";
timer=setInterval(()=>{{for(let k=0;k<3;k++)step();draw();if(it>=600||!isFinite(b[0])||Math.abs(b[0])>1e6){{clearInterval(timer);timer=null;e.target.textContent="Run"}}}},60)}};
document.getElementById("lr_o").textContent="0.300";draw();
</script></body></html>"""
    (ROOT / "dashboard.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
