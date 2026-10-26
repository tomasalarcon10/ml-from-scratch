"""
PCA from scratch on the daily returns of 56 large companies (2003-2007): the hidden factors behind the market.
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
from sklearn.decomposition import PCA as SkPCA

from src.pca import PCA, standardize, covariance, jacobi_eigh, power_iteration, align_signs, marchenko_pastur

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

BG, CARD, TEXT, TEXT2, MUTED, GRID = "#1a1a19", "#222220", "#f0efec", "#c3c2b7", "#8a8980", "#3a3936"
BLUE, ORANGE, AQUA, AMBER, PINK, VIOLET, RED = "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9", "#e66767"
SECTOR_ORDER = ["Energy", "Technology & media", "Financials", "Healthcare", "Consumer staples", "Consumer discretionary",
                "Industrials", "Autos"]
SECTOR_COLORS = dict(zip(SECTOR_ORDER, ["#c98500", "#3987e5", "#199e70", "#d55181", "#9085e9", "#d95926", "#8a8980", "#5fb8c9"]))
DIVERGING = LinearSegmentedColormap.from_list("div", ["#d95926", "#5a2f1c", "#2a2a28", "#1f3b5c", "#3987e5"])

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


def adjust_for_splits(open_, close, tol=0.03):
    """
    Prices in the file are not split-adjusted: a 2-for-1 split looks like an overnight 50% crash.
    An overnight gap (open / previous close) within 3% of 1/2, 1/3, 2/3 or 1/4 is treated as a split,
    and all earlier prices are rescaled by the exact ratio.
    """
    adj = close.copy()
    events = []
    ratios = np.array([1 / 2, 1 / 3, 2 / 3, 1 / 4])
    for t in close.columns:
        gap = (open_[t] / close[t].shift(1)).to_numpy()
        for i in np.where(np.isfinite(gap))[0]:
            rel = np.abs(gap[i] / ratios - 1)
            if rel.min() < tol:
                r = ratios[rel.argmin()]
                adj.iloc[:i, adj.columns.get_loc(t)] *= r
                events.append({"ticker": t, "date": str(close.index[i].date()), "ratio": f"{round(1 / r, 2):g}-for-1" if r != 2 / 3 else "3-for-2"})
    return adj, events


def main():
    prices = pd.read_csv(ROOT / "data" / "stock_prices.csv", parse_dates=["date"])
    companies = pd.read_csv(ROOT / "data" / "companies.csv").set_index("ticker")
    close = prices.pivot(index="date", columns="ticker", values="close")
    open_ = prices.pivot(index="date", columns="ticker", values="open")
    tickers = sorted(close.columns, key=lambda t: (SECTOR_ORDER.index(companies.loc[t, "sector"]), t))
    close, open_ = close[tickers], open_[tickers]
    adj, splits = adjust_for_splits(open_, close)
    R = np.log(adj).diff().dropna()
    lo, hi = R.quantile(0.001), R.quantile(0.999)
    clipped = int(((R < lo) | (R > hi)).sum().sum())
    R = R.clip(lo, hi, axis=1)
    dates = R.index
    Rv = R.to_numpy()
    T, N = Rv.shape
    sectors = [companies.loc[t, "sector"] for t in tickers]
    names = [companies.loc[t, "name"] for t in tickers]
    print(f"{N} stocks, {T} daily returns ({dates[0].date()} to {dates[-1].date()}); {len(splits)} splits adjusted, {clipped} extreme returns winsorized")

    # ------------------------------------------------------------- PCA from scratch on standardized returns
    Z, mu, sd = standardize(Rv)
    pca = PCA().fit(Z)
    C = pca.covariance_
    lam, V = pca.eigenvalues_, pca.components_
    if V[:, 0].sum() < 0:
        V[:, 0] *= -1
    evr = pca.explained_variance_ratio_
    scores = pca.transform(Z)

    # ------------------------------------------------------------- correctness checks
    w_np, W_np = np.linalg.eigh(C)
    w_np, W_np = w_np[::-1], W_np[:, ::-1]
    sk = SkPCA().fit(Z)
    pv, PV, p_hist = power_iteration(C, 3)
    checks = {
        "correlation_matrix_vs_numpy_corrcoef": float(np.max(np.abs(C - np.corrcoef(Rv.T)))),
        "jacobi_eigenvalues_vs_numpy_eigh": float(np.max(np.abs(lam - w_np))),
        "jacobi_eigenvectors_vs_numpy_eigh": float(np.max(np.abs(align_signs(V, W_np) - W_np))),
        "explained_variance_vs_sklearn": float(np.max(np.abs(evr - sk.explained_variance_ratio_))),
        "power_iteration_top3_vs_jacobi": float(np.max(np.abs(pv - lam[:3]))),
        "power_iteration_steps": [len(h) for h in p_hist],
        "jacobi_sweeps": len(pca.jacobi_history_),
        "reconstruction_all_components_error": float(np.max(np.abs(pca.inverse_transform(scores) - Z))),
    }
    print("checks:", checks)

    # ------------------------------------------------------------- noise vs signal (Marchenko-Pastur)
    q = N / T
    # the market factor absorbs part of the variance, so the noise level is what PC1 leaves (Laloux et al., 1999)
    sigma2 = 1 - lam[0] / N
    mp_lo, mp_hi, mp_x, mp_d = marchenko_pastur(q, sigma2)
    n_signal = int((lam > mp_hi).sum())

    # ------------------------------------------------------------- interpretation
    load = V * np.sqrt(lam)  # correlation between each stock and each component
    sector_load = {s: [float(load[[i for i, x in enumerate(sectors) if x == s], k].mean()) for k in range(5)] for s in SECTOR_ORDER}
    ew = Rv.mean(1)
    corr_pc1_market = float(np.corrcoef(scores[:, 0], ew)[0, 1])
    eig_port = {}
    for k in range(3):
        w = V[:, k] / sd
        w = w / np.abs(w).sum()
        eig_port[f"PC{k + 1}"] = Rv @ w
    window = 126
    roll = []
    for i in range(window, T + 1):
        Ci = np.corrcoef(Rv[i - window:i].T)
        ev = np.linalg.eigvalsh(Ci)
        roll.append(ev[-1] / ev.sum())
    roll_dates = dates[window - 1:]
    recon_err = []
    off = ~np.eye(N, dtype=bool)
    for k in range(0, 21):
        Ck = (V[:, :k] * lam[:k]) @ V[:, :k].T if k else np.zeros_like(C)
        recon_err.append(float(np.sqrt(np.mean((C[off] - Ck[off]) ** 2))))
    print(f"PC1 explains {evr[0]:.1%}; first 3 {evr[:3].sum():.1%}; {n_signal} eigenvalues above the noise edge {mp_hi:.2f}; "
          f"corr(PC1, equal-weight market) {corr_pc1_market:.3f}")

    # ------------------------------------------------------------- chart 1: 3D loadings
    fig3d = go.Figure()
    for s in SECTOR_ORDER:
        idx = [i for i, x in enumerate(sectors) if x == s]
        xs, ys, zs = [], [], []
        for i in idx:
            xs += [0, load[i, 0], None]; ys += [0, load[i, 1], None]; zs += [0, load[i, 2], None]
        fig3d.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=SECTOR_COLORS[s], width=2),
                                     opacity=0.35, showlegend=False, hoverinfo="skip", legendgroup=s))
        fig3d.add_trace(go.Scatter3d(
            x=load[idx, 0], y=load[idx, 1], z=load[idx, 2], mode="markers+text", name=s, legendgroup=s,
            text=[tickers[i] for i in idx], textposition="top center", textfont=dict(color=SECTOR_COLORS[s], size=10),
            marker=dict(size=5, color=SECTOR_COLORS[s], line=dict(width=1, color=BG)),
            customdata=[[names[i], s] for i in idx],
            hovertemplate="<b>%{customdata[0]}</b> (%{text})<br>%{customdata[1]}<br>PC1 %{x:.2f} · PC2 %{y:.2f} · PC3 %{z:.2f}<extra></extra>"))
    axis = dict(backgroundcolor=BG, gridcolor=GRID, zerolinecolor=MUTED, color=TEXT2)
    fig3d.update_layout(**plotly_layout(
        title=dict(text=f"56 stocks in the space of their first three principal components ({evr[:3].sum():.0%} of all variance)", x=0.02),
        scene=dict(xaxis=dict(title=f"PC1: market ({evr[0]:.0%})", **axis), yaxis=dict(title=f"PC2 ({evr[1]:.0%})", **axis),
                   zaxis=dict(title=f"PC3 ({evr[2]:.0%})", **axis), camera=dict(eye=dict(x=-1.5, y=-1.6, z=0.6))),
        legend=dict(font=dict(color=TEXT2), x=0.01, y=0.92, itemsizing="constant"), height=780))
    fig3d.write_html(OUT / "01_stocks_in_pc_space_3d.html", include_plotlyjs="cdn")
    try:
        fig3d.write_image(OUT / "01_stocks_in_pc_space_3d.png", width=1600, height=900, scale=2)
    except Exception as exc:  # noqa: BLE001
        print(f"  (static 3D export skipped: {exc.__class__.__name__})")

    # ------------------------------------------------------------- chart 2: spectrum
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.1, 1]})
    k15 = np.arange(1, 16)
    axes[0].bar(k15, 100 * evr[:15], color=[BLUE if l > mp_hi else MUTED for l in lam[:15]], width=0.7)
    ax2 = axes[0]
    cum = 100 * np.cumsum(evr[:15])
    for i, (v, c) in enumerate(zip(100 * evr[:15], cum)):
        ax2.text(i + 1, v + 0.6, f"{v:.1f}%", ha="center", color=TEXT, fontsize=8.5)
    ax2.set_xticks(k15); ax2.set_xlabel("Principal component"); ax2.set_ylabel("Share of total variance (%)"); ax2.grid(axis="y", lw=0.5)
    ax2.text(15, 100 * evr[0] * 0.8, f"Blue: above the noise edge (Marchenko-Pastur)\nFirst 3 components: {cum[2]:.0f}%  |  first 10: {cum[9]:.0f}%",
             ha="right", color=TEXT2, fontsize=9.5)
    title(ax2, "Scree plot: one dominant factor and a few smaller ones")
    axes[1].hist(lam[lam < 4], bins=40, density=True, color=MUTED, alpha=0.8, label="Eigenvalues of the real correlation matrix")
    axes[1].plot(mp_x, mp_d, color=AMBER, lw=2.4, label=f"Pure noise (Marchenko-Pastur, q = {q:.3f}, noise variance {sigma2:.2f})")
    axes[1].axvline(mp_hi, color=AMBER, ls="--", lw=1)
    big = lam[lam >= 4]
    axes[1].annotate(f"Off the chart: PC1 = {lam[0]:.1f}" + (f", PC2 = {lam[1]:.1f}" if lam[1] >= 4 else ""), (3.95, 0.2), ha="right", color=BLUE, fontsize=10)
    axes[1].set_xlabel("Eigenvalue"); axes[1].set_ylabel("Density"); axes[1].legend(frameon=False, labelcolor=TEXT2); axes[1].grid(lw=0.5)
    title(axes[1], f"Signal vs noise: {n_signal} eigenvalues stand out from randomness")
    fig.tight_layout()
    fig.savefig(OUT / "02_eigenvalue_spectrum.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 3: loadings heatmap
    fig, ax = plt.subplots(figsize=(17, 5.6))
    vmax = np.abs(load[:, :5]).max()
    ax.imshow(load[:, :5].T, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(5), [f"PC{k + 1} ({100 * evr[k]:.0f}%)" for k in range(5)])
    ax.set_xticks(range(N), tickers, rotation=90, fontsize=8.5)
    for lbl, s in zip(ax.get_xticklabels(), sectors):
        lbl.set_color(SECTOR_COLORS[s])
    pos = 0
    for s in SECTOR_ORDER:
        n_s = sectors.count(s)
        ax.axvline(pos - 0.5, color=BG, lw=3)
        ax.text(pos + n_s / 2 - 0.5, -0.85, s, ha="center", color=SECTOR_COLORS[s], fontsize=9.5, fontweight="bold")
        pos += n_s
    ax.tick_params(axis="y", colors=TEXT, length=0); ax.tick_params(axis="x", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_ylim(4.5, -1.3)
    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(-vmax, vmax))
    cb = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01); cb.outline.set_visible(False); cb.ax.tick_params(colors=MUTED)
    cb.set_label("Loading (correlation with the component)", color=TEXT2)
    title(ax, "What each component means: loadings of every stock, grouped by sector (blue = moves with the component, orange = against it)", 13)
    fig.tight_layout()
    fig.savefig(OUT / "03_loadings_heatmap.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 4: eigen-portfolios and market factor over time
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})
    axes[0].plot(dates, 100 * np.expm1(np.cumsum(ew)), color=TEXT2, lw=1.6, ls="--", label="Equal-weight portfolio of all 56")
    for (k, r), col in zip(eig_port.items(), [BLUE, AMBER, PINK]):
        axes[0].plot(dates, 100 * np.expm1(np.cumsum(r)), color=col, lw=2 if k == "PC1" else 1.5, label=f"{k} eigen-portfolio")
    axes[0].axhline(0, color=MUTED, lw=0.8)
    axes[0].set_ylabel("Cumulative return (%)"); axes[0].legend(frameon=False, labelcolor=TEXT2, loc="upper left"); axes[0].grid(lw=0.5)
    title(axes[0], f"PC1 is the market: its daily score has correlation {corr_pc1_market:.3f} with the average stock")
    axes[1].plot(roll_dates, 100 * np.array(roll), color=BLUE, lw=2)
    axes[1].fill_between(roll_dates, 0, 100 * np.array(roll), color=BLUE, alpha=0.12)
    axes[1].set_ylabel("Variance explained by PC1 (%)"); axes[1].grid(lw=0.5); axes[1].set_ylim(0, max(roll) * 100 * 1.15)
    i_max = int(np.argmax(roll))
    axes[1].annotate(f"{100 * roll[i_max]:.0f}% ({roll_dates[i_max]:%b %Y})", (roll_dates[i_max], 100 * roll[i_max]), xytext=(-140, 6),
                     textcoords="offset points", color=TEXT, arrowprops=dict(arrowstyle="-", color=MUTED))
    title(axes[1], "How much the market factor dominates (rolling 6 months)")
    fig.tight_layout()
    fig.savefig(OUT / "04_market_factor.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 5: correlation matrix and rank-3 reconstruction
    C3 = (V[:, :3] * lam[:3]) @ V[:, :3].T
    np.fill_diagonal(C3, 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    corr_cmap = LinearSegmentedColormap.from_list("c", ["#5a2f1c", CARD, "#1f3b5c", BLUE, "#b9d6f8"])
    for ax, M, t_ in [(axes[0], C, "Real correlation matrix"), (axes[1], C3, f"Rebuilt from 3 components ({evr[:3].sum():.0%} of variance)"),
                      (axes[2], C - C3 + np.eye(N) * 0, "What is left (real minus rebuilt)")]:
        im = ax.imshow(M, cmap=corr_cmap if ax is not axes[2] else DIVERGING, vmin=-0.2 if ax is not axes[2] else -0.4,
                       vmax=1 if ax is not axes[2] else 0.4)
        pos = 0
        for s in SECTOR_ORDER:
            n_s = sectors.count(s)
            ax.add_patch(plt.Rectangle((pos - 0.5, pos - 0.5), n_s, n_s, fill=False, ec=SECTOR_COLORS[s], lw=1.6))
            pos += n_s
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        title(ax, t_, 12)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02); cb.outline.set_visible(False); cb.ax.tick_params(colors=MUTED)
    fig.text(0.01, 0.02, "Stocks ordered by sector; colored squares mark each sector. The leftover is mostly within-sector correlation that later components capture.",
             color=TEXT2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "05_correlation_reconstruction.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 6: the algorithms converge
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].semilogy(range(len(pca.jacobi_history_)), pca.jacobi_history_, color=BLUE, lw=2.2, marker="o")
    axes[0].set_xlabel("Sweep"); axes[0].set_ylabel("Relative off-diagonal norm"); axes[0].grid(lw=0.5)
    title(axes[0], f"Jacobi rotations diagonalize the matrix in {len(pca.jacobi_history_) - 1} sweeps")
    for h, col, k in zip(p_hist, [BLUE, AMBER, PINK], range(3)):
        err = np.abs(np.array(h) - lam[k]) / lam[k]
        axes[1].semilogy(range(1, len(h) + 1), np.maximum(err, 1e-16), color=col, lw=2, label=f"PC{k + 1}: {len(h)} steps")
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("Relative eigenvalue error"); axes[1].legend(frameon=False, labelcolor=TEXT2); axes[1].grid(lw=0.5)
    title(axes[1], "Power iteration finds the top components one by one")
    axes[2].plot(range(21), recon_err, color=AQUA, lw=2.2, marker="o", ms=4)
    axes[2].set_xlabel("Components kept"); axes[2].set_ylabel("RMS error of the rebuilt correlations"); axes[2].grid(lw=0.5)
    title(axes[2], "Each component explains a bit more of how stocks co-move")
    fig.tight_layout()
    fig.savefig(OUT / "06_algorithms_and_reconstruction.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- metrics
    top = lambda k, n=5: {"positive": [tickers[i] for i in np.argsort(-load[:, k])[:n]], "negative": [tickers[i] for i in np.argsort(load[:, k])[:n]]}
    metrics = {
        "data": {"stocks": N, "daily_returns": T, "start": str(dates[0].date()), "end": str(dates[-1].date()), "splits_adjusted": splits,
                 "winsorized_returns": clipped, "sector_counts": {s: sectors.count(s) for s in SECTOR_ORDER}},
        "checks": checks,
        "eigenvalues": lam.round(4).tolist(), "explained_variance_ratio": evr.round(4).tolist(),
        "cumulative_variance": {"1": round(float(evr[:1].sum()), 4), "3": round(float(evr[:3].sum()), 4), "5": round(float(evr[:5].sum()), 4),
                                "10": round(float(evr[:10].sum()), 4)},
        "marchenko_pastur": {"q": round(q, 4), "noise_variance": round(float(sigma2), 4), "lambda_plus": round(mp_hi, 4), "components_above": n_signal},
        "pc1_correlation_with_equal_weight_market": round(corr_pc1_market, 4),
        "pc1_loading_range": [round(float(load[:, 0].min()), 3), round(float(load[:, 0].max()), 3)],
        "extreme_loadings": {f"PC{k + 1}": top(k) for k in range(5)},
        "sector_mean_loadings": {s: [round(v, 3) for v in vals] for s, vals in sector_load.items()},
        "rolling_pc1_share": {"min": round(float(min(roll)), 4), "max": round(float(max(roll)), 4), "max_date": str(roll_dates[i_max].date())},
        "correlation_reconstruction_rms": [round(v, 4) for v in recon_err],
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("cumulative_variance", "marchenko_pastur", "extreme_loadings", "rolling_pc1_share")}, indent=1))

    # ------------------------------------------------------------- dashboard
    live = {"tickers": tickers, "names": names, "sectors": sectors, "colors": SECTOR_COLORS, "order": SECTOR_ORDER,
            "lam": lam.round(6).tolist(), "V": V.round(6).tolist(), "C": C.round(4).tolist()}
    (OUT / "pca.js").write_text("window.PCA_DATA=" + json.dumps(live, separators=(",", ":")) + ";", encoding="utf-8")
    kpis = {"Stocks x days": f"{N} x {T:,}", "PC1 (market) explains": f"{100 * evr[0]:.1f}%", "Components above noise": str(n_signal),
            "Match with NumPy / scikit-learn": f"{max(checks['jacobi_eigenvalues_vs_numpy_eigh'], checks['explained_variance_vs_sklearn']):.0e}"}
    kpi_html = "".join(f'<div class="kpi"><span>{k}</span><b>{v}</b></div>' for k, v in kpis.items())
    sec = lambda t, n_, f: f'<section><h2>{t}</h2><p class="note">{n_}</p>{f.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})}</section>'
    img = lambda t, n_, src: f'<section><h2>{t}</h2><p class="note">{n_}</p><img src="outputs/{src}" alt="{t}"></section>'
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PCA from Scratch</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><script src="outputs/pca.js"></script>
<style>:root{{--bg:{BG};--card:{CARD};--text:{TEXT};--text2:{TEXT2};--muted:{MUTED};--line:{GRID}}}*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font-family:Inter,"Segoe UI",Arial,sans-serif}}
header,.kpis,footer{{max-width:1240px;margin:auto;padding:0 32px}}header{{padding-top:48px}}h1{{font-size:40px;margin:0 0 8px}}
header p{{color:var(--text2);max-width:860px;line-height:1.55;margin:0}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-top:24px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px}}
.kpi span{{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.8px}}.kpi b{{display:block;font-size:24px;margin-top:6px}}
section{{max-width:1240px;margin:28px auto;padding:20px 24px;background:var(--card);border:1px solid var(--line);border-radius:14px}}
section img{{width:100%;border-radius:8px}}h2{{margin:4px 0 6px;font-size:20px}}.note{{color:var(--text2);margin:0 0 10px;font-size:14px}}
.row{{display:flex;flex-wrap:wrap;gap:16px;align-items:center;color:var(--text2);font-size:14px;margin:6px 0 10px}}input[type=range]{{width:260px;accent-color:{BLUE}}}
select{{background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:5px}}.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
footer{{color:var(--muted);font-size:13px;margin:24px auto 48px}}a{{color:#7fb0f0}}
@media(max-width:760px){{.grid2{{grid-template-columns:1fr}}header,.kpis,footer{{padding:0 16px}}section{{margin:16px;padding:12px}}h1{{font-size:30px}}input[type=range]{{width:100%}}}}</style></head><body>
<header><h1>PCA from Scratch</h1><p>Principal component analysis written in NumPy (covariance, Jacobi eigen-decomposition, power iteration, projection)
applied to {T:,} daily returns of {N} large companies. A handful of hidden factors, led by the market itself, drive how stocks move together.</p></header>
<div class="kpis">{kpi_html}</div>
<section><h2>Rebuild the market with k components</h2><p class="note">The correlation between every pair of stocks, rebuilt from only the first k principal components. With 1 component every pair looks alike (the market); a few more bring back the sectors.</p>
<div class="row"><span>Components kept: <b id="k_o"></b></span><input type="range" id="k" min="1" max="20" value="3"><span id="k_txt"></span></div>
<div class="grid2"><div id="hm_real" style="height:440px"></div><div id="hm_k" style="height:440px"></div></div></section>
<section><h2>Explore any two components</h2><p class="note">Each point is a stock, placed by its loadings. Stocks that move together sit close together.</p>
<div class="row"><span>Horizontal <select id="px"></select></span><span>Vertical <select id="py"></select></span></div><div id="scatter" style="height:520px"></div></section>
{sec("Stocks in 3D", "Drag to rotate. Every stock leans to the right (PC1, the market); the second and third components pull sectors apart.", fig3d)}
{img("Eigenvalue spectrum", "How much each component explains, and which components are more than noise.", "02_eigenvalue_spectrum.png")}
{img("Loadings", "Reading the components through the stocks that weigh most on them.", "03_loadings_heatmap.png")}
{img("The market factor", "Portfolios built from the eigenvectors, and how dominant the market factor was over time.", "04_market_factor.png")}
{img("Correlation matrix", "Three components already rebuild most of the structure.", "05_correlation_reconstruction.png")}
{img("Algorithms", "Both eigen-solvers written from scratch converge to machine precision.", "06_algorithms_and_reconstruction.png")}
<footer>Tomás Alarcón &middot; <a href="https://github.com/tomasalarcon10/ml-from-scratch">Source code</a></footer>
<script>
const P=window.PCA_DATA,N=P.tickers.length,tot=P.lam.reduce((a,b)=>a+b,0);
const lay=(t)=>({{title:{{text:t,font:{{size:14,color:"{TEXT}"}},x:0.02}},paper_bgcolor:"{CARD}",plot_bgcolor:"{CARD}",font:{{color:"{TEXT2}",family:"Inter, Segoe UI, Arial"}},margin:{{l:50,r:10,t:40,b:50}}}});
const cs=[[0,"#5a2f1c"],[0.17,"{CARD}"],[0.5,"#1f3b5c"],[0.8,"{BLUE}"],[1,"#b9d6f8"]];
const ax={{showticklabels:false,ticks:"",showgrid:false,zeroline:false}};
const lbl=P.tickers.map((t,i)=>`${{t}} (${{P.sectors[i]}})`);
Plotly.newPlot("hm_real",[{{z:P.C,x:lbl,y:lbl,type:"heatmap",colorscale:cs,zmin:-0.2,zmax:1,showscale:false,hovertemplate:"%{{y}}<br>%{{x}}<br>correlation %{{z:.2f}}<extra></extra>"}}],
{{...lay("Real correlations"),xaxis:ax,yaxis:{{...ax,autorange:"reversed"}}}},{{displaylogo:false,responsive:true}});
function rebuild(){{const k=+document.getElementById("k").value;const M=[];for(let i=0;i<N;i++){{M.push([]);for(let j=0;j<N;j++){{let s=0;for(let c=0;c<k;c++)s+=P.V[i][c]*P.V[j][c]*P.lam[c];M[i].push(i===j?1:s)}}}}
let err=0,cnt=0;for(let i=0;i<N;i++)for(let j=0;j<N;j++)if(i!==j){{err+=(M[i][j]-P.C[i][j])**2;cnt++}}
document.getElementById("k_o").textContent=k;const ex=P.lam.slice(0,k).reduce((a,b)=>a+b,0)/tot;
document.getElementById("k_txt").textContent=`${{(100*ex).toFixed(1)}}% of the variance; average error per pair ${{Math.sqrt(err/cnt).toFixed(3)}}`;
Plotly.react("hm_k",[{{z:M,x:lbl,y:lbl,type:"heatmap",colorscale:cs,zmin:-0.2,zmax:1,showscale:false,hovertemplate:"%{{y}}<br>%{{x}}<br>rebuilt %{{z:.2f}}<extra></extra>"}}],
{{...lay(`Rebuilt from ${{k}} component${{k>1?"s":""}}`),xaxis:ax,yaxis:{{...ax,autorange:"reversed"}}}},{{displaylogo:false,responsive:true}})}}
document.getElementById("k").oninput=rebuild;rebuild();
const px=document.getElementById("px"),py=document.getElementById("py");
for(let c=0;c<10;c++){{const o=`<option value="${{c}}">PC${{c+1}} (${{(100*P.lam[c]/tot).toFixed(1)}}%)</option>`;px.innerHTML+=o;py.innerHTML+=o}}px.value=1;py.value=2;
function scat(){{const a=+px.value,b=+py.value;const tr=P.order.map(s=>{{const idx=P.sectors.map((x,i)=>x===s?i:-1).filter(i=>i>=0);
return {{x:idx.map(i=>P.V[i][a]*Math.sqrt(P.lam[a])),y:idx.map(i=>P.V[i][b]*Math.sqrt(P.lam[b])),text:idx.map(i=>P.tickers[i]),customdata:idx.map(i=>P.names[i]),name:s,mode:"markers+text",
textposition:"top center",textfont:{{color:P.colors[s],size:10}},marker:{{color:P.colors[s],size:9,line:{{color:"{BG}",width:1}}}},type:"scatter",hovertemplate:"<b>%{{customdata}}</b> (%{{text}})<br>%{{x:.2f}}, %{{y:.2f}}<extra>"+s+"</extra>"}}}});
Plotly.react("scatter",tr,{{...lay(""),margin:{{l:60,r:10,t:10,b:50}},xaxis:{{title:{{text:px.selectedOptions[0].text}},gridcolor:"{GRID}",zerolinecolor:"{MUTED}"}},
yaxis:{{title:{{text:py.selectedOptions[0].text}},gridcolor:"{GRID}",zerolinecolor:"{MUTED}"}},legend:{{font:{{color:"{TEXT2}"}}}}}},{{displaylogo:false,responsive:true}})}}
px.onchange=scat;py.onchange=scat;scat();
</script></body></html>"""
    (ROOT / "dashboard.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
