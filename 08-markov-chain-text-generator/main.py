"""
Markov Chain Text Generator from scratch: transition matrix, stationary distribution and a Shakespeare generator.
Run:  python main.py      (about a minute on a laptop CPU)
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go

from src.markov import (MarkovChain, InterpolatedLM, word_tokens, detokenize, transition_matrix, stationary_power,
                        stationary_eigen, entropy_rate, copy_rate)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

BG, CARD, TEXT, TEXT2, MUTED, GRID = "#1a1a19", "#222220", "#f0efec", "#c3c2b7", "#8a8980", "#3a3936"
BLUE, ORANGE, AQUA, AMBER, PINK, VIOLET, RED = "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9", "#e66767"
SEQ = LinearSegmentedColormap.from_list("seq", [CARD, "#1f3b5c", BLUE, "#b9d6f8"])

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
    text = (ROOT / "data" / "shakespeare.txt").read_text(encoding="utf-8")
    n = len(text)
    train, valid, test = text[:int(0.85 * n)], text[int(0.85 * n):int(0.9 * n)], text[int(0.9 * n):]
    print(f"{n:,} characters, {len(word_tokens(text)):,} word tokens; train 85% / validation 5% / test 10% (contiguous)")

    # ------------------------------------------------------------- part 1: the 27-state letter chain
    letters = re.sub(r"[^a-z]+", " ", train.lower())
    states = list(" abcdefghijklmnopqrstuvwxyz")
    labels = ["space"] + states[1:]
    P, counts = transition_matrix(letters, states)
    pi_pow, pow_hist = stationary_power(P)
    pi_eig, eigvals = stationary_eigen(P)
    empirical = np.array([letters.count(s) for s in states], float); empirical /= empirical.sum()
    mods = np.sort(np.abs(eigvals))[::-1]
    lam2 = float(mods[1])
    H_rate = entropy_rate(P, pi_pow)
    H_iid = float(-(empirical * np.log2(empirical)).sum())
    steps = 16
    starts = ["q", "z", "x", " "]
    evo = {}
    for s0 in starts:
        d = np.zeros(27); d[states.index(s0)] = 1
        rows = [d]
        for _ in range(steps):
            d = d @ P; rows.append(d)
        evo[s0] = np.array(rows)
    tv = {s0: 0.5 * np.abs(evo[s0] - pi_pow).sum(1) for s0 in starts}
    print(f"Letter chain: |lambda2| = {lam2:.3f}, entropy rate {H_rate:.3f} bits vs {H_iid:.3f} for independent letters; "
          f"power iteration vs eigenvector {np.abs(pi_pow - pi_eig).max():.1e}, vs letter frequencies {np.abs(pi_pow - empirical).max():.1e}")

    # ------------------------------------------------------------- part 2: how much context helps (Shannon's experiment)
    tr_c, va_c, te_c = list(train), list(valid), list(test)
    char_orders = list(range(0, 11))
    lm_char = InterpolatedLM(10).fit(tr_c)
    lam_char = lm_char.fit_weights(va_c[:60_000])
    test_bits_char = []
    for k in char_orders:
        sub = InterpolatedLM(k)
        sub.chains, sub.totals, sub.V, sub.N = lm_char.chains[:k + 1], lm_char.totals[:k + 1], lm_char.V, lm_char.N
        sub.lam = np.full(k + 1, 1 / (k + 1)); sub.fit_weights(va_c[:60_000])
        test_bits_char.append(sub.cross_entropy(te_c[:60_000]))
    train_bits_char, contexts_char, coverage_char = [], [], []
    te_set = te_c[:60_000]
    for k in char_orders:
        ch = lm_char.chains[k]
        tot = lm_char.totals[k]
        h = 0.0
        for ctx, c in ch.counts.items():
            T = tot[ctx]
            v = np.array(list(c.values()), float)
            h -= (v * np.log2(v / T)).sum()
        train_bits_char.append(h / sum(tot.values()))
        contexts_char.append(len(ch.counts))
        coverage_char.append(float(np.mean([tuple(te_set[i - k:i]) in ch.counts for i in range(k, len(te_set))])) if k else 1.0)
    best_k_char = int(np.argmin(test_bits_char))
    print("char bits:", [round(b, 3) for b in test_bits_char])

    # ------------------------------------------------------------- part 3: word chains, perplexity vs originality
    w_tr, w_va, w_te = word_tokens(train), word_tokens(valid), word_tokens(test)
    word_orders = [1, 2, 3, 4]
    lm_word = InterpolatedLM(4).fit(w_tr)
    word_ppl, word_lams = [], []
    for k in [0] + word_orders:
        sub = InterpolatedLM(k)
        sub.chains, sub.totals, sub.V, sub.N = lm_word.chains[:k + 1], lm_word.totals[:k + 1], lm_word.V, lm_word.N
        sub.lam = np.full(k + 1, 1 / (k + 1)); lam = sub.fit_weights(w_va)
        word_ppl.append(2 ** sub.cross_entropy(w_te)); word_lams.append(lam.round(3).tolist())
    six = {tuple(w_tr[i:i + 6]) for i in range(len(w_tr) - 5)}
    rng = np.random.default_rng(7)
    copy, samples_word = [], {}
    for k in word_orders:
        gen = lm_word.chains[k].generate(3000, temperature=1.0, rng=rng)
        copy.append(copy_rate(gen, six, 6))
        seed = ["ROMEO", ":", "\n"][-k:] if k <= 3 else ["\n", "ROMEO", ":", "\n"]
        samples_word[k] = detokenize(lm_word.chains[k].generate(70, seed=seed, rng=np.random.default_rng(11)))
    print("word perplexity:", [round(p, 1) for p in word_ppl], "copy rate:", [round(c, 3) for c in copy])
    samples_char = {}
    for k in [1, 3, 5, 8, 12]:
        ch = lm_char.chains[k] if k <= 10 else MarkovChain(k).fit(tr_c)
        seed = list("ROMEO:\n")[-k:] if k <= 7 else list("\n\nROMEO:\n")[-k:]
        if len(seed) < k:
            seed = list(train[:k])
        samples_char[k] = "".join(ch.generate(260, seed=seed, rng=np.random.default_rng(5)))
    next_word = {}
    for ctx in [("my", "lord"), ("I", "will"), ("the", "king"), ("to", "be")]:
        d = lm_word.chains[2].distribution(list(ctx))
        next_word[" ".join(ctx)] = sorted(d.items(), key=lambda x: -x[1])[:8] if d else []

    # ------------------------------------------------------------- chart 1: 3D convergence to the stationary distribution
    Z = evo["q"]
    fig3d = go.Figure(go.Surface(
        x=np.arange(27), y=np.arange(steps + 1), z=100 * Z, colorscale=[[0, "#1f3b5c"], [0.25, BLUE], [1, "#e9f1fc"]],
        cmin=0, cmax=30, showscale=False, opacity=0.97,
        contours=dict(y=dict(show=True, color=BG, width=1, start=0, end=steps, size=1)),
        customdata=np.tile(np.array(labels), (steps + 1, 1)),
        hovertemplate="Step %{y}<br>Next character: %{customdata}<br>Probability %{z:.1f}%<extra></extra>"))
    fig3d.add_trace(go.Scatter3d(x=np.arange(27), y=np.full(27, steps + 0.3), z=100 * pi_pow, mode="lines+markers",
                                 line=dict(color=AMBER, width=6), marker=dict(size=3, color=AMBER),
                                 name="Stationary distribution", hovertemplate="%{text}: %{z:.1f}%<extra>stationary</extra>", text=labels))
    axis = dict(backgroundcolor=BG, gridcolor=GRID, zerolinecolor=GRID, color=TEXT2)
    fig3d.update_layout(**plotly_layout(
        title=dict(text="Starting from the letter q: the chain forgets where it began and settles into its stationary distribution", x=0.02),
        scene=dict(xaxis=dict(title="State (character)", tickvals=list(range(27)), ticktext=["\u2423"] + states[1:], **axis),
                   yaxis=dict(title="Steps after q", **axis), zaxis=dict(title="Probability (%)", **axis),
                   aspectratio=dict(x=1.7, y=1.1, z=0.7), camera=dict(eye=dict(x=1.55, y=-1.55, z=0.85))),
        legend=dict(font=dict(color=TEXT2), x=0.01, y=0.92), height=780))
    fig3d.write_html(OUT / "01_convergence_3d.html", include_plotlyjs="cdn")
    try:
        fig3d.write_image(OUT / "01_convergence_3d.png", width=1600, height=900, scale=2)
    except Exception as exc:  # noqa: BLE001
        print(f"  (static 3D export skipped: {exc.__class__.__name__})")

    # ------------------------------------------------------------- chart 2: transition matrix + stationary distribution
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.4), gridspec_kw={"width_ratios": [1.15, 1]})
    im = axes[0].imshow(P, cmap=SEQ, vmin=0, vmax=np.percentile(P, 99.5))
    axes[0].set_xticks(range(27), ["\u2423"] + states[1:]); axes[0].set_yticks(range(27), ["\u2423"] + states[1:])
    axes[0].tick_params(colors=TEXT, length=0, labelsize=9.5)
    axes[0].set_xlabel("Next character"); axes[0].set_ylabel("Current character")
    for sp in axes[0].spines.values():
        sp.set_visible(False)
    qi = states.index("q")
    axes[0].add_patch(plt.Rectangle((-0.5, qi - 0.5), 27, 1, fill=False, ec=AMBER, lw=1.5))
    axes[0].text(0.2, qi, f"after q: u {100 * P[qi, states.index('u')]:.0f}%", color=AMBER, va="center", fontsize=9.5, fontweight="bold")
    cb = fig.colorbar(im, ax=axes[0], fraction=0.035, pad=0.09); cb.outline.set_visible(False); cb.ax.tick_params(colors=MUTED)
    title(axes[0], "Transition matrix P: each row is the distribution of the next character")
    order = np.argsort(-pi_pow)
    x = np.arange(27)
    axes[1].bar(x - 0.2, 100 * pi_pow[order], width=0.4, color=BLUE, label="Stationary distribution (pi = pi P)")
    axes[1].bar(x + 0.2, 100 * empirical[order], width=0.4, color=MUTED, label="Character frequency in the text")
    axes[1].set_xticks(x, [labels[i] if i else "\u2423" for i in order]); axes[1].tick_params(axis="x", colors=TEXT, length=0)
    axes[1].set_ylabel("Share (%)"); axes[1].legend(frameon=False, labelcolor=TEXT2); axes[1].grid(axis="y", lw=0.5)
    title(axes[1], f"Stationary distribution = how often each character appears (max gap {np.abs(pi_pow - empirical).max():.0e})")
    fig.tight_layout()
    fig.savefig(OUT / "02_transition_matrix_and_stationary.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 3: mixing and spectrum
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), gridspec_kw={"width_ratios": [1.3, 1]})
    for s0, col in zip(starts, [AMBER, PINK, VIOLET, AQUA]):
        axes[0].semilogy(range(steps + 1), np.maximum(tv[s0], 1e-16), color=col, lw=2.2, marker="o", ms=3.5,
                         label=f"start at '{s0 if s0 != ' ' else 'space'}'")
    axes[0].semilogy(range(steps + 1), lam2 ** np.arange(steps + 1), color=MUTED, ls="--", lw=1.4, label=f"|lambda_2|^t = {lam2:.3f}^t")
    axes[0].set_xlabel("Steps"); axes[0].set_ylabel("Distance to the stationary distribution (total variation)")
    axes[0].legend(frameon=False, labelcolor=TEXT2); axes[0].grid(lw=0.5)
    title(axes[0], "Mixing: from any start, the error shrinks geometrically")
    th = np.linspace(0, 2 * np.pi, 400)
    axes[1].plot(np.cos(th), np.sin(th), color=GRID, lw=1)
    axes[1].scatter(eigvals.real, eigvals.imag, color=BLUE, s=36, zorder=3, edgecolor=BG)
    axes[1].scatter([1], [0], color=AMBER, s=90, zorder=4, edgecolor=BG)
    axes[1].annotate("eigenvalue 1:\nstationary distribution", (1, 0), xytext=(0.15, 0.55), color=AMBER,
                     arrowprops=dict(arrowstyle="-", color=AMBER))
    circ = plt.Circle((0, 0), lam2, fill=False, ec=MUTED, ls="--", lw=1)
    axes[1].add_patch(circ)
    axes[1].text(-lam2 * 0.72, lam2 * 0.75, f"|lambda_2| = {lam2:.3f}", color=TEXT2, fontsize=9.5)
    axes[1].set_aspect("equal"); axes[1].set_xlim(-1.1, 1.1); axes[1].set_ylim(-1.1, 1.1); axes[1].grid(lw=0.5)
    axes[1].set_xlabel("Real part"); axes[1].set_ylabel("Imaginary part")
    title(axes[1], "Eigenvalues of P in the complex plane")
    fig.tight_layout()
    fig.savefig(OUT / "03_mixing_and_spectrum.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 4: Shannon's experiment
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.3, 1]})
    axes[0].plot(char_orders, test_bits_char, color=BLUE, lw=2.4, marker="o", label="Unseen text (interpolated model)")
    axes[0].plot(char_orders, train_bits_char, color=MUTED, lw=2, marker="o", ls="--", label="Training text (pure order-k counts)")
    axes[0].scatter([best_k_char], [test_bits_char[best_k_char]], s=150, facecolors="none", edgecolors=AMBER, lw=2, zorder=5)
    axes[0].annotate(f"{test_bits_char[best_k_char]:.2f} bits per character", (best_k_char, test_bits_char[best_k_char]),
                     xytext=(10, 22), textcoords="offset points", color=AMBER)
    axes[0].set_xlabel("Characters of context (order k)"); axes[0].set_ylabel("Bits per character")
    axes[0].legend(frameon=False, labelcolor=TEXT2); axes[0].grid(lw=0.5); axes[0].set_xticks(char_orders)
    title(axes[0], "More context, less surprise, until the text runs out")
    axes[1].plot(char_orders, 100 * np.array(coverage_char), color=AQUA, lw=2.4, marker="o")
    for k in (4, 6, 8, 10):
        axes[1].text(k, 100 * coverage_char[k] + 3, f"{100 * coverage_char[k]:.0f}%", ha="center", color=TEXT, fontsize=9.5)
    axes[1].set_ylim(0, 108); axes[1].set_xticks(char_orders)
    axes[1].set_xlabel("Characters of context (order k)"); axes[1].set_ylabel("Unseen contexts already seen in training (%)")
    axes[1].grid(lw=0.5)
    title(axes[1], "Why: long contexts have never been seen before")
    fig.tight_layout()
    fig.savefig(OUT / "04_context_vs_surprise.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- chart 5: words, perplexity vs originality
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    axes[0].bar([str(k) for k in [0] + word_orders], word_ppl, color=[MUTED] + [BLUE] * 4, width=0.6)
    for i, p_ in enumerate(word_ppl):
        axes[0].text(i, p_ + 6, f"{p_:.0f}", ha="center", color=TEXT)
    axes[0].set_xlabel("Words of context (order)"); axes[0].set_ylabel("Perplexity on unseen text"); axes[0].grid(axis="y", lw=0.5)
    title(axes[0], "Perplexity: 'how many words it hesitates between'")
    axes[1].plot(word_orders, 100 * np.array(copy), color=RED, lw=2.4, marker="o")
    for k, c in zip(word_orders, copy):
        axes[1].text(k, 100 * c + 3.5, f"{100 * c:.0f}%", ha="center", color=TEXT)
    axes[1].set_ylim(0, 108); axes[1].set_xticks(word_orders)
    axes[1].set_xlabel("Words of context (order)"); axes[1].set_ylabel("Generated 6-word sequences copied from Shakespeare (%)")
    axes[1].grid(lw=0.5)
    title(axes[1], "Originality: higher orders start to plagiarize")
    ctx = "I will"
    items = next_word[ctx]
    axes[2].barh([repr(w)[1:-1] if w != "\n" else "\u21b5" for w, _ in items][::-1], [100 * p_ for _, p_ in items][::-1], color=AQUA, height=0.6)
    for i, (_, p_) in enumerate(items[::-1]):
        axes[2].text(100 * p_ + 0.4, i, f"{100 * p_:.1f}%", va="center", color=TEXT, fontsize=9.5)
    axes[2].tick_params(axis="y", colors=TEXT, length=0); axes[2].grid(axis="x", lw=0.5)
    axes[2].set_xlabel("P(next word | 'I will') in the order-2 chain (%)")
    title(axes[2], "One row of the word transition matrix")
    fig.tight_layout()
    fig.savefig(OUT / "05_words_perplexity_vs_originality.png", dpi=170); plt.close(fig)

    # ------------------------------------------------------------- metrics
    metrics = {
        "data": {"characters": n, "word_tokens_train": len(w_tr), "split": "85% train / 5% validation / 10% test, contiguous"},
        "letter_chain": {"states": 27, "lambda2_modulus": round(lam2, 4), "entropy_rate_bits": round(H_rate, 4),
                         "independent_letters_entropy_bits": round(H_iid, 4),
                         "power_iteration_steps": len(pow_hist),
                         "stationary_power_vs_eigenvector_max_diff": float(np.abs(pi_pow - pi_eig).max()),
                         "stationary_vs_letter_frequency_max_diff": float(np.abs(pi_pow - empirical).max()),
                         "p_u_after_q": round(float(P[qi, states.index('u')]), 4),
                         "steps_to_tv_below_1pct_from_q": int(np.argmax(tv["q"] < 0.01)),
                         "stationary": dict(zip(labels, pi_pow.round(5).tolist()))},
        "characters": {"orders": char_orders, "test_bits_per_char": [round(b, 4) for b in test_bits_char],
                       "train_bits_per_char": [round(b, 4) for b in train_bits_char], "contexts": contexts_char,
                       "test_context_coverage": [round(c, 4) for c in coverage_char], "interpolation_weights_order10": lam_char.round(3).tolist()},
        "words": {"orders": [0] + word_orders, "test_perplexity": [round(p, 2) for p in word_ppl], "interpolation_weights": word_lams,
                  "copy_rate_6grams": dict(zip(word_orders, [round(c, 4) for c in copy]))},
        "next_word_examples": {k: [[w, round(p, 4)] for w, p in v] for k, v in next_word.items()},
        "samples": {"characters": {str(k): v for k, v in samples_char.items()}, "words": {str(k): v for k, v in samples_word.items()}},
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT / "samples.txt").write_text("\n\n".join([f"=== characters, order {k} ===\n{v}" for k, v in samples_char.items()] +
                                                 [f"=== words, order {k} ===\n{v}" for k, v in samples_word.items()]), encoding="utf-8")

    # ------------------------------------------------------------- dashboard (generator runs in the browser on the same corpus)
    (OUT / "corpus.js").write_text("window.CORPUS=" + json.dumps(train) + ";", encoding="utf-8")
    kpis = {"Training text": f"{len(train):,} characters", "Best bits per character": f"{test_bits_char[best_k_char]:.2f} (order {best_k_char})",
            "Word perplexity (order 2)": f"{word_ppl[2]:.0f}", "Mixing rate |lambda2|": f"{lam2:.2f}"}
    kpi_html = "".join(f'<div class="kpi"><span>{k}</span><b>{v}</b></div>' for k, v in kpis.items())
    sec = lambda t, n_, f: f'<section><h2>{t}</h2><p class="note">{n_}</p>{f.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})}</section>'
    img = lambda t, n_, src: f'<section><h2>{t}</h2><p class="note">{n_}</p><img src="outputs/{src}" alt="{t}"></section>'
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Markov Chain Text Generator</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><script src="outputs/corpus.js"></script>
<style>:root{{--bg:{BG};--card:{CARD};--text:{TEXT};--text2:{TEXT2};--muted:{MUTED};--line:{GRID}}}*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font-family:Inter,"Segoe UI",Arial,sans-serif}}
header,.kpis,footer{{max-width:1240px;margin:auto;padding:0 32px}}header{{padding-top:48px}}h1{{font-size:40px;margin:0 0 8px}}
header p{{color:var(--text2);max-width:860px;line-height:1.55;margin:0}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-top:24px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px}}
.kpi span{{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.8px}}.kpi b{{display:block;font-size:22px;margin-top:6px}}
section{{max-width:1240px;margin:28px auto;padding:20px 24px;background:var(--card);border:1px solid var(--line);border-radius:14px}}
section img{{width:100%;border-radius:8px}}h2{{margin:4px 0 6px;font-size:20px}}.note{{color:var(--text2);margin:0 0 10px;font-size:14px}}
.controls{{display:flex;flex-wrap:wrap;gap:18px;align-items:center;color:var(--text2);font-size:14px;margin:8px 0 14px}}
select,input[type=text]{{background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font:inherit}}
input[type=range]{{accent-color:{BLUE};width:150px;vertical-align:middle}}button{{background:{BLUE};color:white;border:0;border-radius:8px;padding:9px 18px;font:inherit;font-weight:600;cursor:pointer}}
pre{{white-space:pre-wrap;background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:16px;font:15px/1.6 Georgia,"Times New Roman",serif;color:var(--text);min-height:220px;margin:0}}
.grid2{{display:grid;grid-template-columns:1.6fr 1fr;gap:16px}}.bar{{display:flex;align-items:center;gap:8px;margin:4px 0;font-size:13px;color:var(--text2)}}
.bar i{{display:block;height:12px;background:{AQUA};border-radius:3px}}.bar b{{min-width:90px;color:var(--text);font-weight:500;font-family:monospace}}
.stat{{color:var(--muted);font-size:13px;margin-top:8px}}footer{{color:var(--muted);font-size:13px;margin:24px auto 48px}}a{{color:#7fb0f0}}
@media(max-width:760px){{.grid2{{grid-template-columns:1fr}}header,.kpis,footer{{padding:0 16px}}section{{margin:16px;padding:12px}}h1{{font-size:30px}}}}</style></head><body>
<header><h1>Markov Chain Text Generator</h1><p>Text generated one token at a time from transition probabilities counted in {len(train):,} characters of Shakespeare.
Transition matrices, stationary distributions, mixing and an interpolated language model, all written from scratch.</p></header>
<div class="kpis">{kpi_html}</div>
<section><h2>Generate Shakespeare</h2><p class="note">The chain is built in your browser from the training text. Low orders babble; high orders write convincing lines but start copying the original.</p>
<div class="controls"><span>Level <select id="lvl"><option value="char">Characters</option><option value="word" selected>Words</option></select></span>
<span>Order <input type="range" id="ord" min="1" max="10" value="2"> <b id="ord_o"></b></span>
<span>Temperature <input type="range" id="tmp" min="0.3" max="2" step="0.1" value="1"> <b id="tmp_o"></b></span>
<span>Start <input type="text" id="seed" value="ROMEO:" size="14"></span><button id="go">Generate</button></div>
<div class="grid2"><div><pre id="out"></pre><div class="stat" id="stat"></div></div><div><div class="note" id="dist_t"></div><div id="dist"></div></div></div></section>
{sec("Convergence in 3D", "Drag to rotate. Each slice is the probability of every character k steps after a q; within a few steps it no longer depends on the q.", fig3d)}
{img("Transition matrix and stationary distribution", "The 27-state letter chain and its long-run distribution.", "02_transition_matrix_and_stationary.png")}
{img("Mixing and spectrum", "How fast the chain forgets its starting point, explained by its second eigenvalue.", "03_mixing_and_spectrum.png")}
{img("Context versus surprise", "Bits needed per character as the chain looks further back.", "04_context_vs_surprise.png")}
{img("Words: perplexity versus originality", "The trade-off every n-gram model faces.", "05_words_perplexity_vs_originality.png")}
<footer>Tomás Alarcón &middot; <a href="https://github.com/tomasalarcon10/ml-from-scratch">Source code</a></footer>
<script>
const TXT=window.CORPUS,cache={{}};
const tok=(lvl,s)=>lvl==="char"?Array.from(s):(s.match(/[A-Za-z]+(?:'[A-Za-z]+)?|[.,!?;:\\-]|\\n/g)||[]);
const SEP="\\u0001";
function build(lvl,k){{const key=lvl+k;if(cache[key])return cache[key];const t=tok(lvl,TXT),m=new Map();
for(let i=k;i<t.length;i++){{const c=t.slice(i-k,i).join(SEP);let d=m.get(c);if(!d){{d=new Map();m.set(c,d)}}d.set(t[i],(d.get(t[i])||0)+1)}}
const grams=new Set();if(lvl==="word")for(let i=0;i+6<=t.length;i++)grams.add(t.slice(i,i+6).join(SEP));
return cache[key]={{m,keys:[...m.keys()],t,grams}}}}
function detok(a){{let o="";for(const w of a){{if(w==="\\n")o=o.replace(/ +$/,"")+"\\n";else if(".,!?;:".includes(w)||!o||o.endsWith("\\n"))o+=w;else o+=" "+w}}return o}}
let rs=Date.now()%1e9;const rnd=()=>(rs=(rs*1103515245+12345)%2147483648)/2147483648;
function sample(d,T){{let tot=0;const e=[...d].map(([w,c])=>{{const v=Math.pow(c,1/T);tot+=v;return [w,v]}});let r=rnd()*tot;for(const [w,v] of e){{r-=v;if(r<=0)return w}}return e[e.length-1][0]}}
function showDist(M,ctx,lvl){{const d=M.m.get(ctx.join(SEP));const el=document.getElementById("dist");const t=document.getElementById("dist_t");
const show=x=>x==="\\n"?"\\u21b5":x===" "?"\\u2423":x;t.textContent=`Next ${{lvl==="char"?"character":"word"}} after "${{ctx.map(show).join(lvl==="char"?"":" ")}}"`;
if(!d){{el.innerHTML='<div class="note">Context never seen in the text.</div>';return}}const tot=[...d.values()].reduce((a,b)=>a+b,0);
el.innerHTML=[...d].sort((a,b)=>b[1]-a[1]).slice(0,12).map(([w,c])=>`<div class="bar"><b>${{show(w)}}</b><i style="width:${{180*c/tot}}px"></i>${{(100*c/tot).toFixed(1)}}%</div>`).join("")}}
function run(){{const lvl=document.getElementById("lvl").value,k=+document.getElementById("ord").value,T=+document.getElementById("tmp").value;
const M=build(lvl,k);let out=tok(lvl,document.getElementById("seed").value+(lvl==="word"?"\\n":"\\n"));
if(out.length<k)out=M.keys[Math.floor(rnd()*M.keys.length)].split(SEP);let restarts=0;const N=lvl==="char"?600:140;
for(let i=0;i<N;i++){{let d=M.m.get(out.slice(-k).join(SEP));if(!d){{restarts++;const c=M.keys[Math.floor(rnd()*M.keys.length)];d=M.m.get(c)}}out.push(sample(d,T))}}
document.getElementById("out").textContent=lvl==="char"?out.join(""):detok(out);
let st=`${{M.keys.length.toLocaleString("en-US")}} contexts of ${{k}} ${{lvl==="char"?"characters":"words"}} in the chain`;
if(lvl==="word"){{let hit=0,tot=0;for(let i=0;i+6<=out.length;i++){{tot++;if(M.grams.has(out.slice(i,i+6).join(SEP)))hit++}}st+=` · ${{(100*hit/Math.max(1,tot)).toFixed(0)}}% of the 6-word sequences above are copied from Shakespeare`}}
document.getElementById("stat").textContent=st;showDist(M,out.slice(-k),lvl)}}
const ord=document.getElementById("ord"),lvl=document.getElementById("lvl");
function lim(){{ord.max=lvl.value==="char"?12:5;if(+ord.value>+ord.max)ord.value=ord.max;document.getElementById("ord_o").textContent=ord.value;document.getElementById("tmp_o").textContent=document.getElementById("tmp").value}}
lvl.onchange=()=>{{if(lvl.value==="char"&&+ord.value<5)ord.value=6;lim();run()}};ord.oninput=lim;document.getElementById("tmp").oninput=lim;ord.onchange=run;document.getElementById("tmp").onchange=run;
document.getElementById("go").onclick=run;lim();run();
</script></body></html>"""
    (ROOT / "dashboard.html").write_text(page, encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("letter_chain", "words")}, indent=1)[:2500])


if __name__ == "__main__":
    main()
