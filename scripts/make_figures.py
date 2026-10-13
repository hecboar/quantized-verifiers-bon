"""Paper figures (PDF + PNG in paper/figures/). CPU only, from stored scores and tables.

Fig. 1 frontier.pdf   accuracy vs measured verifier memory of deployed configurations
Fig. 2 curves.pdf     Best-of-N accuracy vs N per verifier and precision
Fig. 3 mechanism.pdf  winner change vs accuracy change, all conditions and studies
Fig. 4 retention.pdf  verifier advantage over majority vote, BF16 vs GPTQ-W4, per regime

Palette: dataviz reference categorical slots 1-4 (validated; slots 3-4 below 3:1 on
white, so every series is also direct-labelled or marker-coded).
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vq.analysis import score_matrix  # noqa: E402
from vq.report import _acc_bon  # noqa: E402
from vq.store import CampaignStore  # noqa: E402

OUT = Path("paper/figures")
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8d8c88", "#e6e5e1"
PREC = {"bf16": ("BF16", BLUE, "o"), "nf4": ("NF4", AQUA, "s"), "gptq-w4": ("GPTQ-W4", ORANGE, "D"),
        "gptq-w3": ("GPTQ-W3", YELLOW, "v")}
VNAME = {"sky-1.7b": "Skywork ORM 1.7B", "sky-4b": "Skywork ORM 4B", "sky-8b": "Skywork ORM 8B",
         "prm-7b": "Qwen2.5-Math PRM 7B"}
G = "qwen2.5-1.5b"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.6, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def majority(K, C):
    out = np.zeros(len(K))
    for i, (k, c) in enumerate(zip(K, C)):
        v = k[k >= 0]
        if len(v):
            ids, first, counts = np.unique(v, return_index=True, return_counts=True)
            w = ids[counts == counts.max()][np.argmin(first[counts == counts.max()])]
            out[i] = float(c[k == w].any())
    return out


def acc32(st, gen, v, cond):
    S, C, K, q = score_matrix(st, gen, v, cond, "eval")
    return _acc_bon(S, C, 32, 200, 7, 11), C, K, q


# ---------------------------------------------------------------- Fig. 1
def fig_frontier(st):
    mem = json.loads(Path("outputs/memory.json").read_text())
    dep = {"bf16": "bf16", "nf4": "nf4-bnb", "gptq-w4": "gptq-w4-packed"}
    rows = []
    for key, m in mem.items():
        v, scheme = key.split("__")
        a, *_ = acc32(st, G, v, dep[scheme])
        rows.append({"v": v, "scheme": scheme, "mem": m["peak_gb"], "acc": 100 * a.mean()})
    # PRM (revision measurement, same workload and GPU type): BF16 and bitsandbytes NF4, whose
    # scores are bit-identical to the simulated NF4 condition
    rev = Path("outputs/memory_revision.json")
    if rev.exists():
        mr = json.loads(rev.read_text())
        for scheme in ("bf16", "nf4"):
            key = f"prm-7b__{scheme}"
            if key in mr and "peak_gb" in mr[key]:
                a, *_ = acc32(st, G, "prm-7b", scheme)
                rows.append({"v": "prm-7b", "scheme": scheme, "mem": mr[key]["peak_gb"], "acc": 100 * a.mean()})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for c in (4, 6, 8, 12, 16):
        ax.axvline(c, color=GRID if c != 6 else MUTED, lw=0.8, ls="--" if c == 6 else "-", zorder=0)
        ax.text(c, 77.15, f"{c} GiB", ha="center", va="bottom", fontsize=7, color=INK2)
    front = df.sort_values("mem")
    best, fx, fy = -1, [], []
    for r in front.itertuples():
        if r.acc > best:
            best = r.acc
            fx.append(r.mem)
            fy.append(r.acc)
    ax.step(fx + [17.5], fy + [fy[-1]], where="post", color=MUTED, lw=1.0, zorder=1)
    mv = pd.read_csv("outputs/tables/revision/majority.csv")
    mv32 = 100 * mv[(mv.population == "main_all_2000") & (mv.N == 32)].majority.iloc[0]
    ax.axhline(mv32, color=INK2, lw=0.9, ls=":", zorder=1)
    ax.text(17.3, mv32 + 0.08, f"majority vote@32, no verifier ({mv32:.1f}%)", ha="right", va="bottom",
            fontsize=6.5, color=INK2)
    for s in ("bf16", "nf4", "gptq-w4"):
        name, col, mk = PREC[s]
        d = df[df.scheme == s]
        ax.scatter(d.mem, d.acc, color=col, marker=mk, s=38 if s != "nf4" else 26, edgecolor="white",
                   linewidth=1.2, zorder=4 if s == "nf4" else 3,
                   label=name + (" (packed)" if s == "gptq-w4" else ""))
        for r in d.itertuples():
            lab = "PRM" if r.v == "prm-7b" else r.v.replace("sky-", "").upper()
            ax.annotate(lab, (r.mem, r.acc), xytext=(4, -9 if s == "nf4" else 4),
                        textcoords="offset points", fontsize=6.5, color=INK2)
    ax.set_xlabel("Peak verifier memory (GiB, 64 x 900-token batch)")
    ax.set_ylabel("Best-of-32 accuracy (%)")
    ax.set_xlim(1.5, 17.5)
    ax.set_ylim(69.5, 77.7)
    ax.legend(loc="lower right", fontsize=7)
    save(fig, "frontier")
    df.to_csv(OUT / "frontier_data.csv", index=False)


# ---------------------------------------------------------------- Fig. 2
def fig_curves():
    m = pd.read_csv("outputs/tables/main.csv")
    mv = pd.read_csv("outputs/tables/revision/majority.csv")
    mv = mv[mv.population == "main_all_2000"].sort_values("N")
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4), sharey=True)
    for ax, v in zip(axes, VNAME):
        d = m[m.verifier == v]
        ref = d[d.condition == "gptq-w4"].sort_values("N")
        ax.plot(mv.N, 100 * mv.majority, color=MUTED, ls=":", lw=1.2)
        ax.plot(ref.N, 100 * ref.acc_ref, color=BLUE, marker="o", ms=3)
        for c in ("nf4", "gptq-w4", "gptq-w3"):
            name, col, mk = PREC[c]
            x = d[d.condition == c].sort_values("N")
            ax.plot(x.N, 100 * x.acc_q, color=col, marker=mk, ms=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
        ax.set_title(VNAME[v], fontsize=8)
        ax.set_xlabel("N")
    axes[0].set_ylabel("Best-of-N accuracy (%)")
    handles = [plt.Line2D([], [], color=PREC[c][1], marker=PREC[c][2], ms=3, label=PREC[c][0])
               for c in ("bf16", "nf4", "gptq-w4", "gptq-w3")]
    handles.append(plt.Line2D([], [], color=MUTED, ls=":", label="Majority vote@N"))
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.08), fontsize=7)
    save(fig, "curves")


# ---------------------------------------------------------------- Fig. 3
def bit_class(c):
    if c in ("bf16-null",):
        return "BF16 null control"
    if c in ("rtn-w8",):
        return "8-bit"
    if "w3" in c:
        return "3-bit"
    return "4-bit"


def fig_mechanism():
    m = pd.read_csv("outputs/tables/main.csv")
    m = m[(m.N == 32) & (m.condition != "bf16") & (m.condition != "bf16-repeat")]
    pts = [{"study": "Main", "v": r.verifier, "c": r.condition, "wc": 100 * r.winner_change,
            "d": 100 * r.delta, "lo": 100 * r.delta_lo, "hi": 100 * r.delta_hi} for r in m.itertuples()]
    for study, tag in (("H6", "h6"), ("H7", "h7")):
        conf = json.loads(Path(f"outputs/tables/{tag}/{tag}_confirmatory.json").read_text())["per_verifier"]
        ex = pd.read_csv(f"outputs/tables/{tag}/{tag}_exploratory.csv")
        for v, r in conf.items():
            pts.append({"study": study, "v": v, "c": "gptq-w4", "wc": 100 * r["winner_change"],
                        "d": 100 * r["delta"], "lo": 100 * r["delta_ci95"][0], "hi": 100 * r["delta_ci95"][1]})
            for c in ("bf16-null", "gptq-w3"):
                e = ex[(ex.verifier == v) & (ex.analysis == f"{c}:delta")].iloc[0]
                lo, hi = (float(x) for x in e.ci95.strip("[]").split(","))
                pts.append({"study": study, "v": v, "c": c, "wc": 100 * e.winner_change, "d": 100 * e.estimate,
                            "lo": 100 * lo, "hi": 100 * hi})
    df = pd.DataFrame(pts)
    df["cls"] = df.c.map(bit_class)
    colors = {"BF16 null control": MUTED, "8-bit": AQUA, "4-bit": BLUE, "3-bit": ORANGE}
    marks = {"Main": "o", "H6": "s", "H7": "^"}
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axhspan(-3, 3, color="#f1f0ec", zorder=0)
    ax.text(1, 2.6, "±3 pp equivalence margin", fontsize=6.5, color=INK2, va="top")
    for (cls, study), d in df.groupby(["cls", "study"]):
        ax.errorbar(d.wc, d.d, yerr=[d.d - d.lo, d.hi - d.d], fmt="none", ecolor=colors[cls], elinewidth=0.7,
                    alpha=0.6, zorder=2)
        ax.scatter(d.wc, d.d, color=colors[cls], marker=marks[study], s=26, edgecolor="white", linewidth=1.0,
                   zorder=3)
    out = df[(df.v == "sky-1.7b") & (df.c == "gptq-w3")].iloc[0]
    ax.annotate("1.7B ORM, GPTQ-W3", (out.wc, out.d), xytext=(-70, 8), textcoords="offset points", fontsize=6.5,
                color=INK2, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6})
    ax.set_xlabel("Winner change vs BF16 (% of problems)")
    ax.set_ylabel(r"$\Delta$ Best-of-32 accuracy (pp)")
    ax.set_xlim(0, 92)
    ax.set_ylim(-15, 4)
    h1 = [plt.Line2D([], [], color=colors[k], marker="o", ls="", ms=5, label=k) for k in colors]
    h2 = [plt.Line2D([], [], color=INK2, marker=marks[k], ls="", ms=5, label=k) for k in marks]
    ax.legend(handles=h1 + h2, loc="lower left", fontsize=6.5, ncol=2)
    save(fig, "mechanism")
    df.to_csv(OUT / "mechanism_data.csv", index=False)


# ---------------------------------------------------------------- Fig. 4
def _boot_ci(x, n=10_000, seed=7):
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(b, [2.5, 97.5])


def fig_retention(st):
    lv = json.loads(Path("data/math_levels.json").read_text())
    regimes = [("Main study, all problems", G, None), ("Main study, MATH Level 5", G, "L5"),
               ("H7: weak generator, new Levels 4-5", "qwen2.5-1.5b-h7", "data/manifest_h7.json"),
               ("H6: strong generator, new Level 5", "qwen2.5-math-7b", "data/manifest_h6_hard.json")]
    rows = []
    for label, gen, sel in regimes:
        for v in ("prm-7b", "sky-8b"):
            a0, C, K, q = acc32(st, gen, v, "bf16")
            a1, *_ = acc32(st, gen, v, "gptq-w4")
            mj = majority(K, C)
            if sel is None:
                msk = np.ones(len(q), bool)
            elif sel == "L5":
                msk = np.array([lv.get(x) == 5 for x in q])
            else:
                keep = set(json.loads(Path(sel).read_text())["ids"])
                msk = np.array([x in keep for x in q])
            g0, g1 = 100 * (a0 - mj)[msk], 100 * (a1 - mj)[msk]
            (l0, h0), (l1, h1) = _boot_ci(g0), _boot_ci(g1)
            rows.append({"regime": label, "v": v, "n": int(msk.sum()), "bf16": g0.mean(), "bf16_lo": l0,
                         "bf16_hi": h0, "w4": g1.mean(), "w4_lo": l1, "w4_hi": h1,
                         "confirmatory": label.startswith("H")})
    df = pd.DataFrame(rows)
    order = [(r[0], v) for r in regimes for v in ("prm-7b", "sky-8b")]
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    ax.axvline(0, color=INK2, lw=0.8)
    for y, (reg, v) in enumerate(order):
        r = df[(df.regime == reg) & (df.v == v)].iloc[0]
        ax.errorbar(r.bf16, y - 0.14, xerr=[[r.bf16 - r.bf16_lo], [r.bf16_hi - r.bf16]], fmt="o", mfc="none",
                    mec=BLUE, ecolor=BLUE, mew=1.4, ms=6, elinewidth=1.0, zorder=2)
        ax.errorbar(r.w4, y + 0.14, xerr=[[r.w4 - r.w4_lo], [r.w4_hi - r.w4]], fmt="D", color=ORANGE,
                    mec="white", ms=5, elinewidth=1.0, zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{reg} | {'PRM' if v == 'prm-7b' else 'ORM 8B'}" for reg, v in order], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("Best-of-32 accuracy minus majority vote@32 (pp), 95% bootstrap CI")
    ax.legend(handles=[plt.Line2D([], [], color=BLUE, marker="o", mfc="none", mew=1.4, ls="", label="BF16"),
                       plt.Line2D([], [], color=ORANGE, marker="D", ls="", label="GPTQ-W4")],
              loc="lower right", fontsize=7)
    save(fig, "retention")
    df.to_csv(OUT / "retention_data.csv", index=False)


if __name__ == "__main__":
    st = CampaignStore("outputs/store")
    fig_curves()
    fig_mechanism()
    fig_frontier(st)
    fig_retention(st)
    print("figures written to", OUT)
