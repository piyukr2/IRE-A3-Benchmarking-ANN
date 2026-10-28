"""Turn results.csv into the figures and tables used in the report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import results

TABLE_DIR = config.RES_DIR / "tables"
TABLE_DIR.mkdir(exist_ok=True)

# colour-blind-safe colours (Okabe-Ito) + a different marker per series
STYLE = {"hnsw": ("#0072B2", "o"), "ivfflat": ("#D55E00", "s"), "ivfpq": ("#009E73", "^"),
         "ivfpq+refine": ("#CC79A7", "D"), "exact": ("#000000", "*")}
LABEL = {"hnsw": "HNSW", "ivfflat": "IVF-Flat", "ivfpq": "IVF-PQ", "ivfpq+refine": "IVF-PQ + re-rank",
         "exact": "Exact (1 query at a time)"}
TARGETS = [0.90, 0.95]

df = results.load()
for col in ("m", "copies"):
    if col in df:
        df[col] = df[col].astype("Int64")


def save_table(df_, name):
    df_.to_csv(TABLE_DIR / f"{name}.csv", index=False)
    (TABLE_DIR / f"{name}.md").write_text(df_.to_markdown(index=False, floatfmt=".4g")
                                          if hasattr(df_, "to_markdown") else df_.to_string(index=False))
    print(f"\n{name}\n{df_.to_string(index=False)}")


def pareto(points):
    """Keep the points not beaten on both recall and qps (the best trade-off curve)."""
    pts = sorted(points, key=lambda p: -p[0])        # high recall first
    best, out = -1, []
    for r, q in pts:
        if q > best:
            out.append((r, q))
            best = q
    return sorted(out)


def pick(group, target):
    """Smallest search parameter whose TUNING recall reaches target; its FINAL row is reported."""
    ok = group[group.recall_tune >= target].sort_values("search_param")
    return ok.iloc[0] if len(ok) else None


def style_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- Part 1
p1 = df[df.part == "part1"]
if len(p1):
    t = p1[p1["index"].str.startswith("exact")][["dataset", "index", "qps"]]
    t = t.pivot(index="dataset", columns="index", values="qps").reset_index()
    save_table(t.round(1), "part1_exact_timing")
    chk = p1[p1["index"] == "check-vs-stored"]
    if len(chk):
        save_table(chk[["dataset", "recall", "identical_sets"]], "part1_check_vs_stored")

# ---------------------------------------------------------------- Part 2
p2 = df[df.part == "part2"]
if len(p2):
    exact = p1[p1["index"] == "exact-single"].set_index("dataset")["qps"]
    qps_rows, cost_rows = [], []
    for name, d in p2.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        for idx, g in d.groupby("index"):
            colour, marker = STYLE[idx]
            curve = pareto(list(zip(g.recall, g.qps)))
            ax.plot(*zip(*curve), color=colour, marker=marker, ms=5, lw=1.6, label=LABEL[idx])
        if name in exact:
            ax.scatter([1.0], [exact[name]], color="black", marker="*", s=110, zorder=5, label=LABEL["exact"])
        for t_ in TARGETS:
            ax.axvline(t_, color="grey", lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_xlim(0.3, 1.01)
        style_axes(ax, "recall@10 (final 5,000 queries)", "queries / second (1 thread, log scale)")
        ax.set_title(f"{name.upper()}: best trade-off curve per index")
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(config.FIG_DIR / f"part2_{name}_recall_vs_qps.png", dpi=170)
        plt.close(fig)

        # tables: for each index take the build setting with the best qps at the target
        for idx, g_idx in d.groupby("index"):
            entry = {"dataset": name, "index": LABEL[idx]}
            for t_ in TARGETS:
                best = None
                for _, g in g_idx.groupby("build"):
                    r = pick(g, t_)
                    if r is not None and (best is None or r.qps > best.qps):
                        best = r
                entry[f"qps@{t_:.2f}"] = (int(round(best.qps)) if best is not None
                                          else f"not reached (max recall {g_idx.recall.max():.2f})")
                entry[f"build@{t_:.2f}"] = best.build if best is not None else "not reached"
                entry[f"recall@{t_:.2f}"] = round(best.recall, 3) if best is not None else "-"
                if t_ == 0.95:
                    reached = best
            qps_rows.append(entry)
            if reached is None:     # never reached 0.95: show the best-recall point instead
                reached = g_idx.sort_values("recall").iloc[-1]
                note = f"max recall {reached.recall:.3f} only"
            else:
                note = ""
            cost_rows.append({"dataset": name, "index": LABEL[idx], "build": reached.build,
                              "build_s": round(reached.build_s, 1), "size_mb": round(reached.size_mb),
                              "bytes_per_vec": round(reached.bytes_per_vec), "note": note})
    save_table(pd.DataFrame(qps_rows), "part2_qps_at_recall")
    save_table(pd.DataFrame(cost_rows), "part2_build_and_memory")

# ---------------------------------------------------------------- Part 3
p3 = df[df.part == "part3"].copy()
if len(p3):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    m_colour = dict(zip(sorted(p3.m.dropna().unique()), ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]))
    for (m, idx), g in p3.groupby(["m", "index"]):
        ax.plot(g.recall, g.qps, color=m_colour[m], marker="o" if idx == "ivfpq" else "D",
                ls="-" if idx == "ivfpq" else "--", ms=4, lw=1.4,
                label=f"m={m}, {'re-rank' if idx == 'ivfpq+refine' else 'plain'}")
    ax.set_yscale("log")
    style_axes(ax, "recall@10", "queries / second (1 thread, log scale)")
    ax.set_title("SIFT: IVF-PQ, plain (solid) vs re-ranked (dashed)")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = axes[1]
    summary = p3[p3.m.notna()].groupby(["m", "index"]).agg(max_recall=("recall", "max"),
                                             bytes_per_vec=("bytes_per_vec", "first")).reset_index()
    ms = sorted(int(m) for m in summary.m.unique())
    x = np.arange(len(ms))
    for off, idx in ((-0.2, "ivfpq"), (0.2, "ivfpq+refine")):
        s = summary[summary["index"] == idx].set_index("m").loc[ms]
        bars = ax.bar(x + off, s.bytes_per_vec, 0.38, color=STYLE[idx][0], label=LABEL[idx])
        for b, r in zip(bars, s.max_recall):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 6, f"{r:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x, [f"m={m}" for m in ms])
    style_axes(ax, "PQ sub-vectors", "bytes per vector (saved index)")
    ax.set_title("Memory per vector (label = best recall reached)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "part3_compression.png", dpi=170)
    plt.close(fig)

    rows = []
    for (m, idx), g_idx in p3.groupby(["m", "index"]):
        r = pick(g_idx, 0.90)
        rows.append({"m": m, "index": LABEL[idx], "bytes_per_vec": round(g_idx.bytes_per_vec.iloc[0]),
                     "max_recall": round(g_idx.recall.max(), 3),
                     "qps@0.90": int(round(r.qps)) if r is not None else "not reached"})
    save_table(pd.DataFrame(rows), "part3_summary")

# ---------------------------------------------------------------- Part 4
p4 = df[df.part == "part4"]
if len(p4):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    copies = sorted(int(c) for c in p4.copies.unique())
    colour = dict(zip(copies, ["#000000", "#0072B2", "#D55E00", "#009E73"]))
    marker = dict(zip(copies, ["o", "s", "^", "D"]))
    for ax, idx in zip(axes, ("hnsw", "ivfflat")):
        for c in copies:
            g = p4[(p4["index"] == idx) & (p4.copies == c)].sort_values("search_param")
            ax.plot(g.search_param, g.recall, color=colour[c], marker=marker[c], ms=4, lw=1.4,
                    label=f"{c} copies" + (" (control)" if c == 1 else ""))
        ax.set_xscale("log", base=2)
        style_axes(ax, "efSearch" if idx == "hnsw" else "nprobe", "recall@10")
        ax.set_title(f"{LABEL[idx]}: recall vs search effort")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "part4_near_duplicates.png", dpi=170)
    plt.close(fig)

    rows = []
    for (idx, c), g_ in p4.groupby(["index", "copies"]):
        r = pick(g_, 0.95)
        rows.append({"index": LABEL[idx], "copies": c, "n_vectors": int(g_.n_train.iloc[0]),
                     "max_recall": round(g_.recall.max(), 3),
                     "param@0.95": int(r.search_param) if r is not None else "not reached",
                     "qps@0.95": int(round(r.qps)) if r is not None else "not reached",
                     "build_s": round(g_.build_s.iloc[0], 1)})
    save_table(pd.DataFrame(rows), "part4_summary")
