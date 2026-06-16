#!/usr/bin/env python3
"""Validation DYNAMICS for animal2vec — compare runs (blue/orange) across training steps on BOTH
probes (Watkins species + signal/noise filtration), so we can decide which parallel run to keep.
Pure plotter (no GPU): reads the accumulated probe JSONs + dynamics_registry.json (tag->run,step).
Add checkpoints with run_dynamics.sh, then re-run this. Run: ~/a2v_env/bin/python a2v_dynamics.py"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
V = "/mnt/c/Users/Iaroslav/CETACEANS/a2v_validation/"
norm = lambda t: t.replace("_slim", "").replace(".pt", "")
load = lambda f: json.load(open(V + f)) if os.path.exists(V + f) else {}

REG = load("dynamics_registry.json")                       # {tag: {run, step}}
# merge both watkins JSONs (trend seed + accumulating), normalize tags
WK = {}
for src in ["animal2vec_watkins_trend.json", "animal2vec_watkins.json"]:
    for t, r in load(src).items(): WK[norm(t)] = r
FL = {norm(t): r for t, r in load("a2v_filter.json").items()}

# 8 kHz reference baselines (same-info comparison; from VERDICT.md / *_baselines.json)
BASE = {"watkins": {"log-mel 8k": 0.675, "AVES 8k": 0.853},
        "filt_f1": {"log-mel 8k": 0.903, "AVES 8k": 0.971},
        "filt_auc": {"log-mel 8k": 0.958, "AVES 8k": 0.994}}

# assemble per-run series
runs = {}
for tag, meta in REG.items():
    t = norm(tag); run = meta["run"]; step = meta["step"]
    wk = WK.get(t, {}).get("best_macro_f1"); cl = WK.get(t, {}).get("clustering", {})
    fb = FL.get(t, {}).get("best", {})
    runs.setdefault(run, []).append(dict(step=step, watkins=wk,
        knn=cl.get("knn_purity"), nmi=cl.get("nmi_kmeans"),
        filt_f1=fb.get("macro_f1"), filt_auc=fb.get("auc")))
for r in runs.values(): r.sort(key=lambda d: d["step"])
PAL = {}; colcyc = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
for i, run in enumerate(sorted(runs)): PAL[run] = colcyc[i % len(colcyc)]


def series(run, key):
    xs = [d["step"] for d in runs[run] if d.get(key) is not None]
    ys = [d[key] for d in runs[run] if d.get(key) is not None]
    return xs, ys


fig, ax = plt.subplots(1, 3, figsize=(16.5, 5))
# Panel A: Watkins species F1
for run in sorted(runs):
    x, y = series(run, "watkins")
    if x: ax[0].plot(x, y, "o-", color=PAL[run], lw=2, label=run)
for name, v in BASE["watkins"].items(): ax[0].axhline(v, ls="--", c="gray", lw=.9); ax[0].text(ax[0].get_xlim()[1], v, " " + name, fontsize=7, va="center", color="gray")
ax[0].axhline(0.032, ls=":", c="k", lw=.7); ax[0].set_title("Watkins species (31-way) macro-F1")
ax[0].set_xlabel("training step"); ax[0].set_ylabel("macro-F1"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=9)

# Panel B: Filtration F1 + AUC
for run in sorted(runs):
    x, y = series(run, "filt_f1");  x and ax[1].plot(x, y, "o-", color=PAL[run], lw=2, label=f"{run} F1")
    x, y = series(run, "filt_auc"); x and ax[1].plot(x, y, "s--", color=PAL[run], lw=1.6, alpha=.7, label=f"{run} AUC")
for name, v in BASE["filt_f1"].items(): ax[1].axhline(v, ls="--", c="gray", lw=.9); ax[1].text(ax[1].get_xlim()[1], v, " " + name, fontsize=7, va="center", color="gray")
ax[1].axhline(0.5, ls=":", c="k", lw=.7); ax[1].set_title("Filtration signal/noise (F1 ●, AUC ▢)")
ax[1].set_xlabel("training step"); ax[1].set_ylabel("score"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)

# Panel C: comparable view — % of the log-mel-8k baseline reached (Anvar's "expect comparable quality")
for run in sorted(runs):
    xs = [d["step"] for d in runs[run]]
    wp = [d["watkins"] / BASE["watkins"]["log-mel 8k"] * 100 if d.get("watkins") else None for d in runs[run]]
    fp = [d["filt_f1"] / BASE["filt_f1"]["log-mel 8k"] * 100 if d.get("filt_f1") else None for d in runs[run]]
    xw = [s for s, v in zip(xs, wp) if v is not None]; yw = [v for v in wp if v is not None]
    xf = [s for s, v in zip(xs, fp) if v is not None]; yf = [v for v in fp if v is not None]
    if xw: ax[2].plot(xw, yw, "o-", color=PAL[run], lw=2, label=f"{run} · Watkins")
    if xf: ax[2].plot(xf, yf, "^--", color=PAL[run], lw=2, alpha=.75, label=f"{run} · filtration")
ax[2].axhline(100, ls="--", c="gray", lw=.9); ax[2].text(ax[2].get_xlim()[1], 100, " = log-mel 8k", fontsize=7, va="center", color="gray")
ax[2].set_title("% of log-mel-8k baseline reached\n(both tasks on one comparable axis)")
ax[2].set_xlabel("training step"); ax[2].set_ylabel("% of log-mel-8k"); ax[2].grid(alpha=.3); ax[2].legend(fontsize=8)

fig.suptitle("animal2vec — validation dynamics across runs (decide which to keep)", fontsize=13, y=1.02)
plt.tight_layout(); plt.savefig(V + "a2v_dynamics.png", dpi=130, bbox_inches="tight")

# markdown table
lines = ["| run | step | Watkins-F1 | filt-F1 | filt-AUC | knn-pur | NMI | %log-mel(W) | %log-mel(F) |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for run in sorted(runs):
    for d in runs[run]:
        wp = f"{d['watkins']/0.675*100:.0f}%" if d.get("watkins") else "—"
        fp = f"{d['filt_f1']/0.903*100:.0f}%" if d.get("filt_f1") else "—"
        g = lambda k, f="{:.3f}": (f.format(d[k]) if d.get(k) is not None else "—")
        lines.append(f"| {run} | {d['step']} | {g('watkins')} | {g('filt_f1')} | {g('filt_auc')} | {g('knn')} | {g('nmi')} | {wp} | {fp} |")
print("\n".join(lines)); print("\nsaved a2v_dynamics.png")
open(V + "_dynamics_table.md", "w").write("\n".join(lines) + "\n")
