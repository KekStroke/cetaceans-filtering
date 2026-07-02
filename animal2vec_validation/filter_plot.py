#!/usr/bin/env python3
"""Plot the signal/noise filtration calibration (no GPU). Reads a2v_filter.json + filter_baselines.json."""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
V = os.path.dirname(os.path.abspath(__file__)) + "/"
A = json.load(open(V + "a2v_filter.json")); B = json.load(open(V + "filter_baselines.json"))
rows = [("animal2vec 13k", A["ckpt13k"]["best"], "#7bb6e0"),
        ("animal2vec 25k", A["ckpt25k"]["best"], "#1f77b4"),
        ("log-mel 8k", B["log-mel-8k"], "#999999"),
        ("AVES 8k", B["AVES-8k"], "#2ca02c")]
names = [r[0] for r in rows]; f1 = [r[1]["macro_f1"] for r in rows]; auc = [r[1]["auc"] for r in rows]; cols = [r[2] for r in rows]
x = np.arange(len(rows)); w = 0.38
fig, ax = plt.subplots(figsize=(8.2, 5))
b1 = ax.bar(x - w/2, f1, w, color=cols, label="macro-F1")
b2 = ax.bar(x + w/2, auc, w, color=cols, alpha=0.5, hatch="//", label="ROC-AUC")
ax.axhline(0.5, ls="--", c="k", lw=.8); ax.text(len(rows)-1, 0.515, "chance F1", fontsize=8, ha="right")
for b in list(b1)+list(b2): ax.text(b.get_x()+b.get_width()/2, b.get_height()+.008, f"{b.get_height():.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9); ax.set_ylim(0.45, 1.02); ax.set_ylabel("score")
ax.set_title("Signal vs noise FILTRATION (same clips, recording-disjoint)\nanimal2vec is learning to filter (13k→25k ↑) but still below a log-mel filter")
ax.legend(loc="lower right", fontsize=9); ax.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(V + "a2v_filter.png", dpi=130)
print("saved a2v_filter.png")
