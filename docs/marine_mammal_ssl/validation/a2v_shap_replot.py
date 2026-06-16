#!/usr/bin/env python3
"""Re-render the animal2vec SHAP figure from a2v_shap.json with CORRECT band labels
(the original used .0f rounding -> '1.0-2k' actually meant 1.0-1.5kHz). No GPU needed."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = json.load(open("/mnt/c/Users/Iaroslav/CETACEANS/a2v_validation/a2v_shap.json"))
classes = list(D["band_importance"].keys())
imp = np.array([D["band_importance"][c] for c in classes])
mean_imp = np.array(D["mean_importance"]); mean_eng = np.array(D["mean_energy"])
BL = D["best_layer"]; r = D["attr_vs_energy_pearson"]; NB = imp.shape[1]
# correct, unambiguous labels (each band is 0.5 kHz wide)
edges = [round(i * 0.5, 1) for i in range(NB + 1)]
lab = [f"{edges[i]:g}–{edges[i+1]:g}" for i in range(NB)]  # 0-0.5, 0.5-1, ... 3.5-4 (kHz)

fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
m = abs(imp).max()
im = ax[0].imshow(imp, aspect='auto', cmap='RdBu_r', vmin=-m, vmax=m)
ax[0].set_yticks(range(len(classes))); ax[0].set_yticklabels(classes, fontsize=8)
ax[0].set_xticks(range(NB)); ax[0].set_xticklabels(lab, rotation=45, fontsize=8, ha='right')
ax[0].set_xlabel("frequency band (kHz)  —  8 kHz model, Nyquist 4 kHz")
ax[0].set_title(f"animal2vec band importance per K-class (occlusion, layer L{BL})")
cb = plt.colorbar(im, ax=ax[0]); cb.set_label("Δ true-class prob when band removed  (red = important)")

ax[1].plot(range(NB), mean_imp, 'o-', lw=2, label='attribution (mean over classes)')
ax[1].plot(range(NB), mean_eng, 's--', c='gray', label='call energy (mean)')
ax[1].axhline(0, c='k', lw=.6)
ax[1].set_xticks(range(NB)); ax[1].set_xticklabels(lab, rotation=45, fontsize=8, ha='right')
ax[1].set_xlabel("frequency band (kHz)"); ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
ax[1].set_title(f"attribution vs call energy (Pearson r = {r:.2f})")
ax[1].annotate("uses only the lowest band\n(0–0.5 kHz), ignores the\n1–4 kHz call bands → undertrained",
               xy=(0, mean_imp[0]), xytext=(2.4, 0.22), fontsize=9,
               arrowprops=dict(arrowstyle='->', color='C0'), color='C0')
fig.suptitle("animal2vec encoder (ckpt 25k, 8 kHz) — which frequencies it relies on", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("/mnt/c/Users/Iaroslav/CETACEANS/a2v_validation/a2v_shap_for_anvar.png", dpi=130, bbox_inches='tight')
print("saved a2v_shap_for_anvar.png with corrected labels:", lab)
