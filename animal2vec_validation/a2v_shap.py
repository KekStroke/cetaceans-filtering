#!/usr/bin/env python3
"""SHAP-style frequency-band attribution for Anvar's animal2vec encoder.
Occlusion: bandstop-remove each 0.5kHz band from the waveform, re-embed, measure the drop in the
frozen probe's predicted prob for the true K-call class -> per-class band importance.
Shows WHICH bands the (undertrained) encoder relies on, vs where the call energy sits.
The sample rate (and therefore the band grid, spanning 0..Nyquist in 0.5kHz steps) is auto-detected
from the checkpoint's own cfg -- 8kHz checkpoints get 8 bands (0..4kHz), 16kHz get 16 (0..8kHz) --
so the band axis is never mislabeled when running against a 16kHz model.
Memory-safe: 1s clips, one model, GPU. Run: uv run python animal2vec_validation/a2v_shap.py <slim_ckpt>"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a2v_extract as E
import numpy as np, torch
from scipy.signal import butter, sosfiltfilt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
SR = 8000; NB = 8; BANDS = [(i * 500, (i + 1) * 500) for i in range(NB)]   # 8kHz default; re-derived post-load
ckpt = sys.argv[1] if len(sys.argv) > 1 else "/home/yarix/a2v_ckpts/ckpt25k_slim.pt"

model = E.load_model(ckpt)   # runs _detect_set_sr(): E.SR now reflects the checkpoint's true rate
# Re-derive the band grid from the detected rate so 16kHz checkpoints get 0..8kHz (16 bands), not a
# mislabeled 0..4kHz half. E.load_audio() already feeds audio at E.SR, so bandstop/band_energy below
# must use the same rate for their frequencies to line up with the axis labels.
SR = E.SR
NB = int(round((SR / 2) / 500))                                  # 0.5kHz bands spanning 0..Nyquist
BANDS = [(i * 500, (i + 1) * 500) for i in range(NB)]
log(f"band grid: SR={SR} Hz -> {NB} x 0.5kHz bands (0..{SR/2/1000:g} kHz)")
items = E.gather_kclass(os.environ.get("A2V_LAB_DIR", "/mnt/c/Users/Iaroslav/CETACEANS/new_training_data"), 100)
files = [f for f, _, _ in items]
le = LabelEncoder(); y = le.fit_transform([c for _, c, _ in items]); groups = np.array([g for _, _, g in items])
classes = list(le.classes_)
log(f"{len(files)} clips, {len(classes)} classes")


@torch.inference_mode()
def embed_layers(wav):
    x = torch.tensor(wav).view(1, -1).to(E.DEV)
    out = model(source=x, features_only=True)
    feats = []
    for lr in (out.get("layer_results") or []):
        t = lr[0] if isinstance(lr, (tuple, list)) else lr
        feats.append(t.reshape(-1, t.shape[-1]).float().mean(0).cpu().numpy())
    return feats


# --- extract all layers, pick the best probe layer ---
XL = None
for i, f in enumerate(files):
    fs = embed_layers(E.load_audio(f, 0))
    if XL is None: XL = [[] for _ in fs]
    for li, v in enumerate(fs): XL[li].append(v)
    if i % 300 == 0: log(f"  extract {i}/{len(files)}")
XL = [np.stack(p) for p in XL]
def probe_f1(X):
    gkf = GroupKFold(5); T, P = [], []
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr]); clf = LogisticRegression(max_iter=2000, C=1, class_weight='balanced', n_jobs=-1).fit(sc.transform(X[tr]), y[tr])
        P.append(clf.predict(sc.transform(X[te]))); T.append(y[te])
    return f1_score(np.concatenate(T), np.concatenate(P), average='macro')
f1s = [probe_f1(X) for X in XL]; BL = int(np.argmax(f1s))
log(f"best layer L{BL} (F1={f1s[BL]:.3f})")

# --- train probe on best layer (all data) ---
Xb = XL[BL]; sc = StandardScaler().fit(Xb)
clf = LogisticRegression(max_iter=2000, C=1, class_weight='balanced', n_jobs=-1).fit(sc.transform(Xb), y)


def bandstop(wav, lo, hi):
    lo = max(lo, 10) / (SR / 2); hi = min(hi, SR / 2 - 10) / (SR / 2)
    sos = butter(4, [lo, hi], btype='bandstop', output='sos')
    return sosfiltfilt(sos, wav).astype(np.float32)


def band_energy(wav):
    f = np.fft.rfftfreq(len(wav), 1 / SR); P = np.abs(np.fft.rfft(wav)) ** 2
    return np.array([P[(f >= lo) & (f < hi)].sum() for lo, hi in BANDS])


# --- occlusion attribution on a subset (10/class) ---
import collections
byc = collections.defaultdict(list)
for f, c, _ in items: byc[c].append(f)
imp = np.zeros((len(classes), NB)); eng = np.zeros((len(classes), NB)); cnt = np.zeros(len(classes))
for ci, c in enumerate(classes):
    for f in byc[c][:10]:
        w0 = E.load_audio(f, 0)
        p0 = clf.predict_proba(sc.transform(embed_layers(w0)[BL].reshape(1, -1)))[0, ci]
        e = band_energy(w0); eng[ci] += e / (e.sum() + 1e-9)
        for bi, (lo, hi) in enumerate(BANDS):
            pm = clf.predict_proba(sc.transform(embed_layers(bandstop(w0, lo, hi))[BL].reshape(1, -1)))[0, ci]
            imp[ci, bi] += (p0 - pm)
        cnt[ci] += 1
    log(f"  occlusion {c} done")
imp /= cnt[:, None]; eng /= cnt[:, None]

# correlate attribution vs energy (does it use where the energy is, or learn elsewhere?)
r = float(np.corrcoef(imp.flatten(), eng.flatten())[0, 1])
out = {"ckpt": ckpt.split("/")[-1], "best_layer": int(BL), "best_f1": float(f1s[BL]), "bands_kHz": [f"{lo/1000:.1f}-{hi/1000:.1f}" for lo, hi in BANDS],
       "band_importance": {classes[i]: imp[i].tolist() for i in range(len(classes))},
       "mean_importance": imp.mean(0).tolist(), "mean_energy": eng.mean(0).tolist(),
       "attr_vs_energy_pearson": r}
json.dump(out, open(os.path.join(HERE, "a2v_shap.json"), "w"), indent=2)

bl = [f"{lo/1000:g}-{hi/1000:g}" for lo, hi in BANDS]   # kHz, exact (no rounding)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
im = ax[0].imshow(imp, aspect='auto', cmap='RdBu_r', vmin=-abs(imp).max(), vmax=abs(imp).max())
ax[0].set_yticks(range(len(classes))); ax[0].set_yticklabels(classes, fontsize=7); ax[0].set_xticks(range(NB)); ax[0].set_xticklabels(bl, rotation=45, fontsize=7)
ax[0].set_title(f"animal2vec band importance (occlusion, L{BL})"); plt.colorbar(im, ax=ax[0])
ax[1].plot(range(NB), imp.mean(0), 'o-', label='attribution (mean)')
ax[1].plot(range(NB), eng.mean(0), 's--', c='gray', label='call energy (mean)')
ax[1].set_xticks(range(NB)); ax[1].set_xticklabels(bl, rotation=45, fontsize=7); ax[1].legend(); ax[1].grid(alpha=.3)
ax[1].set_title(f"attribution vs energy (r={r:.2f})  — does it use the call bands?")
plt.tight_layout(); plt.savefig(os.path.join(HERE, "a2v_shap.png"), dpi=120)
print(f"\n=== animal2vec SHAP (band occlusion, L{BL}) ===")
print("mean band importance:", " ".join(f"{bl[i]}={imp.mean(0)[i]:+.3f}" for i in range(NB)))
print(f"attribution-vs-energy Pearson r = {r:.3f}  (high=uses where energy is; low/neg=elsewhere)")
print("saved a2v_shap.{json,png}")
