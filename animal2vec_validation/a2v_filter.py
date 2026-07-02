#!/usr/bin/env python3
"""Binary SIGNAL vs NOISE filtration probe for animal2vec (Anvar's ask "проверь фильтрацию").
Can the frozen encoder separate call-present clips from ambient noise? This is the encoder's
fitness as a *filter* (the cetaceans-filtering use case), independent of fine call-type ID.
Signal = K-call clips (all K-classes), Noise = the noise class; recording-disjoint GroupKFold.
Also dumps the exact clip manifest so filter_baselines.py scores AVES-8k / log-mel-8k on the SAME split.
Memory-safe: 1s clips, one model, GPU, watchdog. Run: uv run python animal2vec_validation/a2v_filter.py <slim_ckpt>"""
import os, sys, time, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a2v_extract as E
import numpy as np, torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__))
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
DATA = os.environ.get("A2V_LAB_DIR", "/mnt/c/Users/Iaroslav/CETACEANS/new_training_data")
ckpt = sys.argv[1] if len(sys.argv) > 1 else "/home/yarix/a2v_ckpts/ckpt25k_slim.pt"
tag = ckpt.split("/")[-1].replace("_slim.pt", "").replace(".pt", "")
N_PER = 500   # ~500 signal + ~500 noise, balanced

# --- balanced, recording-disjoint signal/noise sample ---
items = E.gather_kclass(DATA, 700)            # up to 700/class incl. noise
sig = [(f, g) for f, c, g in items if c != "noise"]
noi = [(f, g) for f, c, g in items if c == "noise"]
rng = np.random.RandomState(0); rng.shuffle(sig); rng.shuffle(noi)
sig = sig[:N_PER]; noi = noi[:N_PER]
files = [f for f, _ in sig] + [f for f, _ in noi]
y = np.array([1] * len(sig) + [0] * len(noi))           # 1=signal, 0=noise
groups = np.array([g for _, g in sig] + ["noise_" + g for _, g in noi])
log(f"{tag}: {len(sig)} signal + {len(noi)} noise, {len(set(groups))} groups")
json.dump({"files": files, "y": y.tolist(), "groups": groups.tolist()},
          open(os.path.join(HERE, "filter_manifest.json"), "w"))

model = E.load_model(ckpt)


@torch.inference_mode()
def embed_layers(f):
    x = torch.tensor(E.load_audio(f, 0)).view(1, -1).to(E.DEV)
    out = model(source=x, features_only=True)
    feats = [(lr[0] if isinstance(lr, (tuple, list)) else lr).reshape(-1, (lr[0] if isinstance(lr, (tuple, list)) else lr).shape[-1]).float().mean(0).cpu().numpy()
             for lr in (out.get("layer_results") or [])]
    return feats


XL = None
for i, f in enumerate(files):
    fs = embed_layers(f)
    if XL is None: XL = [[] for _ in fs]
    for li, v in enumerate(fs): XL[li].append(v)
    if i % 200 == 0: log(f"  embed {i}/{len(files)}")
XL = [np.stack(p) for p in XL]


def probe(X):
    gkf = GroupKFold(5); T, P, S = [], [], []
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1, class_weight='balanced').fit(sc.transform(X[tr]), y[tr])
        P.append(clf.predict(sc.transform(X[te]))); S.append(clf.predict_proba(sc.transform(X[te]))[:, 1]); T.append(y[te])
    T, P, S = np.concatenate(T), np.concatenate(P), np.concatenate(S)
    return dict(macro_f1=float(f1_score(T, P, average='macro')), bal_acc=float(balanced_accuracy_score(T, P)), auc=float(roc_auc_score(T, S)))


per = {f"L{li}": probe(X) for li, X in enumerate(XL)}
best = max(per, key=lambda k: per[k]['macro_f1'])
out = {"ckpt": tag, "n_signal": len(sig), "n_noise": len(noi), "best_layer": best, "best": per[best], "per_layer": per}
try:
    allr = json.load(open(os.path.join(HERE, "a2v_filter.json")))
except Exception:
    allr = {}
allr[tag] = out
json.dump(allr, open(os.path.join(HERE, "a2v_filter.json"), "w"), indent=2)
b = per[best]
print(f"\n=== animal2vec FILTRATION (signal vs noise) — {tag} ===")
print(f"best {best}: macro-F1={b['macro_f1']:.3f}  bal-acc={b['bal_acc']:.3f}  AUC={b['auc']:.3f}  (chance F1=0.5)")
