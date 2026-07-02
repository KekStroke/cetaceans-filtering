#!/usr/bin/env python3
"""Watkins species validation for animal2vec checkpoints — the fairer 8kHz test Anvar asked for.
31-way marine-mammal species classification (uses Watkins' own train/test split) + clustering.
Runs per-layer probe (final layer is usually worst). Loops over given checkpoints for a trend.
Run: uv run python animal2vec_validation/a2v_watkins.py <ckpt1> [<ckpt2> ...]"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a2v_extract as E
import torch, numpy as np, librosa
from datasets import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, silhouette_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
HERE = os.path.dirname(os.path.abspath(__file__))
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

BASE = os.environ.get(
    "A2V_WATKINS_ARROW_DIR",
    "/home/yarix/.cache/huggingface/datasets/DBD-research-group___beans_watkins/default/0.0.0/a93d9caeb8422992",
)
TR = Dataset.from_file(f"{BASE}/beans_watkins-train.arrow")
TE = Dataset.from_file(f"{BASE}/beans_watkins-test.arrow")
log(f"Watkins train={len(TR)} test={len(TE)}")


def prep(arr, sr):
    # Target rate/length follow E.SR (auto-detected per-checkpoint by E.load_model() via
    # E._detect_set_sr() -- the sinc kernels are rate-specific, so a 16kHz-trained checkpoint
    # MUST get 16kHz input, not silently-downsampled 8kHz audio). 10s of NATIVE audio cap.
    maxlen = 10 * E.SR
    y = np.asarray(arr, dtype=np.float32)
    nmax = 10 * sr                                   # crop to ~10s of NATIVE audio first (bounds resample cost)
    if len(y) > nmax:
        s = (len(y) - nmax) // 2; y = y[s:s + nmax]
    if sr != E.SR: y = librosa.resample(y, orig_sr=sr, target_sr=E.SR)
    if len(y) > maxlen: y = y[:maxlen]               # hard cap
    y = y - y.mean(); s = y.std(); return (y / (s + 1e-8) if s > 1e-8 else y).astype(np.float32)


@torch.inference_mode()
def emb_all(model, ds):
    per = None; labs = []
    for i, ex in enumerate(ds):
        a = ex["path"]; x = torch.tensor(prep(a["array"], a["sampling_rate"])).view(1, -1).to(E.DEV)
        out = model(source=x, features_only=True)
        feats = []
        for lr in (out.get("layer_results") or []):
            t = lr[0] if isinstance(lr, (tuple, list)) else lr
            feats.append(t.reshape(-1, t.shape[-1]).float().mean(0).cpu().numpy())
        feats.append(out["x"][0].reshape(-1, out["x"].shape[-1]).float().mean(0).cpu().numpy())
        if per is None: per = [[] for _ in feats]
        for li, v in enumerate(feats): per[li].append(v)
        labs.append(ex["label"])
        if i % 300 == 0: log(f"  {i}/{len(ds)}")
    return [np.stack(p).astype(np.float32) for p in per], labs


def knn_purity(X, y, k=10):
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X))).fit(X); _, idx = nn.kneighbors(X)
    return float(np.mean([(y[idx[i, 1:]] == y[i]).mean() for i in range(len(X))]))


ALL = {}
for ckpt in sys.argv[1:]:
    tag = ckpt.split("/")[-1].replace(".pt", "")
    log(f"==== {tag} ====")
    model = E.load_model(ckpt)
    XLtr, ytr_s = emb_all(model, TR); XLte, yte_s = emb_all(model, TE)
    le = LabelEncoder().fit(ytr_s + yte_s); ytr = le.transform(ytr_s); yte = le.transform(yte_s)
    nL = len(XLtr); per = {}
    for li in range(nL):
        sc = StandardScaler().fit(XLtr[li])
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced', n_jobs=-1).fit(sc.transform(XLtr[li]), ytr)
        per["final" if li == nL - 1 else f"L{li}"] = float(f1_score(yte, clf.predict(sc.transform(XLte[li])), average='macro'))
    best = max(per, key=per.get); bi = nL - 1 if best == "final" else int(best[1:])
    Xall = np.vstack([XLtr[bi], XLte[bi]]); yall = np.concatenate([ytr, yte])
    Xs = StandardScaler().fit_transform(Xall)
    clustering = {"knn_purity": knn_purity(Xs, yall, 10),
                  "silhouette": float(silhouette_score(Xs, yall)),
                  "nmi_kmeans": float(normalized_mutual_info_score(yall, __import__("sklearn.cluster", fromlist=["KMeans"]).KMeans(len(set(yall)), n_init=10, random_state=0).fit_predict(Xs)))}
    ALL[tag] = {"per_layer": per, "best_layer": best, "best_macro_f1": per[best], "clustering": clustering, "n_classes": len(le.classes_)}
    log(f"  {tag}: BEST {best} macro-F1={per[best]:.4f} | knn-purity {clustering['knn_purity']:.3f} NMI {clustering['nmi_kmeans']:.3f}")
    del model; torch.cuda.empty_cache()

# ACCUMULATE across runs (for the dynamics harness) — merge into the existing JSON, don't overwrite
_jp = os.path.join(HERE, "animal2vec_watkins.json")
try:
    _prev = json.load(open(_jp))
except Exception:
    _prev = {}
_prev.update(ALL)
json.dump(_prev, open(_jp, "w"), indent=2)
print("\n=== animal2vec on WATKINS (31-way species, fair 8kHz test) ===")
print(f"{'ckpt':22s} {'best-layer F1':>14s} {'final F1':>10s} {'knn-pur':>8s} {'NMI':>6s}")
for tag, r in ALL.items():
    print(f"{tag:22s} {r['best_macro_f1']:14.4f} {r['per_layer']['final']:10.4f} {r['clustering']['knn_purity']:8.3f} {r['clustering']['nmi_kmeans']:6.3f}")
print("bars: chance ~0.03 (31-way) | a decent bioacoustic encoder should be >> chance")
