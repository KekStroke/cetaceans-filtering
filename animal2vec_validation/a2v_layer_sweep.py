#!/usr/bin/env python3
"""Per-layer frozen-probe sweep for an animal2vec checkpoint (the final layer is usually the WORST
for SSL probes). Loads model once, extracts ALL transformer-layer mean-pooled embeddings per clip,
probes each layer (GroupKFold-5 LogReg) -> macro-F1. Finds the best layer = the fair number.
Run: uv run python animal2vec_validation/a2v_layer_sweep.py <ckpt> [n_per_class]"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a2v_extract as E   # sets torch2 compat shims, registers animal2vec.nn
import torch, numpy as np, json
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
HERE = os.path.dirname(os.path.abspath(__file__))
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

if len(sys.argv) < 2: sys.exit("usage: a2v_layer_sweep.py <checkpoint.pt> [n_per_class]  (set A2V_LAB_DIR)")
ckpt = sys.argv[1]
npc = int(sys.argv[2]) if len(sys.argv) > 2 else 100
model = E.load_model(ckpt)
items = E.gather_kclass(os.environ.get("A2V_LAB_DIR", "/mnt/c/Users/Iaroslav/CETACEANS/new_training_data"), npc)
files = [f for f, _, _ in items]
y_str = np.array([c for _, c, _ in items]); groups = np.array([g for _, _, g in items])
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder(); y = le.fit_transform(y_str); classes = list(le.classes_)
log(f"clips={len(files)} classes={len(classes)}")


@torch.inference_mode()
def extract_all(files):
    per = None
    for i, f in enumerate(files):
        x = torch.tensor(E.load_audio(f, 0)).view(1, -1).to(E.DEV)
        out = model(source=x, features_only=True)
        lrs = out.get("layer_results") or []
        feats = []
        for lr in lrs:
            t = lr[0] if isinstance(lr, (tuple, list)) else lr
            feats.append(t.reshape(-1, t.shape[-1]).float().mean(0).cpu().numpy())
        feats.append(out["x"][0].reshape(-1, out["x"].shape[-1]).float().mean(0).cpu().numpy())  # final
        if per is None: per = [[] for _ in feats]; log(f"  n_layers (incl final) = {len(feats)}, dim={feats[0].shape}")
        for li, v in enumerate(feats): per[li].append(v)
        if i % 200 == 0: log(f"  {i}/{len(files)}")
    return [np.stack(p).astype(np.float32) for p in per]


XL = extract_all(files)
log(f"extracted {len(XL)} layer-embeddings")


def probe(X):
    gkf = GroupKFold(min(5, len(set(groups)))); T, P = [], []
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", n_jobs=-1).fit(sc.transform(X[tr]), y[tr])
        P.append(clf.predict(sc.transform(X[te]))); T.append(y[te])
    T, P = np.concatenate(T), np.concatenate(P)
    return float(f1_score(T, P, average="macro"))


res = {}
for li, X in enumerate(XL):
    name = "final" if li == len(XL) - 1 else f"L{li}"
    res[name] = probe(X)
    log(f"  {name:6s}: macro-F1 = {res[name]:.4f}")
best = max(res, key=res.get)
json.dump({"per_layer": res, "best_layer": best, "best_f1": res[best], "n_per_class": npc,
           "ckpt": ckpt, "bars": {"log-mel": 0.654, "AVES": 0.894}},
          open(os.path.join(HERE, "animal2vec_layer_sweep.json"), "w"), indent=2)
print("\n=== animal2vec layer sweep (Olga K-class, recording-disjoint) ===")
for k, v in res.items(): print(f"  {k:6s} {v:.4f}")
print(f"\nBEST: {best} = {res[best]:.4f}   vs final {res['final']:.4f}   (bars: log-mel 0.654, AVES 0.874-0.894)")
