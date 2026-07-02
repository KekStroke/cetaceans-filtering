#!/usr/bin/env python3
"""Calibration: AVES (strong 16kHz animal SSL) + log-mel on the SAME Watkins task,
so we can judge whether animal2vec's score is good/low. Memory-safe: 10s cap, small models.
Run: uv run python animal2vec_validation/watkins_baselines.py"""
import os, sys, time, json
import numpy as np, librosa, torch, torchaudio
from datasets import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
HERE = os.path.dirname(os.path.abspath(__file__))
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BASE = os.environ.get(
    "A2V_WATKINS_ARROW_DIR",
    "/home/yarix/.cache/huggingface/datasets/DBD-research-group___beans_watkins/default/0.0.0/a93d9caeb8422992",
)
TR = Dataset.from_file(f"{BASE}/beans_watkins-train.arrow"); TE = Dataset.from_file(f"{BASE}/beans_watkins-test.arrow")
WAV = os.environ.get("A2V_AVES_WEIGHTS", "/mnt/c/Users/Iaroslav/CETACEANS/voxaboxen_weights/aves-base-bio.torchaudio")


def prep(arr, sr, target_sr, maxsec=10):
    y = np.asarray(arr, dtype=np.float32)
    nmax = maxsec * sr
    if len(y) > nmax: s = (len(y) - nmax) // 2; y = y[s:s + nmax]
    if sr != target_sr: y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    if len(y) > maxsec * target_sr: y = y[:maxsec * target_sr]
    y = y - y.mean(); st = y.std(); return (y / (st + 1e-8) if st > 1e-8 else y).astype(np.float32)


def probe_traintest(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced', n_jobs=-1).fit(sc.transform(Xtr), ytr)
    p = clf.predict(sc.transform(Xte))
    return float(f1_score(yte, p, average='macro')), float(accuracy_score(yte, p))


ys_tr = [ex["label"] for ex in TR]; ys_te = [ex["label"] for ex in TE]
le = LabelEncoder().fit(ys_tr + ys_te); ytr = le.transform(ys_tr); yte = le.transform(ys_te)
res = {}

# ---- AVES (16kHz, per-layer best) ----
log("loading AVES ...")
cfg = json.load(open(WAV + ".model_config.json"))
av = torchaudio.models.wav2vec2_model(**cfg, aux_num_out=None)
av.load_state_dict(torch.load(WAV + ".pt", map_location='cpu'), strict=False); av.eval().to(DEV)


@torch.inference_mode()
def aves_all_layers(ds):
    per = None
    for i, ex in enumerate(ds):
        a = ex["path"]; x = torch.tensor(prep(a["array"], a["sampling_rate"], 16000)).view(1, -1).to(DEV)
        feats, _ = av.extract_features(x)
        vs = [f[0].float().mean(0).cpu().numpy() for f in feats]
        if per is None: per = [[] for _ in vs]
        for li, v in enumerate(vs): per[li].append(v)
        if i % 300 == 0: log(f"  aves {i}/{len(ds)}")
    return [np.stack(p) for p in per]


AVtr = aves_all_layers(TR); AVte = aves_all_layers(TE)
av_per = {f"L{li}": probe_traintest(AVtr[li], ytr, AVte[li], yte) for li in range(len(AVtr))}
abest = max(av_per, key=lambda k: av_per[k][0])
res["AVES"] = {"best_layer": abest, "macro_f1": av_per[abest][0], "acc": av_per[abest][1]}
log(f"AVES: best {abest} F1={av_per[abest][0]:.3f} acc={av_per[abest][1]:.3f}")
del av; torch.cuda.empty_cache()

# ---- log-mel baseline ----
def logmel_feat(ds, sr=16000):
    X = []
    for ex in ds:
        a = ex["path"]; y = prep(a["array"], a["sampling_rate"], sr)
        m = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=400, hop_length=160, n_mels=64)
        m = librosa.power_to_db(m + 1e-10); X.append(np.concatenate([m.mean(1), m.std(1)]))
    return np.array(X, dtype=np.float32)


LMtr = logmel_feat(TR); LMte = logmel_feat(TE)
res["log-mel"] = dict(zip(["macro_f1", "acc"], probe_traintest(LMtr, ytr, LMte, yte)))
log(f"log-mel: F1={res['log-mel']['macro_f1']:.3f} acc={res['log-mel']['acc']:.3f}")

res["animal2vec_25k"] = {"macro_f1": 0.542, "acc": None, "note": "8kHz, 8% trained, rising"}
json.dump(res, open(os.path.join(HERE, "watkins_baselines.json"), "w"), indent=2)
print("\n=== WATKINS (31-way species) — calibration ===")
print(f"{'encoder':16s} {'macro-F1':>9s} {'acc':>7s}")
for k in ["AVES", "log-mel", "animal2vec_25k"]:
    r = res[k]; print(f"{k:16s} {r['macro_f1']:9.3f} {(r['acc'] if r.get('acc') else float('nan')):7.3f}")
print("chance ~0.032")
