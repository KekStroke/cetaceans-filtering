#!/usr/bin/env python3
"""Calibration for the signal/noise filtration probe: AVES-8k + log-mel-8k on the EXACT same
clips/groups animal2vec was scored on (filter_manifest.json), so we know if its number is good.
8k = audio bandlimited to 8kHz (matches animal2vec's Nyquist), AVES (16kHz model) gets it upsampled.
Run AFTER a2v_filter.py (separate process => one model on GPU). ~/a2v_env/bin/python filter_baselines.py"""
import time, json
import numpy as np, librosa, soundfile as sf, torch, torchaudio
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
WAV = "/mnt/c/Users/Iaroslav/CETACEANS/voxaboxen_weights/aves-base-bio.torchaudio"
M = json.load(open("/mnt/c/Users/Iaroslav/CETACEANS/a2v_validation/filter_manifest.json"))
files, y, groups = M["files"], np.array(M["y"]), np.array(M["groups"])
log(f"{len(files)} clips ({y.sum()} signal / {(y==0).sum()} noise)")


def load8k(f, maxsec=10):
    a, s = sf.read(f, dtype="float32")
    if a.ndim > 1: a = a.mean(1)
    if s != 8000: a = librosa.resample(a, orig_sr=s, target_sr=8000)   # bandlimit to 8kHz
    a = a[:maxsec * 8000]
    if len(a) < 400: a = np.pad(a, (0, 400 - len(a)))
    a = a - a.mean(); st = a.std(); return (a / (st + 1e-8) if st > 1e-8 else a).astype(np.float32)


def probe(X):
    gkf = GroupKFold(5); T, P, S = [], [], []
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr]); clf = LogisticRegression(max_iter=2000, C=1, class_weight='balanced').fit(sc.transform(X[tr]), y[tr])
        P.append(clf.predict(sc.transform(X[te]))); S.append(clf.predict_proba(sc.transform(X[te]))[:, 1]); T.append(y[te])
    T, P, S = np.concatenate(T), np.concatenate(P), np.concatenate(S)
    return dict(macro_f1=float(f1_score(T, P, average='macro')), bal_acc=float(balanced_accuracy_score(T, P)), auc=float(roc_auc_score(T, S)))


res = {}
# ---- AVES-8k (16kHz model, fed 8k-bandlimited audio upsampled to 16k), per-layer best ----
cfg = json.load(open(WAV + ".model_config.json"))
av = torchaudio.models.wav2vec2_model(**cfg, aux_num_out=None)
av.load_state_dict(torch.load(WAV + ".pt", map_location='cpu'), strict=False); av.eval().to(DEV)


@torch.inference_mode()
def aves_layers(f):
    a = load8k(f); a = librosa.resample(a, orig_sr=8000, target_sr=16000)
    x = torch.tensor(a).view(1, -1).to(DEV)
    feats, _ = av.extract_features(x)
    return [fl[0].float().mean(0).cpu().numpy() for fl in feats]


AV = None
for i, f in enumerate(files):
    fs = aves_layers(f)
    if AV is None: AV = [[] for _ in fs]
    for li, v in enumerate(fs): AV[li].append(v)
    if i % 200 == 0: log(f"  aves {i}/{len(files)}")
AV = [np.stack(p) for p in AV]
ap = {f"L{li}": probe(X) for li, X in enumerate(AV)}
ab = max(ap, key=lambda k: ap[k]['macro_f1'])
res["AVES-8k"] = {"best_layer": ab, **ap[ab]}
log(f"AVES-8k best {ab}: F1={ap[ab]['macro_f1']:.3f} AUC={ap[ab]['auc']:.3f}")
del av; torch.cuda.empty_cache()

# ---- log-mel-8k baseline ----
def logmel(f):
    a = load8k(f); m = librosa.feature.melspectrogram(y=a, sr=8000, n_fft=400, hop_length=160, n_mels=64)
    m = librosa.power_to_db(m + 1e-10); return np.concatenate([m.mean(1), m.std(1)])


LM = np.array([logmel(f) for f in files], dtype=np.float32)
res["log-mel-8k"] = probe(LM)
log(f"log-mel-8k: F1={res['log-mel-8k']['macro_f1']:.3f} AUC={res['log-mel-8k']['auc']:.3f}")

json.dump(res, open("/mnt/c/Users/Iaroslav/CETACEANS/a2v_validation/filter_baselines.json", "w"), indent=2)
print("\n=== FILTRATION calibration (signal vs noise, same clips) ===")
print(f"{'encoder':14s} {'macro-F1':>9s} {'bal-acc':>8s} {'AUC':>7s}")
for k in ["AVES-8k", "log-mel-8k"]:
    r = res[k]; print(f"{k:14s} {r['macro_f1']:9.3f} {r['bal_acc']:8.3f} {r['auc']:7.3f}")
