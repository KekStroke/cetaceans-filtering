#!/usr/bin/env python3
"""
animal2vec checkpoint validation — single turnkey CLI (modern torch 2.x + GPU; lossless by weights).

Loads a data2vec_multi pretraining checkpoint and runs frozen-probe validation. One subcommand per task;
each loads ONE model in the process (memory-safe: 80000-sample input cap, slim checkpoint, no second model).
The input sample rate is auto-detected from the checkpoint (8 kHz vs 16 kHz), so the same commands work for
both; SHAP bands then span 0..Nyquist. Override with A2V_SR if a checkpoint omits the rate.

  python validate.py watkins  <ckpt> [--run R --step S] [--no-baselines]   # 31-way species + clustering
  python validate.py filter   <ckpt> [--run R --step S] [--no-baselines]   # binary signal/noise
  python validate.py shap     <ckpt>                                       # frequency-band attribution + PNG
  python validate.py kclass   <ckpt>                                       # per-layer probe on Olga K-class
  python validate.py dynamics                                              # plot accumulated runs/steps
  python validate.py watch    <save_dir> --run R [--device 0]              # live: preserve+validate new ckpts, track BEST

Env (override the defaults): A2V_REPO (~/a2v), A2V_OUT (./a2v_val_results), A2V_KCLASS, A2V_WATKINS,
A2V_AVES, A2V_SR (force input rate; default auto-detect), A2V_MAXLEN (input cap in samples, default 80000),
A2V_TENSORBOARD_LOGDIR (validation scalar logdir; default A2V_OUT/tensorboard/<run> when --run is set).
Run with the legacy-fairseq GPU env, from the a2v repo dir, e.g.:
  cd ~/a2v && ~/a2v_env/bin/python /path/validate.py watkins ~/a2v_ckpts/ckpt25k_slim.pt
"""
import os, sys, glob, json, time, types, re, argparse, collections, tempfile, warnings
import numpy as np

# The lbfgs probe rarely hits tol within max_iter on high-dim (1024) embeddings, so sklearn spams a
# ConvergenceWarning per fit. The macro-F1 *ranking* is stable regardless (every checkpoint is probed the
# same way), so silence it rather than bump max_iter — changing max_iter mid-curve would break the
# step-over-step comparability of an accumulating dynamics run.
warnings.filterwarnings("ignore", message="lbfgs failed to converge")

SR = int(os.environ.get("A2V_SR", 8000))   # model input sample rate; auto-detected from the ckpt in load_model
_SR_FORCED = "A2V_SR" in os.environ        # if the user pinned A2V_SR, don't override it from the checkpoint
MAXLEN = int(os.environ.get("A2V_MAXLEN", 80000))  # hard input cap in SAMPLES (=10s@8k =5s@16k): matches the
#                                                    train window and caps O(T^2) attention memory at any SR
# determinism: cuBLAS workspace must be set BEFORE CUDA initialises (this runs at import, before T())
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
A2V_REPO  = os.environ.get("A2V_REPO",  os.path.expanduser("~/a2v"))         # animal2vec repo (for `import nn`)
OUT       = os.environ.get("A2V_OUT",   "a2v_val_results")                  # where JSON/PNG outputs go
KCLASS    = os.environ.get("A2V_KCLASS","data/kclass_wavs")                 # dir of labelled K-class .wav clips
WATKINS   = os.environ.get("A2V_WATKINS","data/beans_watkins")              # BEANS Watkins arrow dir (train/test)
AVES      = os.environ.get("A2V_AVES",  "weights/aves-base-bio.torchaudio") # AVES torchaudio weights prefix (.pt/.json)
def outp(name): os.makedirs(OUT, exist_ok=True); return os.path.join(OUT, name)
def load_json(name, default):
    p = os.path.join(OUT, name)
    try: return json.load(open(p))
    except Exception: return default
def save_json(name, obj): json.dump(obj, open(outp(name), "w"), indent=2)

def _numeric_scalars(prefix, obj):
    if isinstance(obj, (int, float, np.integer, np.floating)) and np.isfinite(obj):
        yield prefix, float(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(k)).strip("_")
            yield from _numeric_scalars(f"{prefix}/{key}" if prefix else key, v)

def _write_tensorboard(task, a, res, headline):
    run = getattr(a, "run", None)
    step = getattr(a, "step", None)
    if not run or step is None:
        return
    root = os.environ.get("A2V_TENSORBOARD_LOGDIR") or os.path.join(OUT, "tensorboard")
    logdir = os.path.join(root, str(run))
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(logdir)
        try:
            if isinstance(headline, (int, float, np.integer, np.floating)) and np.isfinite(headline):
                writer.add_scalar(f"validation/{task}/score", float(headline), int(step))
            for name, value in _numeric_scalars(f"validation/{task}", res):
                writer.add_scalar(name, value, int(step))
            writer.flush()
        finally:
            writer.close()
    except Exception as e:
        log(f"tensorboard: skipped scalar write to {logdir}: {e}")

def _detect_set_sr(cfg):
    """Set the global SR from the checkpoint's sample_rate so 16 kHz models get 16 kHz input (the sinc
    kernels are rate-specific — feeding 8 kHz audio to a 16 kHz model misreads every frequency by 2x).
    cfg = ck['cfg']. Env A2V_SR (if set) wins. Returns the SR in use."""
    global SR
    if _SR_FORCED:
        return SR
    rate = None
    for path in (("task", "sample_rate"), ("model", "sample_rate"),
                 ("model", "modalities", "audio", "sample_rate")):
        c = cfg
        try:
            for k in path:
                c = c.get(k) if hasattr(c, "get") else c[k]
            if c: rate = int(c); break
        except Exception:
            continue
    if rate and rate != SR:
        log(f"sample_rate {rate} Hz from checkpoint (was {SR}); using {rate} Hz")
        SR = rate
    return SR

# ======================= model loader (lazy: only when a checkpoint is needed) =======================
_T = None
def T():
    global _T
    if _T is None: import torch; _T = torch
    return _T
_DEV = None
def DEV():
    global _DEV
    if _DEV is None: _DEV = "cuda" if T().cuda.is_available() else "cpu"
    return _DEV

def _setup_fairseq():
    """torch._six shim + register data2vec_multi + swallow newer-kwarg fns (inference-safe). Idempotent."""
    sys.path.insert(0, A2V_REPO)
    import torch
    try: import torch._six  # noqa
    except Exception:
        m = types.ModuleType("torch._six"); m.string_classes=(str,bytes); m.int_classes=(int,)
        import collections.abc as abc; m.container_abcs=abc; sys.modules["torch._six"]=m
    import nn  # noqa: F401 (registers data2vec_multi + audio_ccas task)
    import inspect
    def swallow(mod, fn):
        if not hasattr(mod, fn): return
        orig = getattr(mod, fn)
        try: sig = set(inspect.signature(orig).parameters)
        except (ValueError, TypeError): return
        setattr(mod, fn, lambda *a, **k: orig(*a, **{x: y for x, y in k.items() if x in sig}))
    try:
        import animal2vec.nn.modalities.base as nb; swallow(nb, "compute_mask_indices")
    except Exception as e:
        log(f"mask-indices patch skipped: {e}")

def _del_key(node, fullkey):
    from omegaconf import open_dict
    parts = fullkey.split('.'); n = node
    for p in parts[:-1]:
        try: n = n[p]
        except Exception: return False
    k = parts[-1]
    if isinstance(n, dict):
        if k in n: del n[k]; return True
        return False
    try:
        with open_dict(n):
            if k in n: del n[k]; return True
    except Exception: pass
    return False

def sanitize_and_save(src, dst):
    """Strip Anvar-fork cfg keys not in the public dataclasses + EMA teacher; keep weights. Slim ~1.3GB."""
    from omegaconf import OmegaConf, open_dict
    log("loading checkpoint for sanitize ...")
    ck = T().load(src, map_location='cpu', weights_only=False)
    cfg = ck['cfg']
    _detect_set_sr(cfg)   # pick up the model's sample rate (8k vs 16k) before any audio is loaded
    from animal2vec.nn.audio_tasks import AudioConfigCCAS
    try:
        from animal2vec.nn.data2vec2 import Data2VecMultiConfig
        sections = [('task', AudioConfigCCAS), ('model', Data2VecMultiConfig)]
    except Exception:
        sections = [('task', AudioConfigCCAS)]
    for sec, dc in sections:
        if cfg.get(sec) is None: continue
        try: schema = OmegaConf.structured(dc)
        except Exception as e:
            log(f"  [sanitize] can't structure {sec} ({type(e).__name__}); skip"); continue
        for _ in range(80):
            try: OmegaConf.merge(schema, cfg[sec]); break
            except Exception as e:
                m = re.search(r"full_key:\s*(\S+)", str(e)) or re.search(r"Key '([^']+)' not in", str(e))
                if not m: raise
                fk = m.group(1)
                if not _del_key(cfg[sec], fk):
                    with open_dict(cfg[sec]):
                        kk = fk.split('.')[-1]
                        if kk in cfg[sec]: del cfg[sec][kk]
                        else: raise
                log(f"  [sanitize] {sec}: stripped '{fk}'")
    mc = cfg.get('model')
    if mc is not None:
        try: mc['skip_ema'] = True
        except Exception:
            with open_dict(mc): mc['skip_ema'] = True
    sd = ck.get('model', {})
    for k in list(sd.keys()):
        if k == '_ema' or k.startswith('_ema') or k.startswith('ema.'): del sd[k]
    for heavy in ['last_optimizer_state']:                 # drop only the big Adam moments
        if heavy in ck: del ck[heavy]
    log("saving slim checkpoint ...")
    T().save(ck, dst)
    return dst

def load_model(ckpt):
    _setup_fairseq()
    import torch
    # reproducible probe scores across runs (clean week-over-week dynamics). The real source of
    # nondeterminism is data2vec's random masking — disabled via mask=False in emb_layers; these
    # flags are cheap belt-and-suspenders for the conv frontend.
    torch.manual_seed(0); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    from fairseq import checkpoint_utils
    san = os.path.join(tempfile.gettempdir(), f"a2v_ckpt_sanitized_{os.getpid()}.pt")  # per-process: no clobber
    sanitize_and_save(ckpt, san)
    log("load_model_ensemble ...")
    models, _ = checkpoint_utils.load_model_ensemble([san])
    m = models[0].to(DEV()).eval()
    log(f"model on {DEV()}: {type(m).__name__}, {sum(p.numel() for p in m.parameters())/1e6:.0f}M params")
    return m

def norm_wav(y, sr):
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1: y = y.mean(1)
    nmax = int(MAXLEN * sr / SR)                       # native-sample budget = MAXLEN at the model SR (center-crop first)
    if len(y) > nmax: s = (len(y) - nmax) // 2; y = y[s:s + nmax]
    if sr != SR: import librosa; y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if len(y) > MAXLEN: y = y[:MAXLEN]                 # hard cap in samples (memory-safe at any SR)
    if len(y) < 400: y = np.pad(y, (0, 400 - len(y)))
    y = y - y.mean(); st = y.std(); return (y / (st + 1e-8) if st > 1e-8 else y).astype(np.float32)

def load_wav(path):
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32"); return norm_wav(y, sr)

def emb_layers(model, wav):
    torch = T()
    with torch.inference_mode():
        x = torch.tensor(wav).view(1, -1).to(DEV())
        out = model(source=x, features_only=True, mask=False)   # mask defaults True -> random masking -> stochastic feats; OFF for clean deterministic probe features
        feats = []
        for lr in (out.get("layer_results") or []):
            t = lr[0] if isinstance(lr, (tuple, list)) else lr
            feats.append(t.reshape(-1, t.shape[-1]).float().mean(0).cpu().numpy())
        feats.append(out["x"][0].reshape(-1, out["x"].shape[-1]).float().mean(0).cpu().numpy())  # final
        return feats

# ======================= data + probe helpers =======================
def class_of(path):
    parent = os.path.basename(os.path.dirname(path))
    if parent.startswith("K") or parent == "noise":
        return parent
    name = os.path.basename(path)
    return 'noise' if name.startswith('noise') else name.split('-')[0]
def tape_key(p):
    parts = os.path.basename(p)[:-4].split('_'); return parts[3] if len(parts) >= 4 else os.path.basename(p)
def gather_kclass(lab_dir, n_per_class, include_noise=True):
    files = sorted(glob.glob(os.path.join(lab_dir, "**", "*.wav"), recursive=True))
    if not files:
        raise SystemExit(f"no .wav files under A2V_KCLASS={lab_dir!r} — point A2V_KCLASS at your K-class clip dir")
    by = collections.defaultdict(list)
    for f in files:
        c = class_of(f)
        if c == "noise" or c.startswith("K"):
            by[c].append(f)
    CLASSES = sorted([c for c in by if c != "noise"], key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
    if include_noise and "noise" in by:
        CLASSES.append("noise")
    rng = np.random.RandomState(42); items = []
    for c in CLASSES:
        fs = by[c][:]; rng.shuffle(fs)
        for f in fs[:n_per_class]: items.append((f, c, tape_key(f)))
    return items

def _clf():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced', n_jobs=-1)
def probe_cv(X, y, groups, binary=False):
    """recording-disjoint GroupKFold probe. Returns macro-F1 (+ AUC if binary)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
    gkf = GroupKFold(min(5, len(set(groups)))); T_, P_, S_ = [], [], []
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr]); clf = _clf().fit(sc.transform(X[tr]), y[tr])
        P_.append(clf.predict(sc.transform(X[te]))); T_.append(y[te])
        if binary: S_.append(clf.predict_proba(sc.transform(X[te]))[:, 1])
    yt, yp = np.concatenate(T_), np.concatenate(P_)
    r = dict(macro_f1=float(f1_score(yt, yp, average='macro')), bal_acc=float(balanced_accuracy_score(yt, yp)))
    if binary: r['auc'] = float(roc_auc_score(yt, np.concatenate(S_)))
    return r
def probe_split(Xtr, ytr, Xte, yte):
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, accuracy_score
    sc = StandardScaler().fit(Xtr); clf = _clf().fit(sc.transform(Xtr), ytr); p = clf.predict(sc.transform(Xte))
    return dict(macro_f1=float(f1_score(yte, p, average='macro')), acc=float(accuracy_score(yte, p)))
def knn_purity(X, y, k=10):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X))).fit(X); _, idx = nn.kneighbors(X)
    return float(np.mean([(y[idx[i, 1:]] == y[i]).mean() for i in range(len(X))]))

# ======================= baselines (AVES-8k + log-mel-8k on the same clips) =======================
def aves_logmel_features(wavs):
    """wavs: list of 8kHz np arrays. Returns (aves_per_layer[list of arrays], logmel[array])."""
    import torch, torchaudio, librosa
    cfg = json.load(open(AVES + ".model_config.json"))
    av = torchaudio.models.wav2vec2_model(**cfg, aux_num_out=None)
    av.load_state_dict(torch.load(AVES + ".pt", map_location='cpu'), strict=False); av.eval().to(DEV())
    AV, LM = None, []
    with torch.inference_mode():
        for w in wavs:
            a16 = librosa.resample(w, orig_sr=SR, target_sr=16000)
            feats, _ = av.extract_features(torch.tensor(a16).view(1, -1).to(DEV()))
            vs = [f[0].float().mean(0).cpu().numpy() for f in feats]
            if AV is None: AV = [[] for _ in vs]
            for li, v in enumerate(vs): AV[li].append(v)
            m = librosa.feature.melspectrogram(y=w, sr=SR, n_fft=400, hop_length=160, n_mels=64)
            m = librosa.power_to_db(m + 1e-10); LM.append(np.concatenate([m.mean(1), m.std(1)]))
    del av; torch.cuda.empty_cache()
    return [np.stack(p) for p in AV], np.array(LM, dtype=np.float32)

# ======================= tasks =======================
def task_watkins(a):
    from datasets import Dataset
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import silhouette_score, normalized_mutual_info_score
    from sklearn.cluster import KMeans
    TR = Dataset.from_file(f"{WATKINS}/beans_watkins-train.arrow"); TE = Dataset.from_file(f"{WATKINS}/beans_watkins-test.arrow")
    if getattr(a, "limit", 0):
        TR = TR.select(range(min(a.limit, len(TR)))); TE = TE.select(range(min(a.limit, len(TE))))
    log(f"Watkins train={len(TR)} test={len(TE)}")
    model = load_model(a.ckpt)
    def emb_ds(ds):
        per = None; labs = []; wavs = []
        for i, ex in enumerate(ds):
            w = norm_wav(ex["path"]["array"], ex["path"]["sampling_rate"]); wavs.append(w)
            fs = emb_layers(model, w)
            if per is None: per = [[] for _ in fs]
            for li, v in enumerate(fs): per[li].append(v)
            labs.append(ex["label"])
            if i % 300 == 0: log(f"  emb {i}/{len(ds)}")
        return [np.stack(p) for p in per], labs, wavs
    XLtr, ytr_s, wtr = emb_ds(TR); XLte, yte_s, wte = emb_ds(TE)
    le = LabelEncoder().fit(ytr_s + yte_s); ytr = le.transform(ytr_s); yte = le.transform(yte_s)
    per = {}
    for li in range(len(XLtr)):
        per["final" if li == len(XLtr) - 1 else f"L{li}"] = probe_split(XLtr[li], ytr, XLte[li], yte)["macro_f1"]
    best = max(per, key=per.get); bi = len(XLtr) - 1 if best == "final" else int(best[1:])
    Xall = np.vstack([XLtr[bi], XLte[bi]]); yall = np.concatenate([ytr, yte])
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(Xall)
    clustering = dict(knn_purity=knn_purity(Xs, yall, 10), silhouette=float(silhouette_score(Xs, yall)),
                      nmi_kmeans=float(normalized_mutual_info_score(yall, KMeans(len(set(yall)), n_init=10, random_state=0).fit_predict(Xs))))
    res = dict(best_layer=best, best_macro_f1=per[best], per_layer=per, clustering=clustering, n_classes=len(le.classes_))
    del model; T().cuda.empty_cache()   # free animal2vec before AVES — one model on GPU at a time
    if a.baselines:
        AVtr, LMtr = aves_logmel_features(wtr); AVte, LMte = aves_logmel_features(wte)
        ap = {f"L{li}": probe_split(AVtr[li], ytr, AVte[li], yte)["macro_f1"] for li in range(len(AVtr))}
        res["baselines"] = {"AVES_8k": max(ap.values()), "logmel_8k": probe_split(LMtr, ytr, LMte, yte)["macro_f1"]}
    _record("watkins", a, res, res["best_macro_f1"])
    log(f"WATKINS best {best} macro-F1={per[best]:.4f} | knn {clustering['knn_purity']:.3f} NMI {clustering['nmi_kmeans']:.3f}"
        + (f" | AVES-8k {res['baselines']['AVES_8k']:.3f} logmel-8k {res['baselines']['logmel_8k']:.3f}" if a.baselines else ""))

def task_filter(a):
    items = gather_kclass(KCLASS, 700, include_noise=True)
    sig = [(f, g) for f, c, g in items if c != "noise"]; noi = [(f, g) for f, c, g in items if c == "noise"]
    cap = getattr(a, "limit", 0) or 500
    rng = np.random.RandomState(0); rng.shuffle(sig); rng.shuffle(noi); sig = sig[:cap]; noi = noi[:cap]
    files = [f for f, _ in sig] + [f for f, _ in noi]
    y = np.array([1] * len(sig) + [0] * len(noi))
    groups = np.array([g for _, g in sig] + ["noise_" + g for _, g in noi])
    log(f"filter: {len(sig)} signal + {len(noi)} noise, {len(set(groups))} groups")
    model = load_model(a.ckpt)
    XL = None; wavs = []
    for i, f in enumerate(files):
        w = load_wav(f); wavs.append(w); fs = emb_layers(model, w)
        if XL is None: XL = [[] for _ in fs]
        for li, v in enumerate(fs): XL[li].append(v)
        if i % 200 == 0: log(f"  emb {i}/{len(files)}")
    XL = [np.stack(p) for p in XL]
    per = {("final" if li == len(XL) - 1 else f"L{li}"): probe_cv(X, y, groups, binary=True) for li, X in enumerate(XL)}
    best = max(per, key=lambda k: per[k]['macro_f1'])
    res = dict(best_layer=best, best=per[best], per_layer=per, n_signal=len(sig), n_noise=len(noi))
    del model; T().cuda.empty_cache()   # free animal2vec before AVES — one model on GPU at a time
    if a.baselines:
        AV, LM = aves_logmel_features(wavs)
        ap = {f"L{li}": probe_cv(X, y, groups, binary=True)['macro_f1'] for li, X in enumerate(AV)}
        res["baselines"] = {"AVES_8k": max(ap.values()), "logmel_8k": probe_cv(LM, y, groups, binary=True)['macro_f1']}
    _record("filter", a, res, res["best"]["macro_f1"])
    b = per[best]
    log(f"FILTER best {best}: macro-F1={b['macro_f1']:.3f} AUC={b['auc']:.3f}"
        + (f" | AVES-8k {res['baselines']['AVES_8k']:.3f} logmel-8k {res['baselines']['logmel_8k']:.3f}" if a.baselines else ""))

def task_kclass(a):
    """per-layer frozen probe on Olga K-class call types — the layer sweep."""
    items = gather_kclass(KCLASS, a.n_per_class, include_noise=False)
    files = [f for f, _, _ in items]
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder(); y = le.fit_transform([c for _, c, _ in items]); groups = np.array([g for _, _, g in items])
    model = load_model(a.ckpt)
    XL = None
    for i, f in enumerate(files):
        fs = emb_layers(model, load_wav(f))
        if XL is None: XL = [[] for _ in fs]
        for li, v in enumerate(fs): XL[li].append(v)
        if i % 200 == 0: log(f"  emb {i}/{len(files)}")
    XL = [np.stack(p) for p in XL]
    per = {("final" if li == len(XL) - 1 else f"L{li}"): probe_cv(X, y, groups)["macro_f1"] for li, X in enumerate(XL)}
    best = max(per, key=per.get)
    res = {"per_layer": per, "best_layer": best, "best_macro_f1": per[best], "n_classes": len(le.classes_)}
    _record("kclass", a, res, per[best])
    log(f"KCLASS best {best} macro-F1={per[best]:.3f} ({len(le.classes_)}-way; chance ~{1/len(le.classes_):.2f})")

def task_shap(a):
    """occlusion frequency-band attribution on the best K-class layer."""
    from scipy.signal import butter, sosfiltfilt
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    nper = getattr(a, "limit", 0) or 100; nocc = min(10, getattr(a, "limit", 0) or 10)
    items = gather_kclass(KCLASS, nper)
    files = [f for f, _, _ in items]
    le = LabelEncoder(); y = le.fit_transform([c for _, c, _ in items]); groups = np.array([g for _, _, g in items]); classes = list(le.classes_)
    model = load_model(a.ckpt)   # sets SR (8k vs 16k) -> band grid spans up to the model's Nyquist
    NB = max(8, int((SR / 2) // 500)); BANDS = [(i * 500, (i + 1) * 500) for i in range(NB)]  # 0..SR/2 in 0.5 kHz steps
    XL = None
    for i, f in enumerate(files):
        fs = emb_layers(model, load_wav(f))
        if XL is None: XL = [[] for _ in fs]
        for li, v in enumerate(fs): XL[li].append(v)
    XL = [np.stack(p) for p in XL]
    f1s = [probe_cv(X, y, groups)["macro_f1"] for X in XL]; BL = int(np.argmax(f1s))
    log(f"shap: best layer L{BL} (F1={f1s[BL]:.3f})")
    sc = StandardScaler().fit(XL[BL]); clf = _clf().fit(sc.transform(XL[BL]), y)
    def bandstop(w, lo, hi):
        lo = max(lo, 10) / (SR / 2); hi = min(hi, SR / 2 - 10) / (SR / 2)
        return sosfiltfilt(butter(4, [lo, hi], btype='bandstop', output='sos'), w).astype(np.float32)
    def band_energy(w):
        fr = np.fft.rfftfreq(len(w), 1 / SR); P = np.abs(np.fft.rfft(w)) ** 2
        return np.array([P[(fr >= lo) & (fr < hi)].sum() for lo, hi in BANDS])
    byc = collections.defaultdict(list)
    for f, c, _ in items: byc[c].append(f)
    imp = np.zeros((len(classes), NB)); eng = np.zeros((len(classes), NB)); cnt = np.zeros(len(classes))
    for ci, c in enumerate(classes):
        for f in byc[c][:nocc]:
            w0 = load_wav(f); p0 = clf.predict_proba(sc.transform(emb_layers(model, w0)[BL].reshape(1, -1)))[0, ci]
            e = band_energy(w0); eng[ci] += e / (e.sum() + 1e-9)
            for bi, (lo, hi) in enumerate(BANDS):
                pm = clf.predict_proba(sc.transform(emb_layers(model, bandstop(w0, lo, hi))[BL].reshape(1, -1)))[0, ci]
                imp[ci, bi] += (p0 - pm)
            cnt[ci] += 1
        log(f"  occlusion {c} done")
    imp /= cnt[:, None]; eng /= cnt[:, None]
    r = float(np.corrcoef(imp.flatten(), eng.flatten())[0, 1])   # does it attend where the call energy is?
    save_json(f"shap_{_tag(a.ckpt)}.json", {"best_layer": BL, "bands_kHz": [f"{lo/1000:g}-{hi/1000:g}" for lo, hi in BANDS],
              "band_importance": {classes[i]: imp[i].tolist() for i in range(len(classes))},
              "mean_importance": imp.mean(0).tolist(), "mean_energy": eng.mean(0).tolist(), "attr_vs_energy_pearson": r})
    bl = [f"{lo/1000:g}-{hi/1000:g}" for lo, hi in BANDS]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5)); m = abs(imp).max()
    im = ax[0].imshow(imp, aspect='auto', cmap='RdBu_r', vmin=-m, vmax=m)
    ax[0].set_yticks(range(len(classes))); ax[0].set_yticklabels(classes, fontsize=8)
    ax[0].set_xticks(range(NB)); ax[0].set_xticklabels(bl, rotation=45, fontsize=8, ha='right'); ax[0].set_xlabel("kHz band")
    ax[0].set_title(f"animal2vec band importance (occlusion, L{BL})"); plt.colorbar(im, ax=ax[0])
    ax[1].plot(range(NB), imp.mean(0), 'o-', label='attribution (mean)')
    ax[1].plot(range(NB), eng.mean(0), 's--', c='gray', label='call energy (mean)')
    ax[1].axhline(0, c='k', lw=.6); ax[1].set_xticks(range(NB)); ax[1].set_xticklabels(bl, rotation=45, fontsize=8, ha='right')
    ax[1].legend(fontsize=9); ax[1].grid(alpha=.3); ax[1].set_title(f"attribution vs call energy (r={r:.2f})")
    plt.tight_layout(); plt.savefig(outp(f"shap_{_tag(a.ckpt)}.png"), dpi=120)
    log(f"SHAP saved {outp(f'shap_{_tag(a.ckpt)}.png')}; mean importance {[round(v,3) for v in imp.mean(0)]}; attr-vs-energy r={r:.3f}")

def task_dynamics(a):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    reg = load_json("dynamics_registry.json", {})
    WK = load_json("watkins_results.json", {}); FL = load_json("filter_results.json", {})
    if not reg: log("no dynamics_registry.json yet — run `watkins`/`filter <ckpt> --run R --step S` first"); return
    BASE = {"watkins": 0.675, "filt": 0.903}
    runs = {}
    for tag, meta in reg.items():
        runs.setdefault(meta["run"], []).append(dict(step=meta["step"],
            wk=WK.get(tag, {}).get("best_macro_f1"), fl=FL.get(tag, {}).get("best", {}).get("macro_f1")))
    for r in runs.values(): r.sort(key=lambda d: d["step"])
    pal = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for i, run in enumerate(sorted(runs)):
        xw = [d["step"] for d in runs[run] if d["wk"] is not None]; yw = [d["wk"] for d in runs[run] if d["wk"] is not None]
        xf = [d["step"] for d in runs[run] if d["fl"] is not None]; yf = [d["fl"] for d in runs[run] if d["fl"] is not None]
        c = pal[i % len(pal)]
        if xw: ax[0].plot(xw, yw, "o-", color=c, label=run)
        if xf: ax[1].plot(xf, yf, "o-", color=c, label=run)
    ax[0].axhline(BASE["watkins"], ls="--", c="gray", lw=.8); ax[0].set_title("Watkins species macro-F1"); ax[0].set_xlabel("step"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    ax[1].axhline(BASE["filt"], ls="--", c="gray", lw=.8); ax[1].set_title("filtration macro-F1"); ax[1].set_xlabel("step"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(outp("dynamics.png"), dpi=120)
    log(f"DYNAMICS saved {outp('dynamics.png')} ({sum(len(v) for v in runs.values())} points, {len(runs)} runs)")

def _task_scores(tag):
    wk = load_json("watkins_results.json", {}).get(tag, {})
    kc = load_json("kclass_results.json", {}).get(tag, {})
    fl = load_json("filter_results.json", {}).get(tag, {})
    return {
        "watkins": wk.get("best_macro_f1"),
        "kclass": kc.get("best_macro_f1"),
        "filter": (fl.get("best") or {}).get("macro_f1") if isinstance(fl, dict) else None,
    }

def _composite_score(scores, weights):
    total = 0.0
    for task, weight in weights.items():
        value = scores.get(task)
        if value is None:
            return None
        total += float(weight) * float(value)
    return total

def _set_best_link(keep_dir, link_name, target_name):
    link = os.path.join(keep_dir, link_name)
    try:
        if os.path.lexists(link): os.remove(link)
        os.symlink(target_name, link)
    except OSError:
        pass

def _best_link_targets(keep_dir):
    targets = set()
    for name in ("BEST.pt", "BEST_WATKINS.pt", "BEST_KCLASS.pt", "BEST_FILTER.pt", "BEST_COMPOSITE.pt"):
        link = os.path.join(keep_dir, name)
        if os.path.lexists(link):
            targets.add(os.path.join(keep_dir, os.readlink(link)))
    return targets

def _prune_preserved(keep_dir, keep_best, pat, weights):
    """Keep top composite-score checkpoints plus the most recent checkpoints and all BEST_* targets."""
    files = [f for f in glob.glob(os.path.join(keep_dir, "checkpoint_*_*.pt")) if pat.search(os.path.basename(f))]
    if len(files) <= keep_best * 2: return
    score = lambda f: (_composite_score(_task_scores(_tag(f)), weights) or -1.0)
    step = lambda f: int(pat.search(os.path.basename(f)).group(2))
    keep = set(sorted(files, key=score, reverse=True)[:keep_best]) | set(sorted(files, key=step, reverse=True)[:keep_best])
    keep |= _best_link_targets(keep_dir)
    for f in files:
        if f in keep: continue
        try: os.remove(f); log(f"watch: pruned {os.path.basename(f)} (composite={score(f):.3f})")
        except OSError: pass

def task_watch(a):
    """Watch live training checkpoints, validate Watkins/K-class/filter, and track per-task plus composite bests."""
    import subprocess, shutil
    save_dir = os.path.abspath(a.save_dir)
    keep_dir = a.preserve or os.path.join(save_dir, "validated"); os.makedirs(keep_dir, exist_ok=True)
    pat = re.compile(r"checkpoint_(\d+)_(\d+)\.pt$")
    weights = {"watkins": a.weight_watkins, "kclass": a.weight_kclass, "filter": a.weight_filter}
    seen = set()
    best = {
        "watkins": {"score": -1.0, "step": None, "name": None},
        "kclass": {"score": -1.0, "step": None, "name": None},
        "filter": {"score": -1.0, "step": None, "name": None},
        "composite": {"score": -1.0, "step": None, "name": None},
    }
    idle = 0
    reg = load_json("dynamics_registry.json", {}); wk = load_json("watkins_results.json", {})  # restart-safe
    for tag, meta in reg.items():
        if meta.get("run") == a.run and tag in wk:
            seen.add(int(meta["step"]))
            scores = _task_scores(tag)
            for task in ("watkins", "kclass", "filter"):
                if scores.get(task) is not None and scores[task] > best[task]["score"]:
                    best[task].update(score=scores[task], step=int(meta["step"]), name=tag + ".pt")
            composite = _composite_score(scores, weights)
            if composite is not None and composite > best["composite"]["score"]:
                best["composite"].update(score=composite, step=int(meta["step"]), name=tag + ".pt")
    for task, link_name in (
        ("watkins", "BEST_WATKINS.pt"),
        ("kclass", "BEST_KCLASS.pt"),
        ("filter", "BEST_FILTER.pt"),
        ("composite", "BEST_COMPOSITE.pt"),
    ):
        if best[task]["name"] and os.path.exists(os.path.join(keep_dir, best[task]["name"])):
            _set_best_link(keep_dir, link_name, best[task]["name"])
    if best["composite"]["name"] and os.path.exists(os.path.join(keep_dir, best["composite"]["name"])):
        _set_best_link(keep_dir, "BEST.pt", best["composite"]["name"])
    log(f"watch: {save_dir} -> {keep_dir} | run={a.run} poll={a.poll}s device={a.device} | weights={weights} | {len(seen)} already done")
    while True:
        cands = sorted((int(m.group(2)), ck) for ck in glob.glob(os.path.join(save_dir, "checkpoint_*_*.pt"))
                       for m in [pat.search(os.path.basename(ck))] if m and int(m.group(2)) not in seen)
        for step, ck in cands:
            dst = os.path.join(keep_dir, os.path.basename(ck))
            if not os.path.exists(dst):
                try: os.link(ck, dst)
                except OSError:
                    try: shutil.copy2(ck, dst)
                    except OSError as e: log(f"watch: preserve failed step {step}: {e}"); continue
            log(f"watch: preserved + validating step {step}")
            env = dict(os.environ)
            if a.device is not None: env["CUDA_VISIBLE_DEVICES"] = str(a.device)
            if a.tensorboard_logdir: env["A2V_TENSORBOARD_LOGDIR"] = a.tensorboard_logdir
            cmd = [sys.executable, os.path.abspath(__file__), "watkins", dst, "--run", a.run, "--step", str(step)]
            if not a.baselines: cmd.append("--no-baselines")
            if a.limit: cmd += ["--limit", str(a.limit)]
            rc = subprocess.run(cmd, env=env).returncode
            seen.add(step)
            if rc != 0: log(f"watch: validate rc={rc} step {step} (checkpoint kept; no metric)"); continue
            for task_cmd in (
                ["kclass", dst, "--run", a.run, "--step", str(step), "--n-per-class", "100"],
                ["filter", dst, "--run", a.run, "--step", str(step)],
            ):
                if not a.baselines and task_cmd[0] == "filter":
                    task_cmd.append("--no-baselines")
                task_rc = subprocess.run([sys.executable, os.path.abspath(__file__)] + task_cmd, env=env).returncode
                if task_rc != 0:
                    log(f"watch: {task_cmd[0]} rc={task_rc} step {step} (available metrics kept)")
            scores = _task_scores(_tag(dst))
            for task, link_name in (
                ("watkins", "BEST_WATKINS.pt"),
                ("kclass", "BEST_KCLASS.pt"),
                ("filter", "BEST_FILTER.pt"),
            ):
                value = scores.get(task)
                if value is not None and value > best[task]["score"]:
                    best[task].update(score=value, step=step, name=os.path.basename(dst))
                    _set_best_link(keep_dir, link_name, os.path.basename(dst))
                    log(f"watch: ** NEW {link_name[:-3]} step {step} score={value:.4f} **")
            composite = _composite_score(scores, weights)
            if composite is not None and composite > best["composite"]["score"]:
                best["composite"].update(score=composite, step=step, name=os.path.basename(dst))
                _set_best_link(keep_dir, "BEST_COMPOSITE.pt", os.path.basename(dst))
                _set_best_link(keep_dir, "BEST.pt", os.path.basename(dst))
                log(f"watch: ** NEW BEST_COMPOSITE step {step} score={composite:.4f} "
                    f"(watkins={scores['watkins']:.4f}, kclass={scores['kclass']:.4f}, filter={scores['filter']:.4f}) **")
            subprocess.run([sys.executable, os.path.abspath(__file__), "dynamics"], env=env)
            if a.keep_best: _prune_preserved(keep_dir, a.keep_best, pat, weights)
        if cands:
            idle = 0
            log("watch: best so far "
                f"composite step {best['composite']['step']} score={best['composite']['score']:.4f}; "
                f"watkins step {best['watkins']['step']} score={best['watkins']['score']:.4f}; "
                f"kclass step {best['kclass']['step']} score={best['kclass']['score']:.4f}; "
                f"filter step {best['filter']['step']} score={best['filter']['score']:.4f}; "
                f"({len(seen)} validated)")
        else:
            idle += 1
            if a.max_idle and idle >= a.max_idle:
                log(f"watch: {idle} idle polls - stopping. BEST_COMPOSITE step {best['composite']['step']} "
                    f"score={best['composite']['score']:.4f} -> {keep_dir}/BEST_COMPOSITE.pt"); break
        time.sleep(a.poll)

def task_slim(a):
    """raw ~5GB training checkpoint -> ~1.3GB inference checkpoint (strip optimizer + EMA, sanitize cfg)."""
    _setup_fairseq()
    out = a.out or a.ckpt.replace(".pt", "") + "_slim.pt"
    sanitize_and_save(a.ckpt, out)
    log(f"SLIM saved {out}")

# ======================= plumbing =======================
def _tag(ckpt): return os.path.basename(ckpt).replace("_slim.pt", "").replace(".pt", "")
def _record(task, a, res, headline):
    store = f"{task}_results.json"; allr = load_json(store, {}); allr[_tag(a.ckpt)] = res; save_json(store, allr)
    if getattr(a, "run", None) and getattr(a, "step", None) is not None:
        reg = load_json("dynamics_registry.json", {}); reg[_tag(a.ckpt)] = {"run": a.run, "step": int(a.step)}
        save_json("dynamics_registry.json", reg)
        _write_tensorboard(task, a, res, headline)
    save_json(f"{task}_{_tag(a.ckpt)}.json", res)

def main():
    ap = argparse.ArgumentParser(description="animal2vec checkpoint validation (single CLI)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    def add_ckpt(p, run_step=False, bl=False):
        p.add_argument("ckpt")
        if run_step:
            p.add_argument("--run", default=None, help="run label for dynamics (e.g. blue)")
            p.add_argument("--step", type=int, default=None, help="training step for dynamics")
        if bl:
            p.add_argument("--no-baselines", dest="baselines", action="store_false", help="skip AVES-8k/log-mel-8k calibration")
            p.set_defaults(baselines=True)
    wp = sub.add_parser("watkins"); add_ckpt(wp, run_step=True, bl=True); wp.add_argument("--limit", type=int, default=0, help="cap clips/split for a quick smoke run")
    fp = sub.add_parser("filter");  add_ckpt(fp, run_step=True, bl=True); fp.add_argument("--limit", type=int, default=0, help="cap clips/class for a quick smoke run")
    sp = sub.add_parser("shap");    add_ckpt(sp);                         sp.add_argument("--limit", type=int, default=0, help="cap clips/class for a quick smoke run")
    kc = sub.add_parser("kclass"); add_ckpt(kc, run_step=True); kc.add_argument("--n-per-class", type=int, default=100)
    sl = sub.add_parser("slim"); sl.add_argument("ckpt"); sl.add_argument("--out", default=None, help="output path (default <ckpt>_slim.pt)")
    sub.add_parser("dynamics")
    wt = sub.add_parser("watch"); wt.add_argument("save_dir", help="live training checkpoint dir to watch")
    wt.add_argument("--run", required=True, help="run label for the dynamics curve")
    wt.add_argument("--poll", type=int, default=120, help="seconds between scans")
    wt.add_argument("--device", default=None, help="CUDA_VISIBLE_DEVICES for the validation subprocess (e.g. 0)")
    wt.add_argument("--preserve", default=None, help="dir to hard-link checkpoints into (default <save_dir>/validated)")
    wt.add_argument("--keep-best", type=int, default=5, help="prune preserved/ to top-N by composite score + N most-recent (0=keep all)")
    wt.add_argument("--max-idle", type=int, default=0, help="stop after N idle polls with no new checkpoint (0=forever)")
    wt.add_argument("--limit", type=int, default=0, help="cap clips/split for faster (approximate) live validation")
    wt.add_argument("--tensorboard-logdir", default=None, help="TensorBoard scalar root (default A2V_OUT/tensorboard)")
    wt.add_argument("--weight-watkins", type=float, default=1.0, help="composite score weight for Watkins macro-F1")
    wt.add_argument("--weight-kclass", type=float, default=1.0, help="composite score weight for Olga K-class macro-F1")
    wt.add_argument("--weight-filter", type=float, default=1.0, help="composite score weight for signal/noise filter macro-F1")
    wt.add_argument("--no-baselines", dest="baselines", action="store_false"); wt.set_defaults(baselines=True)
    a = ap.parse_args()
    {"watkins": task_watkins, "filter": task_filter, "shap": task_shap, "kclass": task_kclass,
     "slim": task_slim, "dynamics": task_dynamics, "watch": task_watch}[a.cmd](a)

if __name__ == "__main__":
    main()
