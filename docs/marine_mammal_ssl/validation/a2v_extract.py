#!/usr/bin/env python3
"""
animal2vec (data2vec_multi) embedding extractor — MODERN GPU env (torch 2.9 + cu128 + fairseq 0.12.2).
Run with: /tmp/a2v_env/bin/python a2v_extract.py ...   (a2v repo on sys.path for `import nn`)

Loads a pretraining checkpoint and extracts mean-pooled encoder features (features_only=True) on GPU.
Robust to Anvar's-fork extra config keys (e.g. multi_corpus_keys): they are stripped from the
checkpoint cfg before load — the MODEL is standard data2vec_multi, so the weights load losslessly.
Model trained on 10s @ 8kHz (80000 samples); handles variable length (--target-len 0 = native).
Saves an .npz {X, y, groups, classes} that a2v_validate.py consumes.
"""
import os, sys, argparse, glob, collections, time, types, re
sys.path.insert(0, "/tmp/a2v")
import torch
# torch._six shim (removed in torch 2.x; fairseq 0.12.2 still imports it)
try:
    import torch._six  # noqa
except Exception:
    _m = types.ModuleType("torch._six"); _m.string_classes = (str, bytes); _m.int_classes = (int,)
    import collections.abc as _abc; _m.container_abcs = _abc; sys.modules["torch._six"] = _m
import numpy as np, soundfile as sf, librosa
import nn  # noqa: F401  (registers data2vec_multi + audio_ccas task)
from fairseq import checkpoint_utils
from omegaconf import OmegaConf, open_dict

# --- patch version-mismatched fairseq fns the a2v code calls with newer kwargs (inference-safe) ---
import inspect as _inspect
def _swallow_kwargs(modobj, fname):
    if not hasattr(modobj, fname): return
    orig = getattr(modobj, fname)
    try: sig = set(_inspect.signature(orig).parameters)
    except (ValueError, TypeError): return
    def wrapped(*a, **k): return orig(*a, **{kk: vv for kk, vv in k.items() if kk in sig})
    setattr(modobj, fname, wrapped)
try:
    import nn.modalities.base as _nb
    _swallow_kwargs(_nb, "compute_mask_indices")
except Exception as _e:
    print("compute_mask_indices patch skipped:", _e)

SR = 8000
DEV = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def _del_key(node, fullkey):
    parts = fullkey.split('.'); n = node
    for p in parts[:-1]:
        try: n = n[p]
        except Exception: return False
    k = parts[-1]
    if isinstance(n, dict):                       # plain dict cfg
        if k in n: del n[k]; return True
        return False
    try:                                          # OmegaConf DictConfig
        with open_dict(n):
            if k in n: del n[k]; return True
    except Exception:
        pass
    return False


def sanitize_and_save(src, dst):
    """Strip cfg keys not in the public dataclasses (Anvar-fork additions) so the model loads."""
    log("loading checkpoint for sanitize (5GB, ~30s) ...")
    ck = torch.load(src, map_location='cpu', weights_only=False)
    cfg = ck['cfg']
    from nn.audio_tasks import AudioConfigCCAS
    try:
        from nn.data2vec2 import Data2VecMultiConfig
        sections = [('task', AudioConfigCCAS), ('model', Data2VecMultiConfig)]
    except Exception:
        sections = [('task', AudioConfigCCAS)]
    for sec, dc in sections:
        if cfg.get(sec) is None: continue
        try:
            schema = OmegaConf.structured(dc)
        except Exception as e:
            log(f"  [sanitize] can't structure {sec} schema ({type(e).__name__}); skipping"); continue
        for _ in range(80):
            try:
                OmegaConf.merge(schema, cfg[sec]); break
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
    # inference needs ONLY the student encoder -> skip EMA teacher (avoids EMAModuleConfig.log_norms
    # version mismatch) and drop its weights so the load is not strict-blocked.
    mc = cfg.get('model')
    if mc is not None:
        try: mc['skip_ema'] = True
        except Exception:
            with open_dict(mc): mc['skip_ema'] = True
    sd = ck.get('model', {})
    for k in list(sd.keys()):
        if k == '_ema' or k.startswith('_ema') or k.startswith('ema.'):
            del sd[k]
    log("saving sanitized checkpoint (skip_ema, _ema dropped) ...")
    torch.save(ck, dst)
    return dst


def load_model(ckpt):
    san = "/tmp/ckpt_sanitized.pt"
    sanitize_and_save(ckpt, san)
    log("load_model_ensemble ...")
    models, _ = checkpoint_utils.load_model_ensemble([san])
    m = models[0].to(DEV).eval()
    log(f"model on {DEV}: {type(m).__name__}, {sum(p.numel() for p in m.parameters())/1e6:.0f}M params")
    return m


def load_audio(path, target_len):
    y, s = sf.read(path, dtype="float32")
    if y.ndim > 1: y = y.mean(1)
    if s != SR: y = librosa.resample(y, orig_sr=s, target_sr=SR)
    if target_len and target_len > 0:
        y = np.pad(y, (0, target_len - len(y))) if len(y) < target_len else y[:target_len]
    if len(y) < 400: y = np.pad(y, (0, 400 - len(y)))
    y = y - y.mean(); st = y.std(); y = y / (st + 1e-8) if st > 1e-8 else y
    return y.astype(np.float32)


@torch.inference_mode()
def embed(model, files, target_len, layer):
    embs = []
    for i, f in enumerate(files):
        x = torch.tensor(load_audio(f, target_len)).view(1, -1).to(DEV)
        out = model(source=x, features_only=True)
        if layer is not None and out.get("layer_results"):
            lr = out["layer_results"][layer]; feat = lr[0] if isinstance(lr, (tuple, list)) else lr
            feat = feat.transpose(0, 1) if feat.shape[0] != 1 else feat
            v = feat[0].float().mean(0)
        else:
            v = out["x"][0].float().mean(0)
        embs.append(v.cpu().numpy())
        if i % 200 == 0: log(f"  embed {i}/{len(files)}")
    return np.stack(embs)


def class_of(name): return 'noise' if name.startswith('noise') else name.split('-')[0]
def tape_key(p):
    parts = os.path.basename(p)[:-4].split('_'); return parts[3] if len(parts) >= 4 else os.path.basename(p)


def gather_kclass(lab_dir, n_per_class):
    CLASSES = ['K1','K10','K12','K13','K14','K17','K21','K27','K4','K5','K7','noise']
    by = collections.defaultdict(list)
    for f in sorted(glob.glob(f"{lab_dir}/*.wav")):
        c = class_of(os.path.basename(f))
        if c in set(CLASSES): by[c].append(f)
    rng = np.random.RandomState(42); items = []
    for c in CLASSES:
        fs = by[c][:]; rng.shuffle(fs)
        for f in fs[:n_per_class]: items.append((f, c, tape_key(f)))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--task", default="olga_kclass")
    ap.add_argument("--lab-dir", default="/mnt/c/Users/Iaroslav/CETACEANS/new_training_data")
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--target-len", type=int, default=80000)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model = load_model(a.ckpt)
    items = gather_kclass(a.lab_dir, a.n_per_class)
    if a.smoke: items = items[:a.smoke]
    files = [f for f, _, _ in items]
    log(f"task={a.task} clips={len(files)} target_len={a.target_len} layer={a.layer} dev={DEV}")
    X = embed(model, files, a.target_len, a.layer)
    y = np.array([c for _, c, _ in items]); groups = np.array([g for _, _, g in items])
    log(f"embeddings {X.shape}")
    np.savez(a.out, X=X.astype(np.float32), y=y, groups=groups, classes=np.array(sorted(set(y.tolist()))))
    log(f"saved {a.out}")


if __name__ == "__main__":
    main()
