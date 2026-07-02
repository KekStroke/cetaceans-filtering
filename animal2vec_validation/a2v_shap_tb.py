#!/usr/bin/env python3
"""Proper SHAP (Shapley) frequency-band attribution for an animal2vec checkpoint + TensorBoard.

Unlike a2v_shap.py (single-band OCCLUSION, i.e. only the marginal of removing one band from the
full mix), this computes true SHAPLEY VALUES over frequency-band "players": each 0.5 kHz band is a
coalition member, a coalition S is rendered by summing the band-pass components for bands in S
(empty S = silence baseline), and phi_i is band i's Shapley contribution to the frozen probe's
predicted probability of the true K-call class. Exact enumeration when #bands <= 10, otherwise an
unbiased permutation-sampling estimator. Sample rate (hence #bands) is auto-detected from the
checkpoint, so this is correct for 8 kHz (8 bands, exact) and 16 kHz (16 bands, sampled).

Writes: a per-checkpoint PNG + JSON, and logs the figure + per-band scalars to TensorBoard
(one run dir per checkpoint). Run ONE checkpoint at a time:
    uv run python animal2vec_validation/a2v_shap_tb.py <ckpt> [--shap-clips 4] [--perms 32]
"""
import os, sys, time, json, argparse, itertools, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a2v_extract as E
import numpy as np, torch
from scipy.signal import butter, sosfiltfilt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

HERE = os.path.dirname(os.path.abspath(__file__))
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def bandpass_components(wav, sr, bands):
    """Decompose wav into one band-limited component per band (sum ~= wav)."""
    comps = []
    for lo, hi in bands:
        lo_n = max(lo, 1) / (sr / 2); hi_n = min(hi, sr / 2 - 1) / (sr / 2)
        if lo <= 0:                                    # lowest band -> low-pass
            sos = butter(4, hi_n, btype='low', output='sos')
        elif hi >= sr / 2:                             # top band -> high-pass
            sos = butter(4, lo_n, btype='high', output='sos')
        else:
            sos = butter(4, [lo_n, hi_n], btype='band', output='sos')
        comps.append(sosfiltfilt(sos, wav).astype(np.float32))
    return np.stack(comps)                             # (N, T)


def _norm(y):
    y = y - y.mean(); s = y.std()
    return (y / (s + 1e-8) if s > 1e-8 else y).astype(np.float32)


@torch.inference_mode()
def probe_prob_batch(model, waves, BL, sc, clf, ci, dev, bs=48):
    """Embed a list of same-length waveforms at layer BL, return probe prob for class ci."""
    out = []
    for i in range(0, len(waves), bs):
        chunk = waves[i:i + bs]
        x = torch.tensor(np.stack(chunk)).to(dev)
        o = model(source=x, features_only=True)
        lr = o["layer_results"][BL]
        t = lr[0] if isinstance(lr, (tuple, list)) else lr        # (T,B,D) or (B,T,D)
        t = t.transpose(0, 1) if t.shape[0] != x.shape[0] else t   # -> (B,T,D)
        emb = t.float().mean(1).cpu().numpy()                      # (B,D)
        out.append(clf.predict_proba(sc.transform(emb))[:, ci])
    return np.concatenate(out)


def shapley_exact(N, v):
    """Exact Shapley from a dict v: frozenset(coalition)->value. Returns phi[N]."""
    phi = np.zeros(N)
    facts = [math.factorial(k) for k in range(N + 1)]
    full = list(range(N))
    for i in range(N):
        rest = [j for j in full if j != i]
        for r in range(len(rest) + 1):
            w = facts[r] * facts[N - r - 1] / facts[N]
            for S in itertools.combinations(rest, r):
                fs = frozenset(S)
                phi[i] += w * (v[fs | {i}] - v[fs])
    return phi


def collect_subsets_exact(N):
    return [frozenset(S) for r in range(N + 1) for S in itertools.combinations(range(N), r)]


def shapley_perm(N, v, perms):
    """Unbiased permutation-sampling Shapley from v (dict) over given permutations."""
    phi = np.zeros(N);
    for p in perms:
        pre = frozenset()
        for i in p:
            nxt = pre | {i}
            phi[i] += v[nxt] - v[pre]
            pre = nxt
    return phi / len(perms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--lab-dir", default=os.environ.get("A2V_LAB_DIR", "/mnt/c/Users/Iaroslav/CETACEANS/new_training_data"))
    ap.add_argument("--probe-per-class", type=int, default=60)
    ap.add_argument("--shap-clips", type=int, default=4)      # clips/class for SHAP (heavy)
    ap.add_argument("--perms", type=int, default=32)          # permutations when sampling
    ap.add_argument("--tb", default=os.path.join(HERE, "tb_shap"))
    a = ap.parse_args()

    model = E.load_model(a.ckpt)                              # sets E.SR from ckpt cfg
    SR = E.SR
    NB = int(round((SR / 2) / 500)); BANDS = [(i * 500, (i + 1) * 500) for i in range(NB)]
    tag = a.ckpt.split("/")[-1].replace(".pt", "")
    log(f"{tag}: SR={SR} -> {NB} bands (0..{SR/2/1000:g} kHz)")

    items = E.gather_kclass(a.lab_dir, a.probe_per_class)
    files = [f for f, _, _ in items]
    le = LabelEncoder(); y = le.fit_transform([c for _, c, _ in items])
    groups = np.array([g for _, _, g in items]); classes = list(le.classes_)

    # --- extract all-layer embeddings on the probe set, pick best probe layer ---
    @torch.inference_mode()
    def embed_layers(wav):
        x = torch.tensor(wav).view(1, -1).to(E.DEV)
        o = model(source=x, features_only=True)
        return [ (lr[0] if isinstance(lr, (tuple, list)) else lr).reshape(-1, (lr[0] if isinstance(lr,(tuple,list)) else lr).shape[-1]).float().mean(0).cpu().numpy()
                 for lr in (o.get("layer_results") or []) ]
    XL = None
    for i, f in enumerate(files):
        fs = embed_layers(E.load_audio(f, 0))
        if XL is None: XL = [[] for _ in fs]
        for li, v in enumerate(fs): XL[li].append(v)
        if i % 200 == 0: log(f"  probe-embed {i}/{len(files)}")
    XL = [np.stack(p) for p in XL]
    def probe_f1(X):
        gkf = GroupKFold(5); T, P = [], []
        for tr, te in gkf.split(X, y, groups):
            s = StandardScaler().fit(X[tr]); c = LogisticRegression(max_iter=2000, C=1, class_weight='balanced', n_jobs=-1).fit(s.transform(X[tr]), y[tr])
            P.append(c.predict(s.transform(X[te]))); T.append(y[te])
        return f1_score(np.concatenate(T), np.concatenate(P), average='macro')
    f1s = [probe_f1(X) for X in XL]; BL = int(np.argmax(f1s))
    log(f"best layer L{BL} (probe F1={f1s[BL]:.3f})")
    Xb = XL[BL]; sc = StandardScaler().fit(Xb)
    clf = LogisticRegression(max_iter=2000, C=1, class_weight='balanced', n_jobs=-1).fit(sc.transform(Xb), y)

    # --- Shapley over bands, per class ---
    exact = NB <= 10
    if exact:
        subsets = collect_subsets_exact(NB); log(f"exact Shapley: {len(subsets)} coalitions/clip")
        perms = None
    else:
        rng = np.random.RandomState(0)
        perms = [list(rng.permutation(NB)) for _ in range(a.perms)]
        need = set([frozenset()])
        for p in perms:
            pre = frozenset()
            for i in p: pre = pre | {i}; need.add(pre)
        subsets = list(need); log(f"perm-sampling Shapley: {a.perms} perms -> {len(subsets)} unique coalitions/clip")

    import collections
    byc = collections.defaultdict(list)
    for f, c, _ in items: byc[c].append(f)
    shap = np.zeros((len(classes), NB)); eng = np.zeros((len(classes), NB)); cnt = np.zeros(len(classes))
    for ci, c in enumerate(classes):
        for f in byc[c][:a.shap_clips]:
            w0 = E.load_audio(f, 0)
            comps = bandpass_components(w0, SR, BANDS)                     # (NB, T)
            # band energy (for the attribution-vs-energy overlay)
            ff = np.fft.rfftfreq(len(w0), 1 / SR); P = np.abs(np.fft.rfft(w0)) ** 2
            e = np.array([P[(ff >= lo) & (ff < hi)].sum() for lo, hi in BANDS]); eng[ci] += e / (e.sum() + 1e-9)
            # render + embed every needed coalition
            waves = [ _norm(comps[list(S)].sum(0)) if len(S) else _norm(np.zeros_like(w0)) for S in subsets ]
            probs = probe_prob_batch(model, waves, BL, sc, clf, ci, E.DEV)
            v = {S: float(pp) for S, pp in zip(subsets, probs)}
            phi = shapley_exact(NB, v) if exact else shapley_perm(NB, v, perms)
            shap[ci] += phi; cnt[ci] += 1
        log(f"  SHAP {c} done ({int(cnt[ci])} clips)")
    shap /= cnt[:, None]; eng /= cnt[:, None]
    r = float(np.corrcoef(shap.mean(0), eng.mean(0))[0, 1])

    bl = [f"{lo/1000:g}-{hi/1000:g}" for lo, hi in BANDS]
    out = {"ckpt": tag, "method": "kernel-shapley-exact" if exact else f"shapley-perm-{a.perms}",
           "sample_rate": SR, "best_layer": BL, "best_f1": float(f1s[BL]), "bands_kHz": bl,
           "shap_per_class": {classes[i]: shap[i].tolist() for i in range(len(classes))},
           "mean_shap": shap.mean(0).tolist(), "mean_energy": eng.mean(0).tolist(),
           "attr_vs_energy_pearson": r, "shap_clips_per_class": a.shap_clips}
    jpath = os.path.join(HERE, f"a2v_shap_tb_{SR//1000}k.json"); json.dump(out, open(jpath, "w"), indent=2)

    # --- figure ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    m = np.abs(shap).max()
    im = ax[0].imshow(shap, aspect='auto', cmap='RdBu_r', vmin=-m, vmax=m)
    ax[0].set_yticks(range(len(classes))); ax[0].set_yticklabels(classes, fontsize=8)
    ax[0].set_xticks(range(NB)); ax[0].set_xticklabels(bl, rotation=45, fontsize=7)
    ax[0].set_title(f"animal2vec SHAP (Shapley) band importance — {tag[:22]}\nlayer L{BL}, {'exact' if exact else str(a.perms)+' perms'}, {SR//1000} kHz")
    ax[0].set_xlabel("frequency band (kHz)"); plt.colorbar(im, ax=ax[0], label="mean Shapley value (Δ true-class prob)")
    ax[1].plot(range(NB), shap.mean(0), 'o-', label='SHAP (mean over classes)')
    ax[1].plot(range(NB), eng.mean(0), 's--', c='gray', label='call energy (mean)')
    ax[1].axhline(0, c='k', lw=.6); ax[1].set_xticks(range(NB)); ax[1].set_xticklabels(bl, rotation=45, fontsize=7)
    ax[1].legend(); ax[1].grid(alpha=.3); ax[1].set_xlabel("frequency band (kHz)")
    ax[1].set_title(f"Shapley vs call energy (Pearson r = {r:.2f})")
    plt.tight_layout()
    ppath = os.path.join(HERE, f"a2v_shap_tb_{SR//1000}k.png"); plt.savefig(ppath, dpi=130)

    # --- TensorBoard (one run dir per checkpoint) ---
    tbdir = os.path.join(a.tb, f"{tag[:40]}_{SR//1000}k")
    sw = SummaryWriter(tbdir)
    sw.add_figure("shap/band_importance", fig, global_step=0)
    for i, name in enumerate(bl):
        sw.add_scalar(f"shap_mean/{i:02d}_{name}kHz", float(shap.mean(0)[i]), 0)
        sw.add_scalar(f"call_energy/{i:02d}_{name}kHz", float(eng.mean(0)[i]), 0)
    sw.add_scalar("shap/attr_vs_energy_pearson", r, 0)
    sw.add_scalar("probe/best_layer", BL, 0); sw.add_scalar("probe/best_f1", float(f1s[BL]), 0)
    sw.add_text("shap/summary",
                f"{tag} | {SR//1000}kHz | L{BL} F1={f1s[BL]:.3f} | r={r:.3f} | "
                f"top bands: " + ", ".join(f"{bl[j]}({shap.mean(0)[j]:+.3f})" for j in np.argsort(-shap.mean(0))[:3]), 0)
    sw.close()

    print(f"\n=== animal2vec SHAP (Shapley, L{BL}, {SR//1000}kHz) ===")
    print("mean Shapley:", " ".join(f"{bl[i]}={shap.mean(0)[i]:+.3f}" for i in range(NB)))
    print(f"Shapley-vs-energy Pearson r = {r:.3f}")
    print(f"saved: {ppath}\n        {jpath}\nTensorBoard: {tbdir}")


if __name__ == "__main__":
    main()
