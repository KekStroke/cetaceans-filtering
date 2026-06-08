#!/usr/bin/env python3
"""
Real instrumented BEST-RQ training run on Olga marine audio (the key artifact).
Demonstrates the methodology the team needs: an INTERPRETABLE loss that falls from
~ln(8192)=9.0, with a FROZEN LINEAR PROBE (recording-disjoint) that RISES as the
encoder learns — i.e. "is it training?" answered by construction, plus collapse guards
(target perplexity stays high, grad-norm stable). This is Anvar's drop-in.

Self-supervised (labels ignored for training; used only for the probe health metric).
"""
import os, sys, json, time, glob, collections
os.environ.setdefault("PYTORCH_ALLOC_CONF", "max_split_size_mb:128")
import numpy as np, soundfile as sf, librosa, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bestrq import BestRQModel, BestRQConfig, StepLogger, count_params
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAB = "/mnt/c/Users/Iaroslav/CETACEANS/new_training_data"
OUT = "/mnt/c/Users/Iaroslav/CETACEANS/ml-intern-runs/bestrq-marine/path-2"
CLIP = 16000          # 1.0 s @ 16 kHz
N_TRAIN = 10000       # unlabeled clips held in RAM
STEPS = 2000; BS = 64; LR = 5e-4; WARMUP = 120; CLIP_GRAD = 1.0
PROBE_EVERY = 250; PROBE_PER_CLASS = 130
CLASSES = ['K1','K10','K12','K13','K14','K17','K21','K27','K4','K5','K7','noise']
CIDX = {c: i for i, c in enumerate(CLASSES)}
SEED = 0; rng = np.random.RandomState(SEED); torch.manual_seed(SEED)
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def class_of(n): return 'noise' if n.startswith('noise') else n.split('-')[0]
def tape_key(p):
    parts = os.path.basename(p)[:-4].split('_'); return parts[3] if len(parts) >= 4 else os.path.basename(p)


def load_audio(path, n=CLIP):
    y, sr = sf.read(path, dtype='float32')
    if y.ndim > 1: y = y.mean(1)
    if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    if len(y) < n: y = np.pad(y, (0, n - len(y)))
    else:
        s = (len(y) - n) // 2; y = y[s:s + n]
    return y.astype(np.float32)


# ---- gather files ----
allwav = sorted(glob.glob(f"{LAB}/*.wav"))
by = collections.defaultdict(list)
for f in allwav:
    c = class_of(os.path.basename(f))
    if c in CIDX: by[c].append(f)

# probe set (labeled, recording-disjoint): PROBE_PER_CLASS / class
probe_items = []
for c in CLASSES:
    fs = by[c][:]; rng.shuffle(fs);
    for f in fs[:PROBE_PER_CLASS]: probe_items.append((f, CIDX[c], tape_key(f)))
probe_paths = set(f for f, _, _ in probe_items)

# training set (UNLABELED): random sample of the rest, all classes
train_pool = [f for f in allwav if f not in probe_paths]
rng.shuffle(train_pool); train_pool = train_pool[:N_TRAIN]
log(f"loading {len(train_pool)} train + {len(probe_items)} probe clips ...")
Xtr = np.stack([load_audio(f) for f in train_pool]);
log(f"train audio {Xtr.shape}")
Xpr = np.stack([load_audio(f) for f, _, _ in probe_items])
ypr = np.array([l for _, l, _ in probe_items]); gpr = np.array([g for _, _, g in probe_items])
Xtr_t = torch.from_numpy(Xtr)
Xpr_t = torch.from_numpy(Xpr)
log(f"probe audio {Xpr.shape}, {len(set(gpr))} recordings")

# ---- model ----
cfg = BestRQConfig()
model = BestRQModel(cfg).to(DEV)
log(f"BestRQ params: {count_params(model)/1e6:.2f}M (trainable {count_params(model, True)/1e6:.2f}M)")
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.98))
def lr_at(step):
    if step < WARMUP: return LR * step / WARMUP
    p = (step - WARMUP) / max(1, STEPS - WARMUP)
    return 0.5 * LR * (1 + np.cos(np.pi * p))
logger = StepLogger()


@torch.no_grad()
def embed(audio_t, bs=128):
    model.eval(); embs = []
    for i in range(0, len(audio_t), bs):
        a = audio_t[i:i + bs].to(DEV)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            mel = model.frontend(a); x = model.subsample(mel)
            x = model.pos_enc(x); x = model.encoder(x); x = model.encoder_norm(x)
        embs.append(x.float().mean(1).cpu().numpy())
    model.train(); return np.concatenate(embs, 0)


def run_probe():
    E = embed(Xpr_t)
    gkf = GroupKFold(min(5, len(set(gpr)))); fs = []
    for tr, te in gkf.split(E, ypr, gpr):
        sc = StandardScaler().fit(E[tr])
        clf = LogisticRegression(max_iter=1500, C=1.0, class_weight='balanced', n_jobs=-1).fit(sc.transform(E[tr]), ypr[tr])
        fs.append(f1_score(ypr[te], clf.predict(sc.transform(E[te])), average='macro'))
    return float(np.mean(fs))


# ---- train ----
probe_hist = []
p0 = run_probe(); probe_hist.append((0, p0))
log(f"PROBE @ step 0 (random encoder): macro-F1 = {p0:.4f} (chance ≈ {1/12:.3f})")
model.train()
for step in range(1, STEPS + 1):
    idx = rng.randint(0, len(Xtr_t), BS)
    audio = Xtr_t[idx].to(DEV)
    for g in opt.param_groups: g['lr'] = lr_at(step)
    opt.zero_grad()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model(audio)
    out["loss"].backward()
    gn = StepLogger.grad_norm(model.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
    opt.step()
    rec = logger.log(step, out, gn)
    if step % 100 == 0:
        log(f"step {step:4d} loss={rec['loss']:.3f} acc={rec['masked_acc']:.3f} "
            f"tgt_ppl={rec['target_perplexity']:.0f} pred_ppl={rec['pred_perplexity']:.0f} gn={gn:.2f} lr={lr_at(step):.1e}")
    if step % PROBE_EVERY == 0:
        pf = run_probe(); probe_hist.append((step, pf))
        log(f"   >> PROBE @ step {step}: macro-F1 = {pf:.4f}")

# ---- save ----
torch.save({"model_state_dict": model.state_dict(), "config": cfg.__dict__}, f"{OUT}/bestrq_ckpt.pt")
H = logger.history
metrics = {"steps": STEPS, "n_train": len(train_pool), "params_M": count_params(model)/1e6,
           "loss_init": H[0]["loss"], "loss_final": H[-1]["loss"],
           "acc_init": H[0]["masked_acc"], "acc_final": H[-1]["masked_acc"],
           "target_ppl_final": H[-1]["target_perplexity"], "pred_ppl_final": H[-1]["pred_perplexity"],
           "probe_history": probe_hist, "probe_init": probe_hist[0][1], "probe_final": probe_hist[-1][1],
           "history": H[::20]}
json.dump(metrics, open(f"{OUT}/train_metrics.json", "w"), indent=2)

# ---- figure ----
steps = [h["step"] for h in H]
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0,0].plot(steps, [h["loss"] for h in H]); ax[0,0].axhline(np.log(8192), ls='--', c='r', label='ln(8192)=9.01 (init)')
ax[0,0].set_title("BEST-RQ masked-CE loss (interpretable)"); ax[0,0].set_xlabel("step"); ax[0,0].legend(); ax[0,0].grid(alpha=.3)
ax[0,1].plot(steps, [h["masked_acc"] for h in H], c='g'); ax[0,1].set_title("masked-frame code accuracy"); ax[0,1].grid(alpha=.3)
ps, pv = zip(*probe_hist)
ax[1,0].plot(ps, [v*100 for v in pv], 'o-', c='purple'); ax[1,0].axhline(100/12, ls='--', c='gray', label='chance')
ax[1,0].set_title("FROZEN linear probe macro-F1 (health metric)"); ax[1,0].set_xlabel("step"); ax[1,0].set_ylabel("%"); ax[1,0].legend(); ax[1,0].grid(alpha=.3)
ax[1,1].plot(steps, [h["target_perplexity"] for h in H], label='target ppl (RPQ)')
ax[1,1].plot(steps, [h["pred_perplexity"] for h in H], label='pred ppl', alpha=.7)
ax[1,1].set_title("code perplexity (collapse guard)"); ax[1,1].set_xlabel("step"); ax[1,1].legend(); ax[1,1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/bestrq_training_curve.png", dpi=130)

print("\n=== BEST-RQ TRAINING — SELF-VERIFY ===")
print(f"  loss:        {H[0]['loss']:.3f} -> {H[-1]['loss']:.3f}   (init ~ ln(8192)=9.01)")
print(f"  masked acc:  {H[0]['masked_acc']:.3f} -> {H[-1]['masked_acc']:.3f}")
print(f"  target ppl:  {H[-1]['target_perplexity']:.0f}  (high = no target collapse; codebook=8192)")
print(f"  pred ppl:    {H[-1]['pred_perplexity']:.0f}")
print(f"  PROBE macro-F1: {probe_hist[0][1]:.4f} (random) -> {probe_hist[-1][1]:.4f}  (chance {1/12:.3f})")
ok = (H[-1]['loss'] < H[0]['loss'] - 0.5) and (probe_hist[-1][1] > probe_hist[0][1] + 0.03) and (H[-1]['target_perplexity'] > 50)
print(f"  VERDICT: {'PASS — loss falls, probe rises, no collapse -> the encoder is learning' if ok else 'INSPECT — check curve'}")
