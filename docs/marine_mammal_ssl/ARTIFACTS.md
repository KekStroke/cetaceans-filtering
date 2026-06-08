# Artifacts Index — Marine Mammal SSL session (2026-06-07/08)

Everything produced this session, with paths. Autonomous run for Iaroslav's team (helping Anvar's SSL effort + the paper).

## 📄 Planning & analysis (read these)
| file | what |
|---|---|
| `plan_analysis/PLAN.md` | Master plan (critique-hardened) — all measured results live in **§0.5**; data/loss/speed/benchmark/method/PR/next-steps |
| `plan_analysis/TRAINING_SPEED.md` | **animal2vec speed** analysis (the multimask lever) + BEST-RQ comparison (~1455 clips/s) |
| `plan_analysis/SHAP_ANALYSIS.md` | SHAP consolidation + **bandwidth tie-in (attribution↔energy r=0.867)** |
| `plan_analysis/critique.json` | The adversarial review that hardened the plan |
| `plan_analysis/deck_SSL.txt`, `deck_ROADMAP.txt` | Extracted strategy decks |

## 📊 Measured results (RTX 5090)
| result | files |
|---|---|
| **Fair Table 1** — AVES 0.894 / wav2vec2 0.875 / whisper 0.875 / logmel 0.654 (per-encoder best layer) | `autoresearch-runs/probe-olga-kclass/bench_v1_layerfair_results.json` |
| BenchSuite v0 (2-task, last-layer) | `experiments_5090/bench_suite_results.json` + `bench_suite.png` |
| **Bandwidth 8/16/32 kHz** — 8→16 kHz **+10.2 pts** | `experiments_5090/exp2_bandwidth_results.json` + `.png` |
| Frozen-probe (acc-based) | `experiments_5090/exp1_frozen_probe_results.json` |
| autoresearch probe sweep + **AVES layer selection (+2.1 pt)** | `autoresearch-runs/probe-olga-kclass/RESULTS.md`, `layer_sweep_results.json`, `layer_sweep.png` |
| **SHAP↔bandwidth tie-in (r=0.867, K21)** | `plan_analysis/shap_bandwidth_tiein.{json,png}` |
| BEST-RQ training demo (loss 9.19→3.88, probe 0.642→0.695, ~1455 clips/s) | `ml-intern-runs/bestrq-marine/path-2/train_metrics.json` + `bestrq_training_curve.png` |

## 🧰 Code / tools
| file | what |
|---|---|
| `bench_suite.py` | Reusable frozen-probe benchmark harness (encoders × tasks, GroupKFold by recording) |
| `exp2_bandwidth.py`, `layer_sweep.py`, `layer_sweep_all.py` | bandwidth + per-layer studies |
| `ml-intern-runs/bestrq-marine/path-2/{bestrq.py,bestrq_train.py}` | **BEST-RQ trainer — validated, instrumented (interpretable CE loss + frozen-probe health metric). Anvar's drop-in: swap the audio loader for his corpus.** |
| `autoresearch-runs/probe-olga-kclass/` | autoresearch run (program/PLAN/BUDGET/workflow/FINDINGS) |
| `~/.claude/skills/{autoresearch,ml-intern}` | AlexWortega skills installed + used |

## 🚀 Shipped
- **PR #1** (NOAA SSL prefixes): https://github.com/KekStroke/cetaceans-filtering/pull/1

## 🔑 Key findings (honest / corrected)
1. **Data:** ~3,089 h @ 8 kHz (estimated, unaudited — table foots to 3,359 h); **quality > scale**; 16 kHz justified (+10.2 pt, and SHAP r=0.867 shows 8 kHz removes the bands the encoder *uses*).
2. **"Loss is huge" is a non-problem** — data2vec2/animal2vec MSE is uninterpretable; judge by trend + target-variance + grad-norm + frozen probe. BEST-RQ gives an interpretable CE loss (from ln(8192)=9.0).
3. **Speed:** animal2vec's #1 lever = raise the data2vec-2.0 **multimask M** (amortizes the EMA teacher). BEST-RQ has no teacher → ~2.4× + ~1455 clips/s measured.
4. **Benchmark (fair best-layer):** all SSL cluster **0.875–0.894 ≫ log-mel 0.654**; AVES's animal-domain edge is only **~0.02** on clean calls → the "domain ≫ SSL" claim was a *last-layer artifact*; the real marine advantage must be shown on the **harder field task**.
5. **BEST-RQ trainer built + validated** (3-variant ml-intern workflow → conv-subsample winner) — the concrete "speed + interpretable loss" deliverable.

## ⚠️ Ops note
One laptop reboot from running 3+ concurrent GPU jobs — all artifacts survived; thereafter GPU work was strictly **serialized**.
