# BEST-RQ Smoke Test (path-2, CONV-SUBSAMPLE variant)

- **Result: PASS**
- device: `cuda`  bf16: `True`  torch: `2.9.0+cu128`

## Model

- params total: **13.9983M** (trainable 13.998M) — target [5M,15M]: `True`
- variant: 2x conv stride-2 (4x downsample, factor 4) -> 6-layer transformer (dim 384, 6 heads, ffn 1024)
- front-end: log-mel 80 mels, 16000Hz, n_fft=400, hop=160, per-utt normalized
- RPQ (frozen): proj 80->16, codebook 8192x16, seed 1234; frozen+excluded-from-params: `True`
- masking: ~8% span-starts, span 10 frames, learned mask vector; loss only at masked subsampled positions

## Checks

| check | value | pass |
|---|---|---|
| param count (M) | 13.9983 | True |
| init loss | 9.0672 | True |
| expected init loss ln(cb) | 9.0109 | (target) |
| init loss finite | True | True |
| alignment (logits/targets/mask == ceil(T/4)) | True | True |
| overfit 30 steps final loss | 2.892 | True |
| overfit ratio (final/init) | 0.319 | < 0.7 = True |
| NaN/Inf-free | True | True |

## Instrumentation (last step)

- loss=2.8920, grad_norm=1.933, pred_perplexity=47.9, masked_acc=0.728, n_masked=103
- target_perplexity at init (RPQ usage over masked positions): 78.5 (unique target codes among 103 masked positions: 85)

## Loss trajectory (overfit one batch)

```
step  0  loss=9.0589  grad_norm=2.980  pred_ppl=24.0  acc=0.000
step  1  loss=8.5877  grad_norm=2.577  pred_ppl=9.4  acc=0.019
step  2  loss=8.2982  grad_norm=2.428  pred_ppl=5.9  acc=0.049
step  3  loss=8.0510  grad_norm=2.403  pred_ppl=2.6  acc=0.126
step  4  loss=7.8353  grad_norm=2.403  pred_ppl=1.9  acc=0.146
step  5  loss=7.5910  grad_norm=2.397  pred_ppl=2.5  acc=0.184
step  6  loss=7.3604  grad_norm=2.397  pred_ppl=3.5  acc=0.233
step  7  loss=7.1104  grad_norm=2.407  pred_ppl=5.8  acc=0.301
step  8  loss=6.8899  grad_norm=2.497  pred_ppl=14.3  acc=0.447
step  9  loss=6.6408  grad_norm=2.462  pred_ppl=15.0  acc=0.456
step 10  loss=6.3975  grad_norm=2.400  pred_ppl=20.7  acc=0.495
step 11  loss=6.1760  grad_norm=2.442  pred_ppl=22.1  acc=0.495
step 12  loss=5.9590  grad_norm=2.406  pred_ppl=23.5  acc=0.524
step 13  loss=5.7265  grad_norm=2.401  pred_ppl=24.9  acc=0.534
step 14  loss=5.5111  grad_norm=2.354  pred_ppl=27.3  acc=0.544
step 15  loss=5.3152  grad_norm=2.430  pred_ppl=34.6  acc=0.592
step 16  loss=5.1016  grad_norm=2.373  pred_ppl=33.1  acc=0.612
step 17  loss=4.9058  grad_norm=2.339  pred_ppl=27.4  acc=0.583
step 18  loss=4.7049  grad_norm=2.324  pred_ppl=26.7  acc=0.553
step 19  loss=4.5181  grad_norm=2.375  pred_ppl=29.4  acc=0.602
step 20  loss=4.3292  grad_norm=2.264  pred_ppl=32.8  acc=0.592
step 21  loss=4.1546  grad_norm=2.190  pred_ppl=32.4  acc=0.602
step 22  loss=3.9785  grad_norm=2.211  pred_ppl=43.4  acc=0.680
step 23  loss=3.8111  grad_norm=2.186  pred_ppl=39.5  acc=0.680
step 24  loss=3.6385  grad_norm=2.131  pred_ppl=40.4  acc=0.680
step 25  loss=3.4876  grad_norm=2.139  pred_ppl=42.3  acc=0.699
step 26  loss=3.3360  grad_norm=2.077  pred_ppl=48.6  acc=0.748
step 27  loss=3.1695  grad_norm=2.068  pred_ppl=43.6  acc=0.738
step 28  loss=3.0250  grad_norm=1.987  pred_ppl=49.1  acc=0.757
step 29  loss=2.8920  grad_norm=1.933  pred_ppl=47.9  acc=0.728
```
