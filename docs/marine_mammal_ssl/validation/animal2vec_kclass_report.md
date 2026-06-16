# Validation — animal2vec (ckpt 25k) on olga_kclass

- embeddings: 1800 × 1024, 12 classes

## Classification (frozen linear probe, recording-disjoint CV)
- **macro-F1 = 0.2506**  (95% CI [0.231, 0.269]), acc 25.1%
- vs bars: log-mel 0.654 · AVES (best-layer) 0.894 · wav2vec2 0.875
- **verdict: BELOW log-mel (encoder weak)**

## Clustering / representation quality
- k-NN purity (k=10): 0.143
- silhouette: -0.0671597346663475
- KMeans NMI: 0.043 · ARI: 0.010

![t-SNE](tsne.png)

![per-class](per_class_f1.png)
