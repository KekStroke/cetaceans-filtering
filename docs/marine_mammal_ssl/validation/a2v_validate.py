#!/usr/bin/env python3
"""
Validation harness for a frozen encoder's embeddings (modern env: torch2/sklearn).
Decoupled from the extractor — operates on a precomputed embeddings array + labels.

Does, for one task (Watkins / K-class / field / filtration):
  - CLASSIFICATION: frozen LogReg probe, recording-disjoint GroupKFold, macro-F1 + per-class F1
    + acc + bootstrap 95% CI  (vs bars AVES 0.894 / log-mel 0.654).
  - CLUSTERING / representation quality: k-NN label purity, silhouette, KMeans NMI/ARI vs labels,
    + a 2-D t-SNE figure coloured by class.
Outputs: <out>/metrics.json, <out>/report.md, <out>/tsne.png, <out>/per_class_f1.png

Input: --npz with keys X[,y,groups,classes]  OR  --emb X.npy --labels y.npy [--groups g.npy] [--classes-file]
"""
import os, json, argparse, numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BARS = {"log-mel": 0.654, "AVES (best-layer)": 0.894, "wav2vec2": 0.875}


def knn_purity(X, y, k=10):
    n = min(k + 1, len(X))
    nn = NearestNeighbors(n_neighbors=n).fit(X)
    _, idx = nn.kneighbors(X)
    pur = [(y[idx[i, 1:]] == y[i]).mean() for i in range(len(X))]
    return float(np.mean(pur))


def probe(X, y, groups, classes):
    nsp = min(5, len(set(groups))) if groups is not None else 5
    splitter = GroupKFold(nsp) if groups is not None else StratifiedKFold(nsp, shuffle=True, random_state=0)
    splits = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    T, P = [], []
    for tr, te in splits:
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", n_jobs=-1).fit(sc.transform(X[tr]), y[tr])
        P.append(clf.predict(sc.transform(X[te]))); T.append(y[te])
    T, P = np.concatenate(T), np.concatenate(P)
    perc = f1_score(T, P, average=None, labels=range(len(classes)))
    # bootstrap CI on macro-F1
    rng = np.random.RandomState(0); bs = []
    for _ in range(1000):
        s = rng.choice(len(T), len(T), replace=True); bs.append(f1_score(T[s], P[s], average="macro"))
    return {"macro_f1": float(f1_score(T, P, average="macro")), "acc": float(accuracy_score(T, P)),
            "macro_f1_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "per_class_f1": {classes[i]: float(perc[i]) for i in range(len(classes))},
            "per_class_support": {classes[i]: int((T == i).sum()) for i in range(len(classes))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz"); ap.add_argument("--emb"); ap.add_argument("--labels"); ap.add_argument("--groups")
    ap.add_argument("--classes-file"); ap.add_argument("--name", default="encoder"); ap.add_argument("--task", default="task")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    if a.npz:
        d = np.load(a.npz, allow_pickle=True); X = d["X"]
        ylab = d["y"] if "y" in d else None; groups = d["groups"] if "groups" in d else None
        classes = list(d["classes"]) if "classes" in d else None
    else:
        X = np.load(a.emb); ylab = np.load(a.labels, allow_pickle=True)
        groups = np.load(a.groups, allow_pickle=True) if a.groups else None
        classes = [l.strip() for l in open(a.classes_file)] if a.classes_file else None
    X = np.asarray(X, dtype=np.float32)
    # encode labels (strings or ints)
    if ylab.dtype.kind in "US" or classes is None:
        le = LabelEncoder(); y = le.fit_transform(ylab); classes = list(le.classes_) if classes is None else classes
    else:
        y = np.asarray(ylab, dtype=int)
    print(f"[{a.name}/{a.task}] X={X.shape} classes={len(classes)} groups={'yes' if groups is not None else 'no'}", flush=True)

    res = {"name": a.name, "task": a.task, "n": int(len(X)), "dim": int(X.shape[1]), "n_classes": len(classes)}
    # classification
    res["classification"] = probe(X, y, groups, classes)
    # clustering / representation
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=len(classes), n_init=10, random_state=0).fit_predict(Xs)
    res["clustering"] = {"knn_purity_k10": knn_purity(Xs, y, 10),
                         "silhouette": float(silhouette_score(Xs, y)) if len(set(y)) > 1 else None,
                         "kmeans_nmi": float(normalized_mutual_info_score(y, km)),
                         "kmeans_ari": float(adjusted_rand_score(y, km))}
    json.dump(res, open(f"{a.out}/metrics.json", "w"), indent=2)

    # figures: t-SNE + per-class F1
    try:
        from sklearn.manifold import TSNE
        sub = np.random.RandomState(0).choice(len(X), min(3000, len(X)), replace=False)
        emb2 = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(Xs[sub])
        plt.figure(figsize=(7, 6))
        for ci in range(len(classes)):
            m = y[sub] == ci
            if m.any(): plt.scatter(emb2[m, 0], emb2[m, 1], s=6, label=classes[ci])
        plt.legend(fontsize=6, markerscale=2, ncol=2); plt.title(f"{a.name} — {a.task} t-SNE (knn-purity {res['clustering']['knn_purity_k10']:.2f})")
        plt.tight_layout(); plt.savefig(f"{a.out}/tsne.png", dpi=120); plt.close()
    except Exception as e:
        print("t-SNE skipped:", e)
    pf = res["classification"]["per_class_f1"]; order = sorted(pf, key=lambda c: -pf[c])
    plt.figure(figsize=(8, max(3, .35 * len(order)))); plt.barh(order, [pf[c] for c in order]); plt.gca().invert_yaxis()
    plt.xlabel("per-class F1"); plt.title(f"{a.name} — {a.task} macro-F1 {res['classification']['macro_f1']:.3f}")
    plt.tight_layout(); plt.savefig(f"{a.out}/per_class_f1.png", dpi=120); plt.close()

    # report.md
    c = res["classification"]; cl = res["clustering"]
    with open(f"{a.out}/report.md", "w") as f:
        f.write(f"# Validation — {a.name} on {a.task}\n\n")
        f.write(f"- embeddings: {res['n']} × {res['dim']}, {res['n_classes']} classes\n\n")
        f.write(f"## Classification (frozen linear probe, {'recording-disjoint' if groups is not None else 'stratified'} CV)\n")
        f.write(f"- **macro-F1 = {c['macro_f1']:.4f}**  (95% CI [{c['macro_f1_ci95'][0]:.3f}, {c['macro_f1_ci95'][1]:.3f}]), acc {c['acc']*100:.1f}%\n")
        f.write("- vs bars: " + " · ".join(f"{k} {v}" for k, v in BARS.items()) + "\n")
        verdict = "BELOW log-mel (encoder weak)" if c["macro_f1"] < BARS["log-mel"] else ("competitive (≈ generic SSL)" if c["macro_f1"] < BARS["AVES (best-layer)"] else "BEATS AVES — headline")
        f.write(f"- **verdict: {verdict}**\n\n")
        f.write("## Clustering / representation quality\n")
        f.write(f"- k-NN purity (k=10): {cl['knn_purity_k10']:.3f}\n- silhouette: {cl['silhouette']}\n- KMeans NMI: {cl['kmeans_nmi']:.3f} · ARI: {cl['kmeans_ari']:.3f}\n\n")
        f.write("![t-SNE](tsne.png)\n\n![per-class](per_class_f1.png)\n")
    print(f"  macro-F1={c['macro_f1']:.4f} CI{c['macro_f1_ci95']} | knn-purity {cl['knn_purity_k10']:.3f} NMI {cl['kmeans_nmi']:.3f}")
    print(f"  saved {a.out}/report.md")


if __name__ == "__main__":
    main()
