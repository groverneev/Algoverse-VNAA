"""Make the separation visible on the 1B smoke set.

Projects each conversation's activation onto the difference-of-means direction
per layer, and computes a 2D PCA at a mid layer, then dumps the coordinates as
JSON so they can be plotted. This is the toy smoke set (12 easy prompts), so it
is an illustration of what separation looks like, not a result.
"""

import json

import numpy as np
from sklearn.decomposition import PCA

from vnaa.extract import load_model, extract_activations
from tests.smoke_extract import EVAL, DEPLOY, as_conv, MODEL


def main():
    lm = load_model(MODEL)
    convs = [as_conv(t) for t in EVAL] + [as_conv(t) for t in DEPLOY]
    labels = np.array([1] * len(EVAL) + [0] * len(DEPLOY))
    acts = extract_activations(lm, convs)  # (12, 17, 2048)
    n_layers_plus1 = acts.shape[1]

    # Per-layer projection onto the difference-of-means direction. Fit on all
    # points here purely to visualize the axis of separation (this is a picture,
    # not a held-out score).
    projections = []  # projections[layer] = list of 12 scalar scores
    for layer in range(n_layers_plus1):
        a = acts[:, layer, :].astype(np.float64)
        d = a[labels == 1].mean(0) - a[labels == 0].mean(0)
        n = np.linalg.norm(d)
        d = d / n if n > 0 else d
        proj = a @ d
        # z-score per layer so layers are comparable on one axis
        proj = (proj - proj.mean()) / (proj.std() + 1e-9)
        projections.append(proj.tolist())

    # 2D PCA at a middle layer to see the clusters directly.
    mid = n_layers_plus1 // 2
    coords = PCA(n_components=2).fit_transform(acts[:, mid, :].astype(np.float64))

    out = {
        "labels": labels.tolist(),
        "n_layers_plus1": n_layers_plus1,
        "projections_by_layer": projections,
        "pca_layer": mid,
        "pca_coords": coords.tolist(),
        "eval_texts": [t[:60] for t in EVAL],
        "deploy_texts": [t[:60] for t in DEPLOY],
    }
    path = "data/separation_1b.json"
    with open(path, "w") as f:
        json.dump(out, f)
    print("wrote", path)
    print("PCA layer:", mid)
    # quick text view of the mid-layer projection
    p = np.array(projections[mid])
    print(f"layer {mid} projection (z): eval mean={p[labels==1].mean():+.2f} "
          f"deploy mean={p[labels==0].mean():+.2f}")


if __name__ == "__main__":
    main()
