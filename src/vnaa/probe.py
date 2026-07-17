"""Per-layer difference-of-means probe with AUROC readout.

Slice 1 estimator. Given residual-stream activations captured at one token
position across every layer, fit a difference-of-means direction per layer on a
train split and score a held-out split by projection.

Conventions
-----------
- Activations are shaped (n_samples, n_layers, hidden_dim).
- Labels are binary {0, 1}. Class 1 is the "evaluation" (positive) class.
- The direction is fit on train data only. Scoring never sees train labels.
- AUROC is reported per layer. The sign-invariant strength |AUROC - 0.5|
  is what we compare across layers, since a difference-of-means direction can
  point either way and only the separation matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class LayerProbe:
    """A fitted difference-of-means direction for a single layer.

    direction is unit-normalized and points from the negative-class mean toward
    the positive-class mean, so a larger projection means more eval-like.
    """

    direction: np.ndarray  # (hidden_dim,), unit norm

    def project(self, acts: np.ndarray) -> np.ndarray:
        """Project activations (n_samples, hidden_dim) onto the direction."""
        return acts @ self.direction


def fit_layer(acts: np.ndarray, labels: np.ndarray) -> LayerProbe:
    """Difference-of-means direction for one layer.

    acts: (n_samples, hidden_dim) train activations for a single layer.
    labels: (n_samples,) binary.

    The direction is mean(class 1) - mean(class 0), normalized to unit length.
    If the two class means coincide the direction is undefined; we return a zero
    vector, which yields constant-zero projections and AUROC 0.5.
    """
    acts = np.asarray(acts, dtype=np.float64)
    labels = np.asarray(labels)
    pos = acts[labels == 1]
    neg = acts[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("fit_layer needs at least one sample of each class")
    diff = pos.mean(axis=0) - neg.mean(axis=0)
    norm = np.linalg.norm(diff)
    if norm == 0:
        return LayerProbe(direction=np.zeros_like(diff))
    return LayerProbe(direction=diff / norm)


@dataclass
class ProbeReport:
    """Per-layer scores for a fitted probe stack.

    auroc[i] and strength[i] correspond to layer i. best_layer is the index of
    maximum strength; sign is +1 if the raw AUROC there is >= 0.5, else -1.
    """

    auroc: np.ndarray  # (n_layers,)
    strength: np.ndarray  # (n_layers,) == |auroc - 0.5|
    best_layer: int
    best_auroc: float
    best_strength: float
    best_sign: int

    def summary(self) -> str:
        lines = [f"{'layer':>5}  {'auroc':>7}  {'|auroc-.5|':>10}"]
        for i, (a, s) in enumerate(zip(self.auroc, self.strength)):
            mark = "  <- best" if i == self.best_layer else ""
            lines.append(f"{i:>5}  {a:>7.4f}  {s:>10.4f}{mark}")
        return "\n".join(lines)


def fit_score(
    acts_train: np.ndarray,
    labels_train: np.ndarray,
    acts_test: np.ndarray,
    labels_test: np.ndarray,
) -> ProbeReport:
    """Fit a per-layer probe on train, score AUROC per layer on test.

    acts_train / acts_test: (n_samples, n_layers, hidden_dim).
    labels_*: (n_samples,) binary.

    For each layer independently: fit the difference-of-means direction on the
    train split, project the test split onto it, and compute AUROC against the
    test labels. Returns a ProbeReport whose best layer maximizes |AUROC - 0.5|.
    """
    acts_train = np.asarray(acts_train, dtype=np.float64)
    acts_test = np.asarray(acts_test, dtype=np.float64)
    labels_train = np.asarray(labels_train)
    labels_test = np.asarray(labels_test)

    if acts_train.ndim != 3 or acts_test.ndim != 3:
        raise ValueError("activations must be (n_samples, n_layers, hidden_dim)")
    n_layers = acts_train.shape[1]
    if acts_test.shape[1] != n_layers:
        raise ValueError("train and test have different layer counts")
    if len(np.unique(labels_test)) < 2:
        raise ValueError("AUROC needs both classes present in the test split")

    aurocs = np.empty(n_layers, dtype=np.float64)
    for layer in range(n_layers):
        probe = fit_layer(acts_train[:, layer, :], labels_train)
        scores = probe.project(acts_test[:, layer, :])
        aurocs[layer] = roc_auc_score(labels_test, scores)

    strength = np.abs(aurocs - 0.5)
    best = int(np.argmax(strength))
    return ProbeReport(
        auroc=aurocs,
        strength=strength,
        best_layer=best,
        best_auroc=float(aurocs[best]),
        best_strength=float(strength[best]),
        best_sign=1 if aurocs[best] >= 0.5 else -1,
    )
