"""Probe math checked against planted-answer synthetic data.

No model involved. These pin down that the difference-of-means direction and the
per-layer AUROC readout behave correctly on data whose answer we control, so
that when real activations arrive the only unknown is the extraction loop.
"""

import numpy as np
import pytest

from vnaa.probe import fit_layer, fit_score


def make_data(rng, n_per_class, n_layers, hidden_dim, sep_by_layer):
    """Two Gaussian blobs per layer, separated by sep_by_layer[l].

    Returns (acts, labels) with acts shaped (2*n_per_class, n_layers, hidden).
    Class 1 is shifted by +sep on class-1 samples, spread evenly across all
    hidden dims so the total shift has L2 norm sep. Spreading the signal (rather
    than dumping it on one axis) matters: it forces the probe to aggregate
    across dims like it must on real activations, and it means a random or
    mis-fit direction genuinely lands at chance instead of accidentally
    catching one dominant axis. A larger sep means an easier layer.
    """
    n = 2 * n_per_class
    labels = np.array([0] * n_per_class + [1] * n_per_class)
    acts = rng.standard_normal((n, n_layers, hidden_dim))
    per_dim = 1.0 / np.sqrt(hidden_dim)
    for layer, sep in enumerate(sep_by_layer):
        acts[n_per_class:, layer, :] += sep * per_dim
    return acts, labels


def test_planted_separation_recovers_high_auroc():
    rng = np.random.default_rng(0)
    # Layer 2 is the easy one; the flanks are pure noise.
    sep = [0.0, 0.0, 6.0, 0.0]
    tr_a, tr_y = make_data(rng, 100, 4, 8, sep)
    te_a, te_y = make_data(rng, 100, 4, 8, sep)

    report = fit_score(tr_a, tr_y, te_a, te_y)

    assert report.best_layer == 2
    assert report.best_auroc > 0.95
    # Noise layers stay near chance.
    for layer in (0, 1, 3):
        assert report.strength[layer] < 0.15


def test_shuffled_labels_give_chance_in_expectation():
    # A direction fit on shuffled train labels must not separate held-out data.
    # This holds only in expectation: a single permutation leaves the classes
    # slightly imbalanced across the shuffled groups, and against strong signal
    # that residual can push one draw's AUROC well away from 0.5 (with random
    # sign). So we average strength over many shuffles and check the mean sits
    # at chance, which is the honest form of the claim.
    rng = np.random.default_rng(1)
    tr_a, tr_y = make_data(rng, 150, 1, 256, [6.0])
    te_a, te_y = make_data(rng, 150, 1, 256, [6.0])

    honest = fit_score(tr_a, tr_y, te_a, te_y)
    assert honest.best_auroc > 0.95  # signal is genuinely there

    # Check the signed AUROC, not strength. strength = |AUROC - 0.5| is folded,
    # so its mean is positive even under a symmetric null. Per-draw leakage is
    # sign-symmetric, so the signed AUROC is what averages to chance.
    aurocs = []
    for _ in range(50):
        shuffled_y = tr_y.copy()
        rng.shuffle(shuffled_y)
        aurocs.append(fit_score(tr_a, shuffled_y, te_a, te_y).best_auroc)

    assert abs(np.mean(aurocs) - 0.5) < 0.05  # no separation on average


def test_direction_points_toward_positive_class():
    rng = np.random.default_rng(2)
    acts = rng.standard_normal((200, 16))
    labels = np.array([0] * 100 + [1] * 100)
    acts[100:, 0] += 5.0  # class 1 is more positive on feature 0

    probe = fit_layer(acts, labels)

    # Projections of class 1 should exceed class 0 on average.
    proj = probe.project(acts)
    assert proj[labels == 1].mean() > proj[labels == 0].mean()
    assert np.isclose(np.linalg.norm(probe.direction), 1.0)


def test_sign_below_half_under_flipped_test_distribution():
    # Difference-of-means always orients toward the positive class on its own
    # train split, so same-distribution AUROC is never below 0.5. Raw AUROC
    # drops below 0.5 only under distribution shift: fit on train, then test on
    # data where the class relationship is reversed. Strength stays high and
    # best_sign reports the flip. This is why we compare |AUROC - 0.5|, not the
    # raw value, across conditions.
    rng = np.random.default_rng(3)
    tr_a, tr_y = make_data(rng, 120, 1, 32, [6.0])
    # Test set with the shift attached to the opposite class.
    te_a, te_y = make_data(rng, 120, 1, 32, [6.0])
    te_y = 1 - te_y

    report = fit_score(tr_a, tr_y, te_a, te_y)

    assert report.best_auroc < 0.5
    assert report.best_strength > 0.4
    assert report.best_sign == -1


def test_coincident_means_are_safe():
    # Degenerate direction must not crash; it yields chance AUROC.
    acts = np.ones((10, 4))
    labels = np.array([0] * 5 + [1] * 5)
    probe = fit_layer(acts, labels)
    assert np.allclose(probe.direction, 0.0)


def test_rejects_single_class_test_split():
    rng = np.random.default_rng(4)
    tr_a, tr_y = make_data(rng, 10, 2, 8, [3.0, 3.0])
    te_a = rng.standard_normal((10, 2, 8))
    te_y = np.ones(10)  # only one class
    with pytest.raises(ValueError):
        fit_score(tr_a, tr_y, te_a, te_y)
