"""Slice 1 run: SAD stages_oversight, per-layer difference-of-means AUROC.

Ties data -> extraction -> probe. Loads SAD stages_oversight (eval vs deploy),
extracts residual-stream activations at the last user token across every layer,
fits the difference-of-means probe on a stratified train split, and reports
per-layer AUROC and |AUROC - 0.5| on the held-out test split.

Success criterion (replication): high |AUROC - 0.5| concentrated in early-to-
middle layers. This validates the pipeline against a known result before the
harder disguised conditions.

Outputs (to results/, gitignored): a .npz of activations + labels + sources and
a .json of the per-layer AUROC curve. Prompt text is never written out (SAD
canary discipline); only activations and labels are cached.

Example:
    python -m vnaa.run_slice1 --sad-repo /path/to/sad --model meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from vnaa.data import load_stages_oversight
from vnaa.extract import extract_activations, load_model
from vnaa.probe import fit_score


def main() -> None:
    ap = argparse.ArgumentParser(description="Slice 1: SAD stages_oversight probe")
    ap.add_argument("--sad-repo", required=True, help="path to cloned LRudL/sad repo")
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--n-per-class", type=int, default=None, help="cap per class")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (auto if unset)")
    ap.add_argument("--device-map", default=None,
                    help="shard across GPUs, e.g. 'auto' (needed for 8B on 2x T4)")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    import torch

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]

    data = load_stages_oversight(args.sad_repo, n_per_class=args.n_per_class, seed=args.seed)
    labels = np.array(data.labels)
    print(f"loaded {len(data)} conversations | counts: {data.counts()}")

    lm = load_model(args.model, device=args.device, dtype=dtype,
                    device_map=args.device_map)
    print(f"model={args.model} device={lm.device} dtype={args.dtype} "
          f"n_layers={lm.n_layers} hidden={lm.hidden_size}")
    if lm.device == "cpu" and lm.n_layers > 20:
        print("WARNING: running a large model on CPU will be very slow. "
              "Switch the machine to a GPU (e.g. Lightning L4) before this run.")

    acts = extract_activations(lm, data.conversations)  # (n, n_layers+1, hidden)
    assert np.isfinite(acts).all(), "non-finite activations"

    # Stratified split so both classes appear in train and test.
    idx = np.arange(len(labels))
    tr, te = train_test_split(
        idx, test_size=args.test_frac, random_state=args.seed, stratify=labels
    )
    report = fit_score(acts[tr], labels[tr], acts[te], labels[te])

    print(f"\ntrain n={len(tr)}  test n={len(te)}")
    print(report.summary())
    print(
        f"\nbest layer {report.best_layer} of 0..{lm.n_layers} "
        f"(0=embeddings)  auroc={report.best_auroc:.4f}  "
        f"strength={report.best_strength:.4f}"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.model.split("/")[-1]
    np.savez_compressed(
        out_dir / f"slice1_{tag}_acts.npz",
        acts=acts, labels=labels, sources=np.array(data.sources),
        train_idx=tr, test_idx=te,
    )
    with open(out_dir / f"slice1_{tag}_auroc.json", "w") as f:
        json.dump(
            {
                "model": args.model,
                "n_per_class": data.counts(),
                "auroc_by_layer": report.auroc.tolist(),
                "strength_by_layer": report.strength.tolist(),
                "best_layer": report.best_layer,
                "best_auroc": report.best_auroc,
                "best_strength": report.best_strength,
                "note": "layer 0 = embeddings; 1..n = transformer blocks",
            },
            f, indent=2,
        )
    print(f"\nsaved activations and AUROC curve to {out_dir}/slice1_{tag}_*")


if __name__ == "__main__":
    main()
