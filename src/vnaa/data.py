"""Slice 1 dataset: SAD stages_oversight (evaluation vs deployment).

The SAD stages_oversight task already pairs both classes we need:

- test_oversight: real evaluation/benchmark prompts (label 1, "eval")
- deploy_oversight: real deployment prompts sampled from WildChat (label 0)

We read the raw `body` of each sample and treat it as a single-turn user
message. We deliberately do NOT apply SAD's own "is this test or deployment?"
multiple-choice wrapper: for probing we feed the model the underlying prompt and
read its activations, using SAD's stage label as ground truth. This matches how
prior eval-awareness probes use stages_oversight, and it means slice 1 is
literally "SAD-eval vs WildChat-deploy," pre-assembled.

Canary discipline (required by SAD's license): SAD question text must never be
written to a scrapable location, including our public repo. This loader reads
the prompts directly from the encrypted zip in memory and returns them as Python
objects. It never writes prompt text to disk. Do not commit anything derived
from these prompts in plain text; cached *activations* are fine (they are not
the question text).
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import yaml

# Public canary password from the SAD repo's unzip.sh. Not a secret; the zip
# exists only to keep the questions out of training-data scrapes.
SAD_ZIP_PASSWORD = b"sadtimesforthesetimes"

# Path within the SAD repo to the oversight structs zip and the two batches.
_OVERSIGHT_STRUCTS = "sad/stages/oversight/structs.zip"
_EVAL_BATCH = "batch/test_oversight.yaml"
_DEPLOY_BATCH = "batch/deploy_oversight.yaml"

Conversation = list[dict[str, str]]


@dataclass
class Slice1Data:
    """Assembled slice-1 dataset, held in memory only.

    conversations[i] is a single-turn [{"role": "user", "content": ...}].
    labels[i] is 1 for evaluation, 0 for deployment.
    sources[i] is SAD's provenance tag (e.g. "test_anthropic", "wildchat_sampled").
    """

    conversations: list[Conversation]
    labels: list[int]
    sources: list[str]

    def __len__(self) -> int:
        return len(self.labels)

    def counts(self) -> dict[str, int]:
        n_eval = sum(self.labels)
        return {"eval": n_eval, "deploy": len(self.labels) - n_eval}


def _read_batch_from_zip(structs_zip: Path, member: str) -> list[dict]:
    """Read one batch YAML out of the encrypted structs zip, in memory."""
    with zipfile.ZipFile(structs_zip) as z:
        raw = z.read(member, pwd=SAD_ZIP_PASSWORD)
    doc = yaml.safe_load(io.BytesIO(raw))
    return doc["samples"]


def _resolve_structs_zip(sad_path: Path) -> Path:
    """Accept either a SAD repo root or a direct path to the oversight zip.

    A direct path to the (encrypted) structs.zip is the durable option: it is
    only ~1.3 MB and survives without a full 1.1 GB clone. Passing the repo root
    also works for a full checkout.
    """
    if sad_path.is_file() and sad_path.suffix == ".zip":
        return sad_path
    return sad_path / _OVERSIGHT_STRUCTS


def load_stages_oversight(
    sad_repo: str | Path,
    n_per_class: int | None = None,
    seed: int = 0,
) -> Slice1Data:
    """Load SAD stages_oversight as single-turn eval/deploy conversations.

    sad_repo: path to a cloned LRudL/sad repo root, OR a direct path to the
        oversight structs.zip (the durable ~1.3 MB option).
    n_per_class: cap samples per class (balanced). None uses all available.
    seed: RNG seed for the per-class subsample when capping.

    Returns a balanced-as-possible Slice1Data. Raises if the SAD structs zip is
    not found, so a missing checkout fails loudly rather than silently.
    """
    import random

    structs_zip = _resolve_structs_zip(Path(sad_repo))
    if not structs_zip.exists():
        raise FileNotFoundError(
            f"SAD oversight structs not found at {structs_zip}. Clone "
            "https://github.com/LRudL/sad and pass its root, or pass a direct "
            "path to sad/stages/oversight/structs.zip."
        )

    eval_samples = _read_batch_from_zip(structs_zip, _EVAL_BATCH)
    deploy_samples = _read_batch_from_zip(structs_zip, _DEPLOY_BATCH)

    rng = random.Random(seed)

    def take(samples: list[dict]) -> list[dict]:
        if n_per_class is not None and n_per_class < len(samples):
            return rng.sample(samples, n_per_class)
        return samples

    conversations: list[Conversation] = []
    labels: list[int] = []
    sources: list[str] = []
    for label, samples in ((1, take(eval_samples)), (0, take(deploy_samples))):
        for s in samples:
            body = s["body"]
            conversations.append([{"role": "user", "content": body}])
            labels.append(label)
            sources.append(s.get("splits", {}).get("source", "unknown"))

    return Slice1Data(conversations=conversations, labels=labels, sources=sources)
