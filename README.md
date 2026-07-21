# Probing Evaluation Awareness in Realistic Multi-Turn Conversations

Do language models internally recognize when they are being evaluated, and does
that recognition survive when the evaluation is disguised as a realistic
conversation? We train linear probes on a model's residual-stream activations to
separate evaluation contexts from deployment contexts, and test whether the
signal holds up once we remove the surface cues (formatting, length, single-turn
structure) that a probe could read instead of genuine recognition.

Authors: Neev Grover, Apurva Reddy, Vedaswaroop Kunamneni, Arya Kumar, Maheep
Chaudhary.

## Status

Slice 1 (pipeline validation) is built and validated on a small model. The
end-to-end path — load model, extract activations, train a per-layer
difference-of-means probe, read out AUROC — works on real data. The headline
replication number (on Llama-3.1-8B) has not been run yet. See `CLAUDE.md` for
the full state.

## Approach in one paragraph

We build the thinnest end-to-end pipeline first (one model, one probe, obvious
evaluation vs deployment prompts) and verify it reproduces prior results, then
widen it one step at a time toward the hard case: multi-turn, format-matched,
silence-filtered conversations. Each step is mostly a dataset swap on the same
probe code.

## Repository layout

```
src/vnaa/
  probe.py        per-layer difference-of-means probe + AUROC readout
  extract.py      residual-stream activation extraction
  data.py         SAD stages_oversight loader (evaluation vs deployment)
  run_slice1.py   slice-1 runner: data -> extract -> probe
tests/
  test_probe.py   probe math checked against known-answer synthetic data
  ...             model-dependent smoke/visualization scripts
```

`data/` and `results/` are gitignored (datasets and cached activations are
large and reproducible).

## Setup

Requires Python 3.11–3.13 (not 3.14; PyTorch does not support it yet).

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

Model access: Llama models are gated on Hugging Face. Request access to
`meta-llama/Llama-3.2-1B-Instruct` and `meta-llama/Llama-3.1-8B-Instruct`, then
log in with a token that has the "read access to public gated repos" permission:

```bash
.venv/bin/hf auth login
```

Data: slice 1 uses the SAD `stages_oversight` task. Download the (encrypted)
oversight structs once:

```bash
mkdir -p data/sad
curl -sL -o data/sad/oversight_structs.zip \
  https://github.com/LRudL/sad/raw/main/sad/stages/oversight/structs.zip
```

Note: SAD question text must never be committed anywhere scrapable (it is a
benchmark; leaking it invalidates it). The loader reads it from the encrypted
zip in memory and never writes prompt text to disk. Do not commit SAD prompts.

## Running slice 1

```bash
# local sanity check on the small model
.venv/bin/python -m vnaa.run_slice1 \
  --sad-repo data/sad/oversight_structs.zip \
  --model meta-llama/Llama-3.2-1B-Instruct

# the real replication (needs a ~24 GB GPU)
.venv/bin/python -m vnaa.run_slice1 \
  --sad-repo data/sad/oversight_structs.zip \
  --model meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16
```

Output: a per-layer AUROC table (evaluation vs deployment), plus cached
activations and the AUROC curve saved under `results/`.

## Tests

```bash
.venv/bin/python -m pytest
```

## Compute

White-box probing needs the model's internal activations, which no hosted API
returns — so this work runs on self-hosted open-weight models on a GPU, not
through API credits. Slice 1 is a few hundred forward passes with no generation,
so it fits comfortably on a free GPU tier (Kaggle, Lightning AI). See `CLAUDE.md`
for details and the program compute policy.
