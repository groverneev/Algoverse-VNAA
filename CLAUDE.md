# CLAUDE.md — agent onboarding and running log

This file is the detailed context for any agent picking up this project. Keep it
updated as work progresses (see "Maintaining this file" at the bottom). The
README is the short human-facing version; this is the long one.

Last updated: July 2026. Slice 1 validated on Llama-3.2-1B; 8B run prepared for
Kaggle (multi-GPU), not yet done.

GPU host decision: tried Lightning AI first, abandoned it — the site 404'd on the
Studio dashboard and the account showed 0.00 free credits (a `.org` school email,
not `.edu`, likely got no free grant). Pivoted to **Kaggle** (2x T4, genuinely
free, 30 GPU-hr/week). The 8B in bf16 (~16 GB) does not fit on one 16 GB T4, so it
shards across both T4s with `--device-map auto` (needs `accelerate`). See
`scripts/kaggle_slice1.md` for the exact notebook cells. `scripts/setup_gpu.sh`
is the single-GPU path (kept for a future single-24GB-card host).

---

## The idea

AI safety evaluations are only trustworthy if models behave the same under test
as in real use. Prior work shows models can tell evaluation from deployment,
both behaviorally and from internal activations (Needham et al. 2025, Nguyen et
al. 2025, Chaudhary et al. 2025). Nguyen argues a linear probe on the residual
stream captures genuine eval-awareness; Devbunova (2026) counters that the probe
just tracks surface format (benchmark-canonical structure) and fails to
generalize to free-form prompts. Both sides are single-turn, so they cannot
separate genuine recognition from surface-format detection.

Our contribution: test white-box probing in the realistic, multi-turn,
format-matched, silence-filtered regime — the axis deployment-simulation sets
aside. If the probe stays well above a hierarchy of surface baselines when the
evaluation is disguised as a real conversation, that is genuine recognition. If
it collapses to baseline, realism is an effective defense and prior "awareness"
was largely format detection. The gap between the probe and the baselines is the
result, not the raw AUROC.

Full proposal: `Probing Evaluation Awareness in Realistic Multi-Turn
Conversations` (the source PDF, not in the repo). Authors: Neev Grover, Apurva
Reddy, Vedaswaroop Kunamneni, Arya Kumar, Maheep Chaudhary.

## Build strategy: thinnest slice first, then widen

Build the minimal end-to-end path that produces one honest AUROC number, verify
it, then expand by swapping datasets on the same machinery. Do NOT build the
whole apparatus from the proposal at once.

Expansion order (each is mostly a dataset swap on the same probe code):
1. **Slice 1 (current):** single-turn, obvious eval (SAD) vs deployment
   (WildChat). Pipeline validation / replication, not a hypothesis test.
2. Format-matched single-turn pairs.
3. Multi-turn (adds four token read-points: last user token, post-instruction
   token, mean-pool over final turn, per-turn trajectory).
4. The decisive cell: multi-turn + format-matched + silence-filtered.
5. Deployment-simulation reconstruction (resample final assistant turn; model-
   swap vs eval-context ablation).
6. Silence-filtering internality test (awareness judge; unaware subset; probe-
   vs-verbalization detectability).
7. Steering arm (last).

Success criterion for slice 1: high `|AUROC - 0.5|` concentrated in early-to-
middle layers on the obvious-eval cell.

## How the user (Neev) likes to work

- One slice at a time. Finish and verify before moving on. Do not jump ahead.
- Ground claims in a concrete run before generalizing. A small real number beats
  a big abstract framework.
- Correct him directly when he is wrong; he prefers that to hedging.
- Prose: brief, one job per sentence, minimal em dashes, minimal fancy language.
- State explicitly which slice we are on and what the success criterion is.
- **Do not add a `Co-Authored-By: Claude` trailer to commits** (or any self-
  attribution). Plain commit messages only.

## Environment

- Machine: Apple M3, 16 GB unified memory. macOS.
- Python: system Python is 3.14 (too new for PyTorch). Use the `.venv` (Python
  3.12) created with `uv`. Run everything as `.venv/bin/python`.
- Key package versions: torch 2.13 (MPS works, no CUDA locally), transformers
  5.x (major version; API differs from the 4.x most probing code targets —
  first suspect for weird API errors; `output_hidden_states` still works;
  loading uses `dtype=` not the deprecated `torch_dtype=`).
- **The 8B does not fit locally.** Llama-3.1-8B in bf16 is ~16 GB of weights on
  a 16 GB machine. 4-bit quantization would fit but perturbs the activations we
  measure, defeating a replication. Debug locally on the 1B; run the 8B on a
  rented/free GPU.

## Model access (Hugging Face)

- Llama is gated. Account `LightningPro1` is approved for Llama-3.2-1B; 1B and
  8B access were both requested.
- Gotcha that cost time: a fine-grained token without the "read access to public
  gated repos" scope gives 403 even after approval. Use a token with gated-repo
  read access (`hf auth login`). The metadata endpoint returning 200 is not
  proof of access; file access is.

## Compute policy (Algoverse program) — the important constraints

- **APIs cannot serve this project.** White-box probing needs hidden states; no
  hosted endpoint (Azure, Cerebras, NVIDIA Build, OpenRouter) returns them. We
  are locked to self-hosted open weights on a GPU. Tell teammates this — "just
  use the Azure credits for the 8B" does not work.
- **Slice 1 is free.** ~200 forward passes, no generation = minutes of GPU. The
  policy says validate on free T4 first; only pay when you have measured you need
  more. Do not rent AWS or file the reimbursement form for slice 1.
- The policy explicitly blesses this: scenario "train a linear probe on hidden
  states → T4 is usually enough. Reimbursable."
- **Hardware wrinkle for the 8B:** 8B bf16 is ~16 GB, and a free T4 is exactly
  16 GB (too tight with activations). Use a free 24 GB-class card: Kaggle
  (2×T4 = 32 GB, 30 hr/week) or Lightning AI (L4 24 GB). Avoid 4-bit here — it
  perturbs the activations.
- Paid GPU is for training/fine-tuning, NOT inference. If we ever self-host a
  large model on paid GPU (e.g. the 70B scaling arm), pre-approve on the form.

## Data

Slice 1 uses SAD `stages_oversight`, which already pairs both classes:
- `test_oversight`: evaluation prompts (label 1). Sources: `test_anthropic`,
  `test_bigbench`, `test_wmdp` (~67 each, 200 total).
- `deploy_oversight`: deployment prompts (label 0), all `wildchat_sampled`
  (200). So SAD's deployment side is already WildChat — slice 1 is literally
  "SAD-eval vs WildChat-deploy," pre-assembled, matching prior work.

We use the raw `body` of each sample as a single-turn user message. We do NOT
apply SAD's own "is this test or deployment?" MCQ wrapper — for probing we feed
the underlying prompt and read activations, using the stage as ground truth.

**SAD canary discipline (required by SAD's license):** SAD question text must
never appear in plain text anywhere scrapable, including this public repo. The
loader (`data.py`) reads prompts from the encrypted `structs.zip` in memory
(public password `sadtimesforthesetimes`) and never writes prompt text to disk.
Cached *activations* are fine (not the question text). Never commit SAD prompts,
and never paste them into files that get committed.

The encrypted zip lives at `data/sad/oversight_structs.zip` (gitignored, ~1.3
MB). Re-download with the curl command in the README if missing. It survives
teardowns; the full 1.1 GB SAD clone in scratchpad does not.

## Code and key technical decisions

- `probe.py` — per-layer difference-of-means. Direction = normalized
  mean(eval) − mean(deploy), fit on train only. Score = projection onto
  direction; report AUROC per layer and the sign-invariant `|AUROC − 0.5|`
  (raw AUROC can drop below 0.5 under distribution shift). Verified against
  known-answer synthetic data in `test_probe.py` (6 tests). Two lessons baked
  in: a shuffled-label null must be averaged over many shuffles (one draw leaks
  residual signal against strong separation), and separation must be spread
  across dims to be realistic.
- `extract.py` — loads model, runs each conversation with
  `output_hidden_states=True`, reads one token position across every layer.
  Decisions: (a) the chat template already adds exactly one BOS — we feed its
  ids straight in, never re-tokenize; (b) `hidden_states` has n_layers + 1
  entries — index 0 is the embedding layer, 1..n are transformer blocks; we keep
  all and record that index 0 is embeddings; (c) read point `last_user_token`
  templates with `add_generation_prompt=False` and reads the final token (the
  `<|eot_id|>` closing the user turn) — this is a swappable parameter because the
  multi-turn arm adds more read points; (d) batch size 1 so the read index is
  always the true last token with no padding.
- `data.py` — SAD loader (above). Accepts either a SAD repo root or a direct
  path to `structs.zip`.
- `run_slice1.py` — ties it together: load data, extract, stratified train/test
  split, `fit_score`, print per-layer AUROC, save activations + AUROC curve to
  `results/` (never prompt text).

Note on the read point and layer 0: because the read token is the same
`<|eot_id|>` for every prompt, the embedding layer (index 0) carries no per-
example info, so its AUROC is exactly 0.5. Separation appears only after block 1
once attention has aggregated the turn. This is expected and a good sanity check;
it does NOT rule out surface confounds (length/format still get aggregated into
that position).

## Results so far

**Slice 1 on Llama-3.2-1B (local, MPS), 200 eval / 200 deploy, stratified 280
train / 120 test:**
- Layer 0 (embeddings): AUROC 0.500 (as expected).
- Layers 1–16: 0.91 → 0.99. Best layer 5 of 16, AUROC 0.992.
- Shape matches the replication prediction: early-to-middle peak, high and stable
  after.
- **Surface-baseline check (length):** on the same test split, token-length
  AUROC = 0.37, char-length = 0.39 (eval prompts are shorter: median 57 vs 101
  tokens). The probe's 0.99 is NOT a length artifact — probe strength 0.49 vs
  length strength ~0.13.

Honest scope: this is the 1B, not the 8B replication. Length is only one
baseline; format (MCQ structure) and lexical (bag-of-words) confounds are not
ruled out yet. So this is "pipeline validated, replication shape reproduced, not
a length artifact" — NOT yet "genuine eval-awareness."

## What's next

1. **Run slice 1 on Llama-3.1-8B** (bf16) on a free 24 GB GPU. This is the
   headline replication number. 8B has 32 layers, so "early-to-middle" is a more
   meaningful claim there. Chosen host: Lightning AI single L4 (24 GB) — one card
   fits 8B bf16, so our single-device code runs unchanged; Kaggle's 2×T4 would
   force multi-GPU sharding. Setup is scripted: `scripts/setup_gpu.sh` (clone the
   public repo on a Studio, run it, switch to the L4, run `vnaa.run_slice1`).
   `setup_gpu.sh` installs on top of the Studio's CUDA torch (never reinstalls
   torch). PyYAML is now an explicit dep (data.py needs it; was transitive).
2. Add the rest of the surface-baseline hierarchy: formatting classifier,
   bag-of-words, style-matched control (random and length already understood).
   The probe-minus-baseline gap is the actual result.
3. Then begin the expansion arms in order (format-matched, then multi-turn).

## Gotchas / hard-won notes

- transformers 5.x: use `dtype=`, not `torch_dtype=`. `apply_chat_template(...,
  return_dict=True)` returns a `BatchEncoding`, not a bare tensor.
- The scratchpad directory and full SAD clone get wiped on session teardown.
  Durable state lives in the repo's gitignored `data/` (the encrypted SAD zip)
  and `results/` (activations).
- Do not trust HF metadata 200s for gated access; test file access.
- Model load of the 1B takes ~1–2.5 min the first time (download + load).

## Maintaining this file

Update CLAUDE.md and README.md as part of finishing each unit of work — new
results, new decisions, changed status, new gotchas. CLAUDE.md gets the details
and reasoning; README.md gets the short human summary. Never put SAD prompt text
in either (canary rule). Keep the "Last updated" line and "Results so far"
current.
