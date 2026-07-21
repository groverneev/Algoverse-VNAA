"""Residual-stream activation extraction.

Slice 1 read path. Given conversations (lists of chat messages), run each
through an open-weight model with output_hidden_states=True and cache the
residual stream at every layer at one token position.

Decisions baked in here, each a place slice 1 could go quietly wrong:

- BOS: apply_chat_template already prepends exactly one BOS. We feed its ids
  straight to the model and never re-tokenize, so there is never a double BOS.

- Layer axis: hidden_states has n_layers + 1 entries. Index 0 is the embedding
  output (before any transformer block); indices 1..n_layers are block outputs.
  We keep all of them and record that index 0 is embeddings. Separation that
  is already present at layer 0 is lexical/format signal in the input, not
  something the model computes, which is exactly what the format-sensitivity
  question cares about.

- Read point: "last_user_token" templates with add_generation_prompt=False and
  reads the final token, which is the <|eot_id|> closing the last user turn.
  This is the literal "last token of the final user turn." The read point is a
  parameter because the multi-turn arm adds more of them.

- Batch size 1: we process one conversation at a time so the read index is
  always the true last token with no padding to reason about. Slice 1 is a few
  hundred short forward passes with no generation, so this is fast enough.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

Conversation = list[dict[str, str]]  # [{"role": "user"|"assistant"|"system", "content": ...}]

READ_POINTS = ("last_user_token",)


def pick_device(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class LoadedModel:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    device: str
    n_layers: int  # transformer blocks, not counting the embedding layer
    hidden_size: int


def load_model(
    model_name: str,
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
    device_map: str | None = None,
) -> LoadedModel:
    """Load tokenizer and model for activation extraction.

    dtype defaults to float32 for faithful activations on CPU/MPS. On a CUDA box
    running the 8B, pass torch.bfloat16 to fit memory; that is a deliberate
    fidelity/size tradeoff, not the default.

    device_map (e.g. "auto") shards the model across multiple GPUs via accelerate.
    Needed when the model does not fit on one card — e.g. the 8B in bf16 (~16 GB)
    across Kaggle's 2x T4 (16 GB each). When set, we do not call .to(); accelerate
    places the layers, and inputs go to the embedding layer's device. When None,
    the model loads on a single device as usual.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map=device_map
        )
        model.eval()
        # Inputs must land on whatever device holds the embedding layer.
        input_device = str(model.get_input_embeddings().weight.device)
        return LoadedModel(
            tokenizer=tokenizer,
            model=model,
            device=input_device,
            n_layers=model.config.num_hidden_layers,
            hidden_size=model.config.hidden_size,
        )

    device = pick_device(device)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()
    return LoadedModel(
        tokenizer=tokenizer,
        model=model,
        device=device,
        n_layers=model.config.num_hidden_layers,
        hidden_size=model.config.hidden_size,
    )


def _read_index(read_point: str, seq_len: int) -> int:
    """Token index to read, given the templated sequence length.

    For last_user_token the sequence is templated with add_generation_prompt
    False, so the final token is the user turn's closing delimiter.
    """
    if read_point == "last_user_token":
        return seq_len - 1
    raise ValueError(f"unknown read_point {read_point!r}; known: {READ_POINTS}")


@torch.no_grad()
def extract_activations(
    lm: LoadedModel,
    conversations: list[Conversation],
    read_point: str = "last_user_token",
    show_progress: bool = True,
) -> np.ndarray:
    """Extract residual-stream activations at one token position, every layer.

    Returns an array of shape (n_conversations, n_layers + 1, hidden_size),
    float32 on CPU. Axis 1 index 0 is the embedding layer; indices 1..n_layers
    are transformer block outputs.
    """
    if read_point not in READ_POINTS:
        raise ValueError(f"unknown read_point {read_point!r}; known: {READ_POINTS}")

    add_gen_prompt = False  # last_user_token: stop at the user turn's eot
    out = np.empty(
        (len(conversations), lm.n_layers + 1, lm.hidden_size), dtype=np.float32
    )
    iterator = tqdm(conversations, disable=not show_progress, desc="extract")
    for i, conv in enumerate(iterator):
        enc = lm.tokenizer.apply_chat_template(
            conv,
            add_generation_prompt=add_gen_prompt,
            return_dict=True,
            return_tensors="pt",
        )
        enc = {k: v.to(lm.device) for k, v in enc.items()}
        seq_len = enc["input_ids"].shape[1]
        idx = _read_index(read_point, seq_len)

        result = lm.model(**enc, output_hidden_states=True)
        # hidden_states: tuple of (n_layers + 1) tensors, each (1, seq, hidden).
        # Stack the read-token vector across layers -> (n_layers + 1, hidden).
        vecs = [hs[0, idx, :].float().cpu().numpy() for hs in result.hidden_states]
        out[i] = np.stack(vecs, axis=0)
    return out
