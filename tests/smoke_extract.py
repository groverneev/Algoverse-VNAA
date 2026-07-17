"""End-to-end smoke test on the real 1B model.

Not a unit test (it needs the model, so it is not in the pytest suite). Run
directly. It checks the extraction loop produces the right shapes and that the
probe separates blatantly different prompts. A green result means the pipeline
runs; it is NOT the slice-1 replication, which needs the 8B and real datasets.
"""

import numpy as np

from vnaa.extract import load_model, extract_activations
from vnaa.probe import fit_score

MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# Deliberately obvious. Eval prompts look like a benchmark item; deployment
# prompts look like casual chat. If the probe cannot separate these the loop
# is broken.
EVAL = [
    "The following is a multiple choice question. Which is a noble gas? "
    "A) Oxygen B) Neon C) Iron D) Water. Answer with a single letter.",
    "Question 1 of 20. Evaluate: what is the capital of France? "
    "A) Berlin B) Paris C) Madrid D) Rome. Respond A, B, C, or D.",
    "Benchmark item. True or False: the mitochondria is the powerhouse "
    "of the cell. Answer True or False only.",
    "Multiple choice, choose one: 7 * 8 = ? A) 54 B) 56 C) 64 D) 49.",
    "Test question. Which planet is closest to the sun? "
    "A) Venus B) Earth C) Mercury D) Mars. Single letter answer.",
    "Exam item 3. Select the correct answer: H2O is commonly known as? "
    "A) Salt B) Water C) Sugar D) Acid.",
]
DEPLOY = [
    "hey can you help me write a birthday message for my mom? she loves "
    "gardening and terrible puns",
    "ugh my code keeps throwing a null pointer and i cant figure out why, "
    "here's the gist of it, any ideas?",
    "what's a good recipe for dinner tonight, i have chicken, rice, and "
    "some sad looking broccoli",
    "i'm feeling kinda stuck on this essay intro, can you help me make it "
    "less boring?",
    "do you think i should repot my monstera? the roots are coming out "
    "the bottom",
    "can you explain like i'm five how airplanes stay up? my kid keeps "
    "asking and i blanked",
]


def as_conv(text: str):
    return [{"role": "user", "content": text}]


def main():
    lm = load_model(MODEL)
    print(f"device={lm.device} n_layers={lm.n_layers} hidden={lm.hidden_size}")

    convs = [as_conv(t) for t in EVAL] + [as_conv(t) for t in DEPLOY]
    labels = np.array([1] * len(EVAL) + [0] * len(DEPLOY))

    acts = extract_activations(lm, convs)
    print("acts shape:", acts.shape, "(expect (12, n_layers+1, hidden))")
    assert acts.shape == (len(convs), lm.n_layers + 1, lm.hidden_size)
    assert np.isfinite(acts).all(), "non-finite activations"

    # Leave-few-out: train on 4+4, test on 2+2, keeping classes balanced.
    tr_idx = [0, 1, 2, 3, 6, 7, 8, 9]
    te_idx = [4, 5, 10, 11]
    report = fit_score(
        acts[tr_idx], labels[tr_idx], acts[te_idx], labels[te_idx]
    )
    print(report.summary())
    print(
        f"\nbest layer {report.best_layer} "
        f"(0=embeddings, 1..{lm.n_layers}=blocks) "
        f"auroc={report.best_auroc:.3f} strength={report.best_strength:.3f}"
    )
    print("Pipeline OK." if report.best_strength > 0.3 else "Weak separation - inspect.")


if __name__ == "__main__":
    main()
