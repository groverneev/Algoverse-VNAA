"""Render the separation charts to a static SVG (phone-friendly, no canvas)."""

import json

with open("data/separation_1b.json") as f:
    D = json.load(f)

labels = D["labels"]
proj = D["projections_by_layer"]
pca = D["pca_coords"]
nL = D["n_layers_plus1"]
EVAL_C, DEP_C = "#2a78d6", "#eb6834"

W = 720
parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="720" '
    f'viewBox="0 0 {W} 720" font-family="sans-serif">',
    f'<rect width="{W}" height="720" fill="#ffffff"/>',
]

# ---- Panel 1: PCA scatter (top) ----
px0, py0, pw, ph = 70, 60, 580, 240
xs = [p[0] for p in pca]
ys = [p[1] for p in pca]
xmin, xmax = min(xs) - 0.5, max(xs) + 0.5
ymin, ymax = min(ys) - 0.5, max(ys) + 0.5


def sx(v):
    return px0 + (v - xmin) / (xmax - xmin) * pw


def sy(v):
    return py0 + ph - (v - ymin) / (ymax - ymin) * ph


parts.append(
    f'<text x="{px0}" y="40" font-size="15" fill="#333">2D PCA of activations '
    f'at layer {D["pca_layer"]} — the two classes cluster apart</text>'
)
parts.append(
    f'<rect x="{px0}" y="{py0}" width="{pw}" height="{ph}" fill="none" '
    f'stroke="#e1e0d9"/>'
)
for i, (x, y) in enumerate(pca):
    c = EVAL_C if labels[i] == 1 else DEP_C
    parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="7" fill="{c}"/>')
parts.append(
    f'<text x="{px0}" y="{py0+ph+22}" font-size="12" fill="#898781">PC1 →</text>'
)

# ---- Legend ----
ly = py0 + ph + 46
parts.append(f'<circle cx="{px0+8}" cy="{ly}" r="7" fill="{EVAL_C}"/>')
parts.append(
    f'<text x="{px0+22}" y="{ly+4}" font-size="13" fill="#333">evaluation '
    f'(benchmark-style)</text>'
)
parts.append(f'<circle cx="{px0+250}" cy="{ly}" r="7" fill="{DEP_C}"/>')
parts.append(
    f'<text x="{px0+264}" y="{ly+4}" font-size="13" fill="#333">deployment '
    f'(casual chat)</text>'
)

# ---- Panel 2: per-layer projection (bottom) ----
qx0, qy0, qw, qh = 70, 430, 580, 230
pmin, pmax = -1.8, 1.4
parts.append(
    f'<text x="{qx0}" y="410" font-size="15" fill="#333">Projection onto probe '
    f'direction by layer — classes split after layer 0</text>'
)
parts.append(
    f'<rect x="{qx0}" y="{qy0}" width="{qw}" height="{qh}" fill="none" '
    f'stroke="#e1e0d9"/>'
)
# zero line
zy = qy0 + qh - (0 - pmin) / (pmax - pmin) * qh
parts.append(
    f'<line x1="{qx0}" y1="{zy:.1f}" x2="{qx0+qw}" y2="{zy:.1f}" '
    f'stroke="#c3c2b7" stroke-dasharray="3,3"/>'
)
for L in range(nL):
    lx = qx0 + (L + 0.5) / nL * qw
    if L % 2 == 0:
        parts.append(
            f'<text x="{lx:.1f}" y="{qy0+qh+18}" font-size="11" '
            f'fill="#898781" text-anchor="middle">{L}</text>'
        )
    for i in range(12):
        jitter = (i % 6 - 2.5) * 1.2
        cx = lx + jitter
        val = proj[L][i]
        cy = qy0 + qh - (val - pmin) / (pmax - pmin) * qh
        c = EVAL_C if labels[i] == 1 else DEP_C
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="{c}"/>')
parts.append(
    f'<text x="{qx0}" y="{qy0+qh+36}" font-size="12" fill="#898781">'
    f'layer (0 = embeddings) →</text>'
)

parts.append("</svg>")
with open("data/separation_1b.svg", "w") as f:
    f.write("\n".join(parts))
print("wrote data/separation_1b.svg")
