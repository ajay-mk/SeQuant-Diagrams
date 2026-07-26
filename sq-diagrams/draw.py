"""Render a Brandow diagram from the extractor's JSON."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LEVEL = {"eri": 1, "fock": 1, "ampl": 0}

_VERTEX_HALFWIDTH = 0.7
_STUB = 0.6
_SLOT_SPLIT = 0.12   # hole/particle offset either side of an interaction point

def assign_positions(diagram):
    """Vertex id -> (x, y). Fixed levels; vertices spread evenly per level."""
    pos = {}
    by_level = {}
    for v in diagram["vertices"]:
        by_level.setdefault(LEVEL[v["kind"]], []).append(v["id"])
    for level, ids in by_level.items():
        for k, vid in enumerate(ids):
            pos[vid] = (2.0 * k, float(level))
    return pos

def line_direction(line):
    """Particle lines point up, hole lines point down (Goldstone convention)."""
    return "up" if line["type"] == "particle" else "down"

def _anchor(vid, ltype, pos, ncols, positions):
    """A slot's (x, y): spread a vertex's legs across its horizontal bar.

    The k-th bra and k-th ket share one interaction point, so hole lines are
    nudged left of it and particle lines right. Offsetting by line type rather
    than by bra/ket keeps both ends of a line on the same side, which is what
    makes an f-t1 loop draw as two parallel verticals instead of a crossing.
    """
    cx, cy = positions[vid]
    span = 2 * _VERTEX_HALFWIDTH
    x = cx - _VERTEX_HALFWIDTH + (span * (pos + 0.5) / max(ncols, 1))
    return x + (-_SLOT_SPLIT if ltype == "hole" else _SLOT_SPLIT), cy

def _slot_count(diagram, vid):
    v = diagram["vertices"][vid]
    return max(len(v["bra"]), len(v["ket"]), 1)

def render(diagram, out_path):
    positions = assign_positions(diagram)
    fig, ax = plt.subplots(figsize=(4, 4))

    # vertices: dashed bar for interactions (eri/fock), heavy solid for amplitudes
    for v in diagram["vertices"]:
        cx, cy = positions[v["id"]]
        style = "--" if v["kind"] in ("eri", "fock") else "-"
        lw = 1.5 if v["kind"] in ("eri", "fock") else 3.0
        ax.plot([cx - _VERTEX_HALFWIDTH, cx + _VERTEX_HALFWIDTH], [cy, cy],
                style, color="black", lw=lw)
        ax.text(cx, cy + 0.12, v["label"], ha="center", fontsize=12)

    # lines
    for line in diagram["lines"]:
        up = line_direction(line) == "up"
        eps = line["endpoints"]
        a = _anchor(eps[0]["vertex"], line["type"], eps[0]["pos"],
                    _slot_count(diagram, eps[0]["vertex"]), positions)
        if line["external"]:
            b = (a[0], a[1] + (_STUB if up else -_STUB))
        else:
            b = _anchor(eps[1]["vertex"], line["type"], eps[1]["pos"],
                        _slot_count(diagram, eps[1]["vertex"]), positions)
        # arrow points "up" for particle, "down" for hole
        lo, hi = (a, b) if a[1] <= b[1] else (b, a)
        src, dst = (lo, hi) if up else (hi, lo)
        ax.annotate("", xy=dst, xytext=src,
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2))

    # annotate() arrows don't feed the autoscaler, so external stubs would be
    # clipped away; size the axes from the vertex positions plus a stub.
    xs = [x for x, _ in positions.values()]
    ys = [y for _, y in positions.values()]
    ax.set_xlim(min(xs) - _VERTEX_HALFWIDTH - 0.3, max(xs) + _VERTEX_HALFWIDTH + 0.3)
    ax.set_ylim(min(ys) - _STUB - 0.3, max(ys) + _STUB + 0.3)

    ax.set_title(f"${diagram.get('prefactor', '')}$", fontsize=12)
    ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    src = sys.stdin if sys.argv[1] == "-" else open(sys.argv[1])
    with src as f:
        render(json.load(f), sys.argv[2])
