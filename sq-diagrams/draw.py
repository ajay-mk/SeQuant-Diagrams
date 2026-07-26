"""Render a Brandow diagram from the extractor's JSON."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LEVEL = {"eri": 1, "fock": 1, "ampl": 0}

_POINT_GAP = 0.7     # spacing between interaction points on one vertex
_BAR_OVERHANG = 0.25  # how far a T-amplitude bar runs past its outermost point
_FOCK_STUB = 0.55    # length of the one-particle vertex's dashed tail
_STUB = 0.6          # length of an external line's free end
_BOW = 0.22          # arc curvature that opens a hole/particle pair into a loop

def assign_positions(diagram):
    """Vertex id -> (x, y). Fixed levels; vertices spread evenly per level."""
    pos = {}
    by_level = {}
    for v in diagram["vertices"]:
        by_level.setdefault(LEVEL[v["kind"]], []).append(v["id"])
    for level, ids in by_level.items():
        # centre each level on x=0, else a lone vertex sits above the leftmost
        # of the level below instead of between them (g over two t1 vertices)
        for k, vid in enumerate(ids):
            pos[vid] = (2.0 * k - (len(ids) - 1), float(level))
    return pos

def line_direction(line):
    """Particle lines point up, hole lines point down (Goldstone convention)."""
    return "up" if line["type"] == "particle" else "down"

def _slot_count(diagram, vid):
    v = diagram["vertices"][vid]
    return max(len(v["bra"]), len(v["ket"]), 1)

def _anchor(vid, pos, ncols, positions):
    """The (x, y) of one interaction point on a vertex.

    Every line in slot position `pos` attaches here, hole and particle alike:
    Shavitt & Bartlett Fig. 10.1 rules 2-3 put both ends of a pair on a shared
    point, which is what makes each loop close into the familiar lens rather
    than a pair of parallel rails.
    """
    cx, cy = positions[vid]
    return cx + (pos - (ncols - 1) / 2) * _POINT_GAP, cy

def _points(diagram, vid, positions):
    n = _slot_count(diagram, vid)
    return [_anchor(vid, k, n, positions) for k in range(n)]

def _draw_vertex(ax, v, pts):
    """One vertex glyph, per Fig. 10.1: `>--x` for f, `>--<` for g, bar for T."""
    (x0, y), (x1, _) = pts[0], pts[-1]
    if v["kind"] == "fock":
        # rule 2: one-particle vertex is a dashed stub ending in a cross
        ax.plot([x0, x0 + _FOCK_STUB], [y, y], "--", color="black", lw=1.2)
        ax.plot([x0 + _FOCK_STUB], [y], marker="x", color="black", ms=7, mew=1.5)
        label_x = x0 + _FOCK_STUB + 0.12
    elif v["kind"] == "eri":
        # rule 3: two-particle vertex spans exactly between its two points
        ax.plot([x0, x1], [y, y], "--", color="black", lw=1.2)
        ax.plot([x0, x1], [y, y], "o", color="black", ms=3.5)
        # above the line, not beside it: loops leave the end point going outward
        ax.text(x1 + 0.06, y + 0.1, v["label"], ha="left", va="bottom",
                fontsize=12, style="italic")
        return x1 + 0.3
    else:
        ax.plot([x0 - _BAR_OVERHANG, x1 + _BAR_OVERHANG], [y, y],
                "-", color="black", lw=2.5)
        label_x = x1 + _BAR_OVERHANG + 0.12
    ax.text(label_x, y, v["label"], ha="left", va="center",
            fontsize=12, style="italic")
    return label_x

def _bows(diagram):
    """Line index -> arc curvature, oriented bottom-to-top. A hole/particle pair
    sharing both endpoints bows apart into a loop; anything else stays straight.

    Curvature is stored for the upward orientation and negated at draw time for
    downward lines, since arc3's rad is relative to the direction of travel: a
    pair drawn in opposite directions with opposite rad lands on the same side.
    """
    groups = {}
    for i, line in enumerate(diagram["lines"]):
        if line["external"]:
            continue
        key = frozenset((e["vertex"], e["pos"]) for e in line["endpoints"])
        groups.setdefault(key, []).append(i)
    bows = {}
    for members in groups.values():
        if len(members) == 2:
            for sign, i in zip((1, -1), members):
                bows[i] = sign * _BOW
    return bows

def render(diagram, out_path):
    positions = assign_positions(diagram)
    pts = {v["id"]: _points(diagram, v["id"], positions)
           for v in diagram["vertices"]}
    fig, ax = plt.subplots(figsize=(4, 4))

    right = max(_draw_vertex(ax, v, pts[v["id"]]) for v in diagram["vertices"])

    bows = _bows(diagram)
    for i, line in enumerate(diagram["lines"]):
        up = line_direction(line) == "up"
        eps = line["endpoints"]
        a = pts[eps[0]["vertex"]][eps[0]["pos"]]
        if line["external"]:
            b = (a[0], a[1] + (_STUB if up else -_STUB))
        else:
            b = pts[eps[1]["vertex"]][eps[1]["pos"]]
        # arrow points "up" for particle, "down" for hole
        lo, hi = (a, b) if a[1] <= b[1] else (b, a)
        src, dst = (lo, hi) if up else (hi, lo)
        rad = bows.get(i, 0.0) if up else -bows.get(i, 0.0)
        ax.annotate("", xy=dst, xytext=src,
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2,
                                    shrinkA=0, shrinkB=0,
                                    connectionstyle=f"arc3,rad={rad}"))

    # annotate() arrows don't feed the autoscaler, so external stubs and bows
    # would be clipped away; size the axes from the interaction points instead.
    xs = [x for vp in pts.values() for x, _ in vp]
    ys = [y for vp in pts.values() for _, y in vp]
    ax.set_xlim(min(xs) - _BAR_OVERHANG - 0.4, right + 0.4)
    ax.set_ylim(min(ys) - _STUB - 0.3, max(ys) + _STUB + 0.3)

    ax.set_title(f"${diagram.get('prefactor', '')}$", fontsize=12)
    ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    src = sys.stdin if sys.argv[1] == "-" else open(sys.argv[1])
    with src as f:
        render(json.load(f), sys.argv[2])
