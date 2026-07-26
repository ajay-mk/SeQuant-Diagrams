"""Render a Brandow diagram from the extractor's JSON."""
import json
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# the interaction is the top vertex in a CC diagram (Shavitt & Bartlett p.295);
# in UCC a de-excitation amplitude sits above it again.
LEVEL = {"ampl": 0, "eri": 1, "fock": 1, "deexc": 2}

_POINT_GAP = 0.7     # spacing between interaction points on one vertex
_BAR_OVERHANG = 0.25  # how far a T-amplitude bar runs past its outermost point
_FOCK_STUB = 0.55    # length of the one-particle vertex's dashed tail
_STUB = 0.6          # length of an external line's free end
_BOW = 0.22          # arc curvature that opens a hole/particle pair into a loop
_HEAD = 0.01         # half-length of the stub carrying the mid-line arrowhead
_LABEL_OFF = 0.17    # perpendicular offset of a line's index label

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

def _indices(labels):
    return r"\,".join(labels)

def _vertex_tex(v):
    """The factor a vertex contributes, per Fig. 10.1 rules 2-4."""
    if v["kind"] == "eri":
        return r"\langle %s||%s\rangle" % (_indices(v["bra"]), _indices(v["ket"]))
    if v["kind"] == "fock":
        return "f_{%s}" % _indices(v["bra"] + v["ket"])
    sym = r"{t^{\dagger}}" if v["kind"] == "deexc" else "t"
    return "%s^{%s}_{%s}" % (sym, _indices(v["bra"]), _indices(v["ket"]))

def _prefactor_tex(diagram):
    """SeQuant renders a Constant wrapped in three brace levels; unwrap it and
    drop a bare 1, which carries no information."""
    p = diagram.get("prefactor", "")
    if p.startswith("{{{") and p.endswith("}}}"):
        p = p[3:-3]
    return "" if p in ("1", "") else p

def term_expression(diagram):
    """The algebraic term the diagram stands for, as a mathtext string.

    Rule 5 sums over the internal line labels only; external lines carry the
    target indices of an amplitude equation and are not summed.
    """
    internal = [l for l in diagram["lines"] if not l["external"]]
    summed = ([l["index"] for l in internal if l["type"] == "hole"] +
              [l["index"] for l in internal if l["type"] == "particle"])
    parts = [_prefactor_tex(diagram)]
    if summed:
        parts.append(r"\sum_{%s}" % _indices(summed))
    parts += [_vertex_tex(v) for v in diagram["vertices"]]
    # thin spaces, not concatenation: "\rangle" abutting "t" parses as "\ranglet"
    return "$" + r"\,".join(p for p in parts if p) + "$"

def count_loops(diagram):
    """Number of loops, or None if the diagram is open.

    A terminal is one (vertex, slot, pos). Two perfect matchings live on them:
    the lines themselves, and the within-vertex pairing that joins slot position
    k of the bra to position k of the ket (Fig. 10.1 rule 3's left-out/left-in).
    The union of two perfect matchings is a disjoint set of alternating cycles,
    and those cycles are the loops.
    """
    if any(l["external"] for l in diagram["lines"]):
        return None  # rule 8's quasiloops for paired external lines: not handled
    line_end = {}
    for line in diagram["lines"]:
        (a, b) = [(e["vertex"], e["slot"], e["pos"]) for e in line["endpoints"]]
        line_end[a], line_end[b] = b, a
    partner = lambda t: (t[0], "ket" if t[1] == "bra" else "bra", t[2])

    seen, loops = set(), 0
    for start in line_end:
        if start in seen:
            continue
        loops += 1
        t = start
        while t not in seen:
            seen.add(t)
            other = line_end[t]
            seen.add(other)
            t = partner(other)
    return loops

def diagram_sign(diagram):
    """Fig. 10.1 rule 8: -1^(h-l) from hole lines and loops. None if open."""
    loops = count_loops(diagram)
    if loops is None:
        return None
    holes = sum(1 for l in diagram["lines"] if l["type"] == "hole")
    return -1 if (holes - loops) % 2 else 1

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

def _draw_line(ax, src, dst, rad):
    """Draw one directed line, arrowhead at its midpoint. Returns that midpoint
    and the unit tangent there.

    matplotlib's arc3 is a quadratic Bezier whose control point sits at the
    chord midpoint displaced by rad*(dy, -dx), so the curve midpoint is
    M + rad/2*(dy, -dx) and the tangent at t=0.5 is just the chord direction.
    """
    ax.annotate("", xy=dst, xytext=src,
                arrowprops=dict(arrowstyle="-", color="black", lw=1.2,
                                shrinkA=0, shrinkB=0,
                                connectionstyle=f"arc3,rad={rad}"))
    (x1, y1), (x2, y2) = src, dst
    dx, dy = x2 - x1, y2 - y1
    mx, my = (x1 + x2) / 2 + rad * dy / 2, (y1 + y2) / 2 - rad * dx / 2
    norm = math.hypot(dx, dy) or 1.0
    ux, uy = dx / norm, dy / norm
    ax.annotate("", xy=(mx + _HEAD * ux, my + _HEAD * uy),
                xytext=(mx - _HEAD * ux, my - _HEAD * uy),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2,
                                shrinkA=0, shrinkB=0, mutation_scale=14))
    return (mx, my), (ux, uy)

def _draw(ax, diagram, fontsize=13):
    positions = assign_positions(diagram)
    pts = {v["id"]: _points(diagram, v["id"], positions)
           for v in diagram["vertices"]}

    right = max(_draw_vertex(ax, v, pts[v["id"]]) for v in diagram["vertices"])

    all_x = [x for vp in pts.values() for x, _ in vp]
    centre = (min(all_x) + max(all_x)) / 2
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
        (mx, my), (ux, uy) = _draw_line(ax, src, dst, rad)
        # rule 1: label the line just outside its own arc. arc3 bulges toward
        # rad*(dy, -dx), so that direction is always the outside of a loop; a
        # straight line has no bulge, so push it away from the diagram's axis.
        if rad:
            px, py = math.copysign(1, rad) * uy, -math.copysign(1, rad) * ux
        else:
            px, py = -uy, ux
            if (mx - centre) * px < 0:
                px, py = -px, -py
        ax.text(mx + _LABEL_OFF * px, my + _LABEL_OFF * py,
                "$%s$" % line["index"], ha="center", va="center",
                fontsize=fontsize - 3)

    # annotate() arrows don't feed the autoscaler, so external stubs and bows
    # would be clipped away; size the axes from the interaction points instead.
    xs = [x for vp in pts.values() for x, _ in vp]
    ys = [y for vp in pts.values() for _, y in vp]
    ax.set_xlim(min(xs) - _BAR_OVERHANG - 0.4, right + 0.4)
    ax.set_ylim(min(ys) - _STUB - 0.3, max(ys) + _STUB + 0.3)

    # the interpretation goes under the diagram, as in Shavitt & Bartlett p.297
    ax.text(0.5, -0.04, term_expression(diagram), transform=ax.transAxes,
            ha="center", va="top", fontsize=fontsize)
    loops = count_loops(diagram)
    if loops is not None:
        holes = sum(1 for l in diagram["lines"] if l["type"] == "hole")
        sign = "+" if diagram_sign(diagram) > 0 else "-"
        ax.text(0.5, -0.17, f"$h={holes},\\ l={loops}\\ \\Rightarrow\\ {sign}$",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=fontsize - 4, color="0.45")
    ax.set_aspect("equal"); ax.axis("off")

def render(diagram, out_path):
    fig, ax = plt.subplots(figsize=(4, 4))
    _draw(ax, diagram)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def render_grid(diagrams, out_path, ncols=4):
    """One panel per term, for a whole Sum."""
    nrows = -(-len(diagrams) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.6 * nrows),
                             squeeze=False)
    flat = [a for row in axes for a in row]
    for ax, d in zip(flat, diagrams):
        _draw(ax, d, fontsize=8)
    for ax in flat[len(diagrams):]:
        ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    src = sys.stdin if sys.argv[1] == "-" else open(sys.argv[1])
    with src as f:
        loaded = json.load(f)
    if isinstance(loaded, list):
        render_grid(loaded, sys.argv[2])
    else:
        render(loaded, sys.argv[2])
