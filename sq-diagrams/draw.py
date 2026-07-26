"""Render a Brandow diagram from the extractor's JSON."""
import itertools
import json
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# the interaction is the top vertex in a CC diagram (Shavitt & Bartlett p.295);
# in UCC a de-excitation amplitude sits above it again.
LEVEL = {"ampl": 0, "eri": 1, "fock": 1, "deexc": 2}

_POINT_GAP = 0.95    # spacing between interaction points on one vertex; wide
                     # enough that two adjacent loops' inner labels stay apart
_BAR_OVERHANG = 0.25  # how far a T-amplitude bar runs past its outermost point
_FOCK_STUB = 0.55    # length of the one-particle vertex's dashed tail
_STUB = 0.6          # length of an external line's free end
_BOW = 0.22          # arc curvature that opens a hole/particle pair into a loop
_HEAD = 0.01         # half-length of the stub carrying the mid-line arrowhead
_LABEL_OFF = 0.17    # perpendicular offset of a line's index label
_CLEARANCE = 0.12    # how wide of a vertex glyph a passing line must stay
_LABEL_ZONE = 0.38   # room reserved for a vertex's label past its glyph
_LABEL_CLEAR = 0.24  # radius a line label wants clear of lines and other labels
# candidate vertex offsets, in half-point-gap steps so the search keeps the same
# freedom relative to vertex width whenever _POINT_GAP changes
# 11 is where this stops paying: 9 shifts leave 28 crossings over the UCCSD BCH2
# file, 11 leave 26, and 13 also leave 26 for 3.5x the search time
_X_SHIFTS = tuple(k * _POINT_GAP / 2 for k in range(-5, 6))

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

_TARGET = -1   # virtual vertex standing for the projection onto the target manifold

def count_loops(diagram):
    """Number of loops (quasiloops included), or None if they cannot be closed.

    A terminal is one (vertex, slot, pos). Two perfect matchings live on them:
    the lines themselves, and the within-vertex pairing that joins slot position
    k of the bra to position k of the ket (Fig. 10.1 rule 3's left-out/left-in).
    The union of two perfect matchings is a disjoint set of alternating cycles,
    and those cycles are the loops.
    """
    targets = diagram.get("targets") or {"bra": [], "ket": []}
    external = [l for l in diagram["lines"] if l["external"]]
    if external and not (targets["bra"] or targets["ket"]):
        return None  # open, but nothing says how its free ends pair up

    line_end = {}
    for line in diagram["lines"]:
        ends = [(e["vertex"], e["slot"], e["pos"]) for e in line["endpoints"]]
        if line["external"]:
            # rule 8: paired external lines close through imaginary extensions
            # into quasiloops. Slot k of the target bra pairs with slot k of the
            # target ket, exactly as within a vertex, so treat the projection as
            # one more vertex (id -1).
            for slot in ("bra", "ket"):
                if line["index"] in targets[slot]:
                    ends.append((_TARGET, slot, targets[slot].index(line["index"])))
                    break
            else:
                return None  # a free end that is not a target index
        a, b = ends
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

def _glyph_span(diagram, vid, pts):
    """The x-interval a vertex occupies, label included -- a line struck through
    a vertex label is as bad as one struck through its bar."""
    xs = [p[0] for p in pts[vid]]
    x0, x1 = min(xs), max(xs)
    kind = diagram["vertices"][vid]["kind"]
    if kind == "fock":
        return x0, x0 + _FOCK_STUB + _LABEL_ZONE
    if kind == "eri":
        return x0, x1 + _LABEL_ZONE
    return x0 - _BAR_OVERHANG, x1 + _BAR_OVERHANG + _LABEL_ZONE

def _target_level(pts):
    return max(p[1] for vp in pts.values() for p in vp) + _STUB

def _endpoints_xy(diagram, line, pts):
    e = line["endpoints"]
    a = pts[e[0]["vertex"]][e[0]["pos"]]
    if line["external"]:
        # external lines carry the target indices and run to the top of the
        # diagram, whichever way their arrow points (cf. Fig. 10.2)
        return a, (a[0], _target_level(pts))
    return a, pts[e[1]["vertex"]][e[1]["pos"]]

def _crosses(p, q, r, s):
    """True if open segments pq and rs properly intersect."""
    def side(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1, d2 = side(r, s, p), side(r, s, q)
    d3, d4 = side(p, q, r), side(p, q, s)
    return (d1 * d2 < 0) and (d3 * d4 < 0)

def penalties(diagram, pts):
    """(lines struck through a vertex, line-line crossings, total length)."""
    segs = [_endpoints_xy(diagram, l, pts) for l in diagram["lines"]]

    overlaps = 0
    for line, (a, b) in zip(diagram["lines"], segs):
        touched = {e["vertex"] for e in line["endpoints"]}
        for v in diagram["vertices"]:
            vid = v["id"]
            if vid in touched:
                continue
            vy = pts[vid][0][1]
            if (a[1] - vy) * (b[1] - vy) >= 0:
                continue           # does not cross this vertex's level
            t = (vy - a[1]) / (b[1] - a[1])
            x = a[0] + t * (b[0] - a[0])
            lo, hi = _glyph_span(diagram, vid, pts)
            if lo - _CLEARANCE < x < hi + _CLEARANCE:
                overlaps += 1

    # two vertices sharing a level must not overlap, or one's label lands on the
    # other's bar
    ids = [v["id"] for v in diagram["vertices"]]
    for i, vi in enumerate(ids):
        for vj in ids[i + 1:]:
            if pts[vi][0][1] != pts[vj][0][1]:
                continue
            lo1, hi1 = _glyph_span(diagram, vi, pts)
            lo2, hi2 = _glyph_span(diagram, vj, pts)
            if min(hi1, hi2) + _CLEARANCE > max(lo1, lo2):
                overlaps += 1

    crossings = sum(_crosses(*segs[i], *segs[j])
                    for i in range(len(segs)) for j in range(i + 1, len(segs)))
    length = sum(math.dist(a, b) for a, b in segs)
    return overlaps, crossings, length

def curves_of(diagram, pts):
    """(src, dst, rad) per line, oriented so the arrow runs src -> dst."""
    bows = _bows(diagram)
    out = []
    for i, line in enumerate(diagram["lines"]):
        up = line_direction(line) == "up"
        a, b = _endpoints_xy(diagram, line, pts)
        # arrow points "up" for particle, "down" for hole
        lo, hi = (a, b) if a[1] <= b[1] else (b, a)
        src, dst = (lo, hi) if up else (hi, lo)
        out.append((src, dst, bows.get(i, 0.0) if up else -bows.get(i, 0.0)))
    return out

def label_clashes(diagram, pts, curves):
    """How many labels still sit on a line, a vertex or another label.

    Placement is greedy, so a crowded diagram can run out of clear seats; this
    counts what it had to settle for.
    """
    curve_pts = [_bezier(s, d, r, k / 16)[0]
                 for s, d, r in curves for k in range(17)]
    vlabels = place_vertex_labels(diagram, pts, curve_pts)
    vpos = [(p[0], p[1]) for p in vlabels.values()]
    seats = _place_labels(diagram, pts, curves, vpos)

    bad = 0
    for i, seat in enumerate(seats):
        own = [_bezier(*curves[i], k / 16)[0] for k in range(17)]
        near_line = any(math.dist(seat, p) < _LABEL_CLEAR * 0.6
                        for p in curve_pts if p not in own)
        near_label = any(math.dist(seat, s) < _LABEL_CLEAR for j, s in enumerate(seats)
                         if j != i) or any(math.dist(seat, v) < _LABEL_CLEAR
                                           for v in vpos)
        bad += bool(near_line or near_label)
    for i, a in enumerate(vpos):
        if any(math.dist(a, b) < _LABEL_CLEAR for b in vpos[i + 1:]):
            bad += 1
    return bad

def _score(diagram, pts):
    """Lower is better: glyph overlaps dominate, then crossings, then length.

    Weights are lexicographic in practice -- total line length runs to tens of
    units on a wide diagram, so a merely-large crossing weight would let the
    search buy its way out of one by shortening lines.
    """
    overlaps, crossings, length = penalties(diagram, pts)
    return 10000 * overlaps + 1000 * crossings + length

def layout_points(diagram):
    """Attachment points per vertex, chosen to keep lines clear of the glyphs.

    Diagrams here are small (a handful of vertices of rank <= 2), so an exact
    search over slot orderings and vertex offsets is both cheaper to write and
    better than a Sugiyama-style crossing heuristic.
    """
    base = assign_positions(diagram)
    ids = [v["id"] for v in diagram["vertices"]]
    counts = {i: _slot_count(diagram, i) for i in ids}
    orderings = [list(itertools.permutations(range(counts[i]))) for i in ids]

    best = None
    for shifts in itertools.product(_X_SHIFTS, repeat=len(ids)):
        for perms in itertools.product(*orderings):
            pts = {}
            for vid, shift, perm in zip(ids, shifts, perms):
                cx, cy = base[vid]
                n = counts[vid]
                pts[vid] = [(cx + shift + (perm[k] - (n - 1) / 2) * _POINT_GAP, cy)
                            for k in range(n)]
            s = _score(diagram, pts)
            if best is None or s < best[0]:
                best = (s, pts)
    return best[1]

def _vertex_extent(diagram, vid, pts):
    """(left, right, y) of the glyph actually drawn for this vertex."""
    vpts = pts[vid]
    y = vpts[0][1]
    # layout may permute slots, so points are not sorted by x
    x0, x1 = min(p[0] for p in vpts), max(p[0] for p in vpts)
    kind = diagram["vertices"][vid]["kind"]
    if kind == "fock":
        return x0, x0 + _FOCK_STUB, y
    if kind == "eri":
        return x0, x1, y
    return x0 - _BAR_OVERHANG, x1 + _BAR_OVERHANG, y

def place_vertex_labels(diagram, pts, obstacles):
    """vertex id -> (x, y, horizontal alignment).

    The label takes whichever side is clear of the lines. It cannot simply go
    right: an external line leaving the rightmost point rises straight through
    that spot, and such a line is exempt from the layout's overlap test because
    it attaches to this very vertex.
    """
    gap, lift = 0.14, 0.13
    placed = {}
    for v in diagram["vertices"]:
        left, right, y = _vertex_extent(diagram, v["id"], pts)
        candidates = [(right + gap, y + lift, "left"), (right + gap, y - lift, "left"),
                      (left - gap, y + lift, "right"), (left - gap, y - lift, "right"),
                      (right + gap, y, "left"), (left - gap, y, "right")]
        near = obstacles + [(p[0], p[1]) for p in placed.values()]
        placed[v["id"]] = min(
            candidates,
            key=lambda c: sum(math.dist((c[0], c[1]), o) < _LABEL_CLEAR
                              for o in near))
    return placed

def _draw_vertex(ax, diagram, vid, pts, label_pos, fontsize=12):
    """One vertex glyph, per Fig. 10.1: `>--x` for f, `>--<` for g, bar for T."""
    left, right, y = _vertex_extent(diagram, vid, pts)
    vpts = pts[vid]
    x0, x1 = min(p[0] for p in vpts), max(p[0] for p in vpts)
    kind = diagram["vertices"][vid]["kind"]
    if kind == "fock":
        # rule 2: one-particle vertex is a dashed stub ending in a cross
        ax.plot([left, right], [y, y], "--", color="black", lw=1.2)
        ax.plot([right], [y], marker="x", color="black", ms=7, mew=1.5)
    elif kind == "eri":
        # rule 3: two-particle vertex spans exactly between its two points
        ax.plot([x0, x1], [y, y], "--", color="black", lw=1.2)
        ax.plot([x0, x1], [y, y], "o", color="black", ms=3.5)
    else:
        ax.plot([left, right], [y, y], "-", color="black", lw=2.5)

    lx, ly, ha = label_pos
    ax.text(lx, ly, diagram["vertices"][vid]["label"], ha=ha, va="center",
            fontsize=fontsize, style="italic")
    return max(right, lx) + 0.2

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

def _bezier(src, dst, rad, t):
    """Point and unit tangent at parameter t along a matplotlib arc3 curve,
    which is the quadratic Bezier with control point at the chord midpoint
    displaced by rad*(dy, -dx)."""
    (x1, y1), (x2, y2) = src, dst
    dx, dy = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2 + rad * dy, (y1 + y2) / 2 - rad * dx
    px = (1 - t) ** 2 * x1 + 2 * t * (1 - t) * cx + t ** 2 * x2
    py = (1 - t) ** 2 * y1 + 2 * t * (1 - t) * cy + t ** 2 * y2
    tx = 2 * (1 - t) * (cx - x1) + 2 * t * (x2 - cx)
    ty = 2 * (1 - t) * (cy - y1) + 2 * t * (y2 - cy)
    norm = math.hypot(tx, ty) or 1.0
    return (px, py), (tx / norm, ty / norm)

def _draw_line(ax, src, dst, rad):
    """Draw one directed line with its arrowhead at the curve's midpoint."""
    ax.annotate("", xy=dst, xytext=src,
                arrowprops=dict(arrowstyle="-", color="black", lw=1.2,
                                shrinkA=0, shrinkB=0,
                                connectionstyle=f"arc3,rad={rad}"))
    (mx, my), (ux, uy) = _bezier(src, dst, rad, 0.5)
    ax.annotate("", xy=(mx + _HEAD * ux, my + _HEAD * uy),
                xytext=(mx - _HEAD * ux, my - _HEAD * uy),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2,
                                shrinkA=0, shrinkB=0, mutation_scale=14))

def _place_labels(diagram, pts, curves, extra=()):
    """An (x, y) for each line's index label, kept off the lines, the vertex
    bars, the vertex labels and each other.

    Each label may slide along its own curve and sit at one of a few distances
    from it; the first candidate clear of everything already placed wins, so
    labels stay close to their line unless something is in the way.
    """
    samples = [[_bezier(s, d, r, k / 16)[0] for k in range(17)] for s, d, r in curves]
    bars = []
    for v in diagram["vertices"]:
        lo, hi = _glyph_span(diagram, v["id"], pts)
        y = pts[v["id"]][0][1]
        # sample by distance, not a fixed count: a fixed count leaves gaps wider
        # than the clearance radius on a long bar, and labels slip through them
        n = max(int((hi - lo) / (_LABEL_CLEAR / 2)), 2)
        bars += [(lo + (hi - lo) * k / n, y) for k in range(n + 1)]

    placed = [None] * len(curves)
    # two passes: a label placed early cannot see the ones placed after it, so
    # revisit each once the whole set is down
    for i in list(range(len(curves))) * 2:
        src, dst, rad = curves[i]
        others = ([p for j, s in enumerate(samples) if j != i for p in s]
                  + bars + list(extra)
                  + [q for j, q in enumerate(placed) if q and j != i])
        # a bowed line is labelled outside its arc, where the loop is already
        # open; a straight one may take either side, which is what keeps two
        # neighbouring verticals from pushing their labels to the same spot
        if rad:
            sides = [math.copysign(1, rad)]
        else:
            outward = 1 if _bezier(src, dst, 0, 0.5)[0][0] >= _centre_x(pts) else -1
            sides = [-outward, outward]

        best, best_cost = None, None
        for t in (0.5, 0.36, 0.64, 0.24, 0.76, 0.14, 0.86):
            for scale in (1.0, 1.5, 2.1, 2.8):
                for rank, side in enumerate(sides):
                    (px, py), (ux, uy) = _bezier(src, dst, rad, t)
                    nx, ny = side * uy, -side * ux
                    cand = (px + _LABEL_OFF * scale * nx,
                            py + _LABEL_OFF * scale * ny)
                    clashes = sum(math.dist(cand, o) < _LABEL_CLEAR for o in others)
                    cost = (clashes, scale, rank, abs(t - 0.5))
                    if best_cost is None or cost < best_cost:
                        best, best_cost = cand, cost
                    if clashes == 0:
                        break
                if best_cost[0] == 0:
                    break
            if best_cost[0] == 0:
                break
        placed[i] = best
    return placed

def _centre_x(pts):
    xs = [x for vp in pts.values() for x, _ in vp]
    return (min(xs) + max(xs)) / 2

def _draw(ax, diagram, fontsize=13):
    pts = layout_points(diagram)

    curves = curves_of(diagram, pts)
    curve_pts = [_bezier(s, d, r, k / 16)[0]
                 for s, d, r in curves for k in range(17)]
    vlabels = place_vertex_labels(diagram, pts, curve_pts)
    right = max(_draw_vertex(ax, diagram, v["id"], pts, vlabels[v["id"]],
                             fontsize + 1) for v in diagram["vertices"])

    for curve in curves:
        _draw_line(ax, *curve)

    # rule 1: every line carries its index. Vertex labels are already fixed, so
    # feed their real positions in -- a label placed to the left of its glyph
    # falls outside the span the line labels otherwise avoid.
    seats = _place_labels(diagram, pts, curves,
                          [(p[0], p[1]) for p in vlabels.values()])
    for line, (lx, ly) in zip(diagram["lines"], seats):
        ax.text(lx, ly, "$%s$" % line["index"], ha="center", va="center",
                fontsize=fontsize - 1)

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

def render_grid(diagrams, out_path, ncols=None):
    """One panel per term, for a whole Sum."""
    if ncols is None:
        # amplitude diagrams carry external target lines and run much wider than
        # closed energy ones, so give them fewer, roomier columns
        ncols = 3 if max(len(d["lines"]) for d in diagrams) > 5 else 4
    nrows = -(-len(diagrams) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 4.2 * nrows),
                             squeeze=False)
    flat = [a for row in axes for a in row]
    for ax, d in zip(flat, diagrams):
        _draw(ax, d, fontsize=11)
    for ax in flat[len(diagrams):]:
        ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)

if __name__ == "__main__":
    src = sys.stdin if sys.argv[1] == "-" else open(sys.argv[1])
    with src as f:
        loaded = json.load(f)
    if isinstance(loaded, list):
        render_grid(loaded, sys.argv[2])
    else:
        render(loaded, sys.argv[2])
