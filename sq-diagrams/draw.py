"""Render a Brandow diagram from the extractor's JSON."""
import json

LEVEL = {"eri": 1, "fock": 1, "ampl": 0}

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
