import json, os, subprocess, pathlib, sys
ROOT = pathlib.Path(__file__).parent.parent
BIN = os.environ.get("SQ_DIAGRAM_BIN", str(ROOT / "build" / "sq-diagram-extract"))
sys.path.insert(0, str(ROOT / "sq-diagrams"))
import draw

def run(term):
    out = subprocess.run([BIN, term], capture_output=True, text=True, check=True)
    return out.stdout.strip()

def test_emits_json():
    d = json.loads(run("t{a1,a2;i1,i2}"))
    assert d["term"] == "t{a1,a2;i1,i2}" and "vertices" in d

def test_vertices():
    d = json.loads(run("g{i1,i2;a1,a2} t{a2,a3;i1,i2}"))
    assert [v["kind"] for v in d["vertices"]] == ["eri", "ampl"]
    v0 = d["vertices"][0]
    assert v0["label"] == "g"
    assert v0["bra"] == ["i_1", "i_2"] and v0["ket"] == ["a_1", "a_2"]
    assert d["vertices"][1]["ket"] == ["i_1", "i_2"]

def test_lines():
    d = json.loads(run("g{i1,i2;a1,a2} t{a2,a3;i1,i2}"))
    lines = {l["index"]: l for l in d["lines"]}
    # i_1,i_2 shared (internal, hole); a_2 shared (internal, particle);
    # a_1,a_3 appear once (external, particle).
    assert lines["i_1"]["type"] == "hole" and lines["i_1"]["external"] is False
    assert lines["a_2"]["type"] == "particle" and lines["a_2"]["external"] is False
    assert lines["a_1"]["external"] is True and lines["a_1"]["type"] == "particle"
    assert len(lines["i_1"]["endpoints"]) == 2
    assert len(lines["a_1"]["endpoints"]) == 1

def test_levels_put_H_above_amplitudes():
    d = json.loads(run("g{i1,i2;a1,a2} t{a2,a3;i1,i2}"))
    pos = draw.assign_positions(d)
    y = {v["id"]: pos[v["id"]][1] for v in d["vertices"]}
    assert y[0] > y[1]   # eri (id 0) drawn above ampl (id 1)

def test_dagger_amplitude():
    # t-dagger is non-ASCII, so this also guards the UTF-8 round trip
    d = json.loads(run("g{a1,a2;i1,i2} t⁺{i1,i2;a1,a2}"))
    assert [v["kind"] for v in d["vertices"]] == ["eri", "deexc"]
    assert d["vertices"][1]["label"] == "t⁺"

def test_sum_emits_one_diagram_per_term():
    d = json.loads(run("f{i1;a1} t{a1;i1} + 1/4 g{i1,i2;a1,a2} t{a1,a2;i1,i2}"))
    assert isinstance(d, list) and len(d) == 2
    assert {v["kind"] for t in d for v in t["vertices"]} == {"fock", "ampl", "eri"}

def test_slot_position_survives_antisymmetry():
    # Terminal::slot_group_ord is identically 0 for antisymmetric tensors, so
    # position has to come from the index's place in the bra/ket list. Typing a
    # term without :A-C-S gives NONsymmetric tensors, where the ordinal does
    # increment -- which is why this needs the explicit annotation to bite.
    d = json.loads(run("1/4 g{i_1,i_2;a_1,a_2}:A-C-S * t{a_1,a_2;i_1,i_2}:A-N-S"))
    lines = {l["index"]: l for l in d["lines"]}
    assert lines["i_1"]["endpoints"][0]["pos"] == 0
    assert lines["i_2"]["endpoints"][0]["pos"] == 1
    assert draw.count_loops(d) == 2       # collapses to 1 if pos is degenerate

def test_signs_of_the_three_energy_diagrams():
    # Shavitt & Bartlett eq. (10.21): all three CC energy terms carry +
    for term in ("1/4 g{i1,i2;a1,a2} t{a1,a2;i1,i2}",
                 "f{i1;a1} t{a1;i1}",
                 "1/2 g{i1,i2;a1,a2} t{a1;i1} t{a2;i2}"):
        assert draw.diagram_sign(json.loads(run(term))) == 1, term

def test_antisymmetriser_is_a_projection_not_a_vertex():
    d = json.loads(run("Â{i_1;a_1}:A-C-S * f{a_1;i_1}:A-C-S"))
    assert [v["label"] for v in d["vertices"]] == ["f"]
    assert d["targets"] == {"bra": ["i_1"], "ket": ["a_1"]}
    assert all(l["external"] for l in d["lines"])   # target indices run free
    # and rule 8's quasiloops close them, so the sign is still computable
    assert draw.diagram_sign(d) == 1

def test_sign_is_none_for_open_diagrams():
    d = json.loads(run("-1/2 g{i1,i2;a1,a2} t{a2,a3;i1,i2}"))
    assert draw.diagram_sign(d) is None   # quasiloops not handled

def test_term_expression():
    d = json.loads(run("1/4 g{i1,i2;a1,a2} t{a1,a2;i1,i2}"))
    s = draw.term_expression(d)
    assert r"\frac{1}{4}" in s and r"\langle" in s and "||" in s
    assert r"\sum_{i_1\,i_2\,a_1\,a_2}" in s   # rule 5: internal labels only

def test_layout_keeps_lines_off_the_glyphs():
    # a_1 runs from t straight up to t-dagger, skipping the level f sits on, so
    # a naive shared-centre layout draws it through the f vertex
    d = json.loads(run("-1 f{i_1;i_2}:A-C-S * t{a_1;i_1}:A-N-S * t⁺{i_2;a_1}:A-N-S"))
    pts = draw.layout_points(d)
    overlaps, crossings, _ = draw.penalties(d, pts)
    assert (overlaps, crossings) == (0, 0)
    # and no label ends up sitting on a line, a vertex or another label
    assert draw.label_clashes(d, pts, draw.curves_of(d, pts)) == 0

def test_levels_are_centred():
    # g over two t1 vertices: the lone interaction sits between them, not above
    # the leftmost one.
    d = {"vertices": [{"id": 0, "kind": "eri"},
                      {"id": 1, "kind": "ampl"}, {"id": 2, "kind": "ampl"}]}
    pos = draw.assign_positions(d)
    assert pos[0][0] == 0.0
    assert pos[1][0] == -1.0 and pos[2][0] == 1.0

def test_line_direction():
    assert draw.line_direction({"type": "particle"}) == "up"
    assert draw.line_direction({"type": "hole"}) == "down"

def test_render_smoke(tmp_path):
    d = json.loads(run("g{i1,i2;a1,a2} t{a2,a3;i1,i2}"))
    out = tmp_path / "d.svg"
    draw.render(d, str(out))
    assert out.exists() and out.stat().st_size > 0
