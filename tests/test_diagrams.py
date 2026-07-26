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

def test_line_direction():
    assert draw.line_direction({"type": "particle"}) == "up"
    assert draw.line_direction({"type": "hole"}) == "down"
