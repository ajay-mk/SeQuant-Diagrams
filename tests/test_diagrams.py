import json, os, subprocess, pathlib
BIN = os.environ.get("SQ_DIAGRAM_BIN",
                     str(pathlib.Path(__file__).parent.parent / "build" / "sq-diagram-extract"))

def run(term):
    out = subprocess.run([BIN, term], capture_output=True, text=True, check=True)
    return out.stdout.strip()

def test_parse_roundtrip():
    # Task 1: binary echoes the re-serialized term; proves link+context+parse.
    out = run("t{a1,a2;i1,i2}")
    assert "t" in out and "a_1" in out and "i_1" in out
