"""Charts fail silently by drawing a wrong picture, not by raising. These tests aim at
exactly the inputs that produce a plausible-looking lie: nothing, one point, a flat
line (0/0 in the naive normaliser), a NaN in the middle, and curves of unequal length.
"""
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import charts as C                                          # noqa: E402

ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  ok   {name}")
    else:
        fail += 1
        print(f"  FAIL {name}  {detail}")


def nums(svg, attr):
    return [float(x) for x in re.findall(attr + r"='(-?[\d.]+)'", svg)]


print("== _range")
lo, hi = C._range([0.5, 0.5, 0.5])
check("flat series widens instead of collapsing", hi > lo and lo < 0.5 < hi,
      f"got {lo},{hi}")
check("flat range is small, not unit-wide", hi - lo < 0.05, f"span {hi-lo}")
lo, hi = C._range([0.0, 0.0])
check("flat at zero still widens", hi > lo, f"got {lo},{hi}")
lo, hi = C._range([0.1, 0.9])
check("normal series is padded outward", lo < 0.1 and hi > 0.9, f"got {lo},{hi}")

print("== _finite")
check("None dropped", C._finite([1.0, None, 2.0]) == [1.0, 2.0])
check("nan dropped", C._finite([1.0, float("nan"), 2.0]) == [1.0, 2.0])
check("inf dropped", C._finite([1.0, float("inf")]) == [1.0])
check("ints coerced", C._finite([1, 2]) == [1.0, 2.0])

print("== trajectory degenerate inputs")
for label, arg in [("empty", []), ("all-None", [None, None]),
                   ("all-nan", [float("nan")])]:
    s = C.trajectory(arg)
    check(f"{label} -> note, not svg", "<svg" not in s and "muted" in s, s[:60])

s = C.trajectory([0.6])
check("single point renders an svg", "<svg" in s)
check("single point draws no data polyline", s.count("class='series'") == 0)
check("single point still draws a marker", "circle" in s)

s = C.trajectory([0.5, 0.5, 0.5])
ys = nums(s, "cy")
check("flat series is not pinned to the axis",
      len(set(round(y, 1) for y in ys)) == 1 and min(ys) > C.PAD["t"],
      f"cy={ys}")

print("== trajectory geometry")
vals = [0.1, 0.5, 0.3, 0.9]
s = C.trajectory(vals)
cys = nums(s, "cy")
check("one marker per point", len(cys) == 4, f"{len(cys)}")
check("higher value maps to smaller y (svg y grows downward)",
      cys[3] < cys[1] < cys[2] < cys[0], f"{cys}")
# These two asserted only that the string "pt-best" appears, so they passed with the
# highlight on the WRONG point -- proven by mutating charts.py to `bi = 0` and watching them
# stay green. Assert the marker's COORDINATES instead.
check("exactly one best marker", s.count("pt-best") == 1)
_bm = re.search(r"cx='([\d.]+)' cy='([\d.]+)' r='[\d.]+' class='pt-best'", s)
check("the best marker exists with coordinates", _bm is not None, s[:200])
if _bm:
    _bx, _by = float(_bm.group(1)), float(_bm.group(2))
    _imax = max(range(len(vals)), key=lambda i: vals[i])
    _all = [(float(a), float(b)) for a, b in
            re.findall(r"cx='([\d.]+)' cy='([\d.]+)'", s)]
    check("the best marker sits on the MAXIMUM point, not merely somewhere",
          (_bx, _by) == _all[_imax], f"marker at {(_bx, _by)}, max point at {_all[_imax]}")
    check("the best marker is the highest point on the canvas (smallest y)",
          abs(_by - min(y for _x, y in _all)) < 1e-6, f"{_by} vs {min(y for _x,y in _all)}")

print("== trajectory baseline and band")
s = C.trajectory([0.601, 0.603], baseline=0.6016, band=0.0008)
check("band rect drawn", "class='band'" in s)
check("reference rule drawn", "class='ref'" in s)
check("baseline value labelled", "0.6016" in s)
check("band is labelled as seed sd", "seed sd" in s)
r_h = nums(s, "height")
check("band has non-zero height", r_h and r_h[0] > 0, f"{r_h}")
s2 = C.trajectory([0.601], baseline=0.6016)
check("baseline without band draws no rect", "class='band'" not in s2)
# a baseline far outside the data must still be inside the drawn range
s3 = C.trajectory([0.9, 0.91], baseline=0.60, band=0.001)
ref_y = nums(s3, "y1")
check("far baseline stays inside the frame",
      all(C.PAD["t"] - 1 <= y <= 260 - C.PAD["b"] + 1 for y in ref_y), f"{ref_y}")

print("== trajectory running best is monotone non-decreasing")
s = C.trajectory([0.5, 0.2, 0.8, 0.1])
step = re.search(r"class='best-step'", s)
check("best-step drawn", step is not None)
pts = re.search(r"points='([^']+)' class='best-step'", s)
ys = [float(p.split(",")[1]) for p in pts.group(1).split()] if pts else []
check("running best never decreases (y never increases)",
      all(b <= a + 1e-6 for a, b in zip(ys, ys[1:])), f"{ys}")

print("== multiline")
check("empty -> note", "<svg" not in C.multiline([]))
check("all-empty series -> note", "<svg" not in C.multiline([("a", []), ("b", [])]))
s = C.multiline([("a", [1.0, 2.0, 3.0]), ("b", [1.0, 1.5])])
# each curve is <polyline ...><title/></polyline>, so count the OPENING tag
check("both curves drawn", s.count("<polyline") == 2, s.count("<polyline"))
check("unequal lengths do not raise", "<svg" in s)
xs = [p.split(",")[0] for p in
      re.findall(r"points='([^']+)'", s)[0].split()]
check("longest curve spans the full width",
      abs(float(xs[-1]) - (880 - C.PAD["r"])) < 1e-6, f"{xs[-1]}")
short = re.findall(r"points='([^']+)'", s)[1].split()
check("short curve ends before the right edge",
      float(short[-1].split(",")[0]) < 880 - C.PAD["r"], f"{short[-1]}")
s = C.multiline([("a", [1.0, 2.0]), ("b", [3.0, 4.0])], emphasise="b")
check("emphasis applied to the named curve", "curve-em" in s)
check("only one curve emphasised", s.count("curve-em") == 1)
# `pt-best` appears whenever the curve is emphasised, so asserting its presence said
# nothing about invert_best. invert_best moves only the marker's INDEX, so check where the
# marker landed -- and check both directions against the same data.
_vals = [3.0, 1.0, 2.0]
s_min = C.multiline([("a", _vals)], invert_best=True, emphasise="a")
s_max = C.multiline([("a", _vals)], invert_best=False, emphasise="a")


def _marker(svg):
    m = re.search(r"cx='([\d.]+)' cy='([\d.]+)' r='[\d.]+' class='pt-best'", svg)
    return (float(m.group(1)), float(m.group(2))) if m else None


check("invert_best=True marks the MINIMUM, not just any point",
      _marker(s_min) is not None and _marker(s_max) is not None
      and _marker(s_min) != _marker(s_max),
      f"min-marker {_marker(s_min)} vs max-marker {_marker(s_max)}")
check("the inverted marker is the LOWEST point (largest y)",
      _marker(s_min)[1] > _marker(s_max)[1],
      f"{_marker(s_min)} should sit below {_marker(s_max)}")
check("the title names the epoch of the value it marked",
      "epoch 2 = 1" in s_min, re.findall(r"<title>([^<]*)</title>", s_min)[-1:])
s = C.multiline([("a", [1.0, None, 3.0])])
check("None inside a curve is dropped, not plotted as zero",
      "<svg" in s and len(re.findall(r"points='([^']+)'", s)[0].split()) == 2)
s = C.multiline([("a", [5.0])])
check("single-point curve renders", "<svg" in s)

print("== bars")
check("empty -> note", "<svg" not in C.bars([]))
check("all-zero totals -> note", "<svg" not in C.bars([("a", 0, 0)]))
s = C.bars([("SetLoss", 5, 1), ("SetK", 2, 0)])
check("one bar per item", s.count("class='bar'") == 2)
check("highlight only where subset > 0", s.count("class='bar-hi'") == 1)
check("improved count labelled as 'total / improved'", "5 / 1" in s, s[:300])
w_all = nums(s, "width")
check("subset bar never exceeds its total", w_all[1] <= w_all[0] + 1e-6, f"{w_all}")
s = C.bars([("a", 2, 5)])
w_all = nums(s, "width")
check("subset larger than total is clamped", w_all[1] <= w_all[0] + 1e-6, f"{w_all}")
check("the 'improved' word is gone from row labels (the caption says it once)",
      "improved" not in C.bars([("a", 5, 1)]))
check("a highlighted row still shows both numbers", "5 / 1" in C.bars([("a", 5, 1)]))


def rightmost_text_end(svg, per_char=6.2):
    """Approximate right edge of every <text>. Anchored-end labels end AT x."""
    worst = 0.0
    for m in re.finditer(r"<text ([^>]*)>([^<]*)</text>", svg):
        attrs, txt = m.group(1), m.group(2)
        x = float(re.search(r"x='([-\d.]+)'", attrs).group(1))
        worst = max(worst, x if "text-anchor='end'" in attrs else x + len(txt) * per_char)
    return worst


print("== no chart may draw text outside its own viewBox (bug: bars clipped every label)")
for name, svg in [
        ("bars worst case", C.bars([("SetLoss", 999, 999), ("x", 1, 0)])),
        ("bars typical", C.bars([("SetLoss", 5, 1), ("SetK", 2, 0)])),
        ("bars long action name", C.bars([("SetListwiseTemperature", 12, 3)])),
        ("bars single item", C.bars([("a", 1, 1)])),
        ("trajectory with baseline", C.trajectory([0.601, 0.603], baseline=0.6016, band=0.0008)),
        ("multiline", C.multiline([("pointwise 43bda4b2", [0.59, 0.60, 0.601])])),
]:
    W = int(re.search(r"viewBox='0 0 (\d+)", svg).group(1))
    end = rightmost_text_end(svg)
    check(f"{name}: text fits in {W}u (ends ~{end:.0f})", end <= W, f"overflows by {end - W:.0f}u")
check("LABEL_ROOM is documented and used, not a bare literal",
      C.LABEL_ROOM >= 60 and "LABEL_ROOM" in
      (Path(__file__).resolve().parents[1] / "agent" / "charts.py").read_text())

print("== escaping and self-containment")
s = C.multiline([("<script>x</script>&", [1.0, 2.0])])
check("series name escaped", "<script>" not in s and "&lt;script&gt;" in s)
s = C.bars([("<b>&", 1, 0)])
check("bar label escaped", "<b>" not in s)
s = C.trajectory([1.0, 2.0], labels=["<i>", "ok"])
check("point label escaped", "<i>" not in s)
for f, a in [(C.trajectory, ([1.0, 2.0],)), (C.multiline, ([("a", [1.0, 2.0])],)),
             (C.bars, ([("a", 1, 0)],))]:
    s = f(*a)
    check(f"{f.__name__}: no script/style/foreignObject",
          not any(t in s for t in ("<script", "<style", "<foreignObject")))
    check(f"{f.__name__}: no external reference",
          "http" not in s and "url(" not in s)
    check(f"{f.__name__}: has viewBox and aria-label",
          "viewBox=" in s and "aria-label=" in s)
    check(f"{f.__name__}: balanced svg tags",
          s.count("<svg") == 1 and s.count("</svg>") == 1)

print(f"\n{ok} passed, {fail} failed")
raise SystemExit(1 if fail else 0)
