"""Stage-1 drafting tests: the profiler, the heuristic draft, and the validator.

The decisive test is at the bottom: hand `validate()` the REAL official spec and it must
confirm the organisers' exact row counts (1,141,112 / 124,909 / 170,588) from the files
on disk. A validator that cannot reproduce a known-correct split cannot be trusted to
reject a wrong one.
"""
import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import draft_task as DT                                     # noqa: E402
from agent import profile_data as PD                                   # noqa: E402

ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  ok   {name}")
    else:
        fail += 1
        print(f"  FAIL {name}  {detail}")


TMP = Path(tempfile.mkdtemp(prefix="track2_draft_"))


def write_csv(name, header, rows):
    p = TMP / name
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return p


print("== profiler: types, binary detection, dates")
# date must NOT determine the label. An earlier version used date = 20220401 + i%10
# with label = i%2, which makes the label constant within each date (10 and 2 share a
# factor) -- and the validator correctly rejected the resulting draft. Using i//40 for
# the date keeps the two independent.
rows = [[f"u{i%7}", f"v{i%11}", 20220401 + (i // 40), i % 2, i * 0.5, ""]
        for i in range(400)]
write_csv("log.csv", ["user_id", "video_id", "date", "clicked", "dwell", "empty"], rows)
pr = PD.profile_csv(TMP / "log.csv")
byname = {c["name"]: c for c in pr["columns"]}
check("exact row count", pr["rows"] == 400, str(pr["rows"]))
check("int column typed int", byname["date"]["type"] == "int")
check("float column typed float", byname["dwell"]["type"] == "float")
check("str column typed str", byname["user_id"]["type"] == "str")
check("binary column detected", byname["clicked"]["binary"] is True)
check("non-binary not marked binary", byname["dwell"]["binary"] is False)
check("positive rate computed", abs(byname["clicked"]["positive_rate"] - 0.5) < 0.01)
check("date pattern detected", byname["date"]["looks_like_date"] is True)
check("non-date int not called a date", byname["clicked"]["looks_like_date"] is False)
check("all-empty column is 100% null", byname["empty"]["null_rate"] == 1.0)
check("empty column is not binary", byname["empty"]["binary"] is False)
# These four were one assertion ending in `or True`, which cannot fail -- and it was
# masking a real gap: a file whose header row is present but EMPTY is not caught by
# StopIteration (csv.reader yields [] rather than stopping), so it profiled as a valid
# zero-column file and the failure surfaced much later as a draft with `label: None`.
_zero = TMP / "zero.csv"
_zero.write_bytes(b"")
_blank = write_csv("blank.csv", [], [])          # a lone "\r\n": header row present, empty
check("a zero-byte file is flagged", "error" in PD.profile_csv(_zero),
      str(PD.profile_csv(_zero)))
check("an empty HEADER ROW is flagged too, not read as zero columns",
      "error" in PD.profile_csv(_blank), str(PD.profile_csv(_blank)))
check("the two get different explanations",
      PD.profile_csv(_zero)["error"] != PD.profile_csv(_blank)["error"],
      f"{PD.profile_csv(_zero)['error']!r} vs {PD.profile_csv(_blank)['error']!r}")
_hdr_only = write_csv("hdronly.csv", ["a", "b"], [])
check("a header-only file is NOT an error (a real table with no rows)",
      "error" not in PD.profile_csv(_hdr_only)
      and PD.profile_csv(_hdr_only)["n_columns"] == 2)
check("but it yields no role candidates, so it cannot be picked as a log",
      not any(PD.candidates(PD.profile_csv(_hdr_only)).values()))

print("== profiler: role candidates are ranked, not chosen")
c = PD.candidates(pr)
check("label candidate found", c["label"] and c["label"][0]["column"] == "clicked",
      str(c["label"]))
check("time candidate is the date column", c["time"][0]["column"] == "date",
      str(c["time"]))
check("group candidate is user_id", c["group"][0]["column"] == "user_id", str(c["group"]))
check("item candidate is video_id", c["item"][0]["column"] == "video_id", str(c["item"]))
check("candidates are score-ordered",
      all(a["score"] >= b["score"] for a, b in zip(c["time"], c["time"][1:])))
const = write_csv("const.csv", ["a", "flag"], [[i, 0] for i in range(50)])
check("a constant binary column is not a label candidate",
      not PD.candidates(PD.profile_csv(const))["label"])

print("== goal matching is token-prefix, not substring")
g = DT._tokens("rank each user's impressions by whether they long-viewed the video")
check("long_view reached from 'long-viewed'", DT._named_in("long_view", g))
check("is_click not matched", not DT._named_in("is_click", g))
check("is_like not matched", not DT._named_in("is_like", g))
check("user_id matched", DT._named_in("user_id", g))
check("stop words alone do not match", not DT._named_in("is_follow", g))
check("empty column name never matches", not DT._named_in("_", g))
check("tokens split on punctuation",
      DT._tokens("a-b_c.d") == ["a", "b", "c", "d"], str(DT._tokens("a-b_c.d")))

print("== a goal mention is a BONUS, never an override")
check("GOAL_BONUS is smaller than a structural signal", DT.GOAL_BONUS <= 3.0,
      str(DT.GOAL_BONUS))

print("== span overlap")
check("identical spans overlap", DT._overlaps([1, 10], [1, 10]))
check("contained span overlaps", DT._overlaps([1, 10], [3, 5]))
check("adjacent spans do not overlap", not DT._overlaps([1, 10], [11, 20]))
check("disjoint spans do not overlap", not DT._overlaps([1, 5], [50, 60]))
check("a 1-day touch of a long span is below the threshold",
      not DT._overlaps([1, 100], [100, 300]), "10% of the shorter")
check("None span never overlaps", not DT._overlaps(None, [1, 2]))

print("== three-way split never puts a boundary inside a time value")
sp = DT._three_way(list(range(20220401, 20220431)))
check("splits are ordered and disjoint",
      sp["train"][1] < sp["valid"][0] < sp["valid"][1] < sp["test"][0], str(sp))
check("split covers the whole range",
      sp["train"][0] == 20220401 and sp["test"][1] == 20220430, str(sp))
check("degenerate 1-value range does not crash", DT._three_way([5]) is not None)
check("2-value range does not crash", DT._three_way([5, 6]) is not None)
sp3 = DT._three_way([1, 2, 3])
check("3-value range yields three non-empty splits",
      sp3["train"][0] <= sp3["train"][1] < sp3["valid"][0] and sp3["valid"][1] < sp3["test"][0],
      str(sp3))

print("== validator rejects what it should")
GOOD = {"slug": "t", "title": "t", "log_files": ["log.csv"], "label": "clicked",
        "group_by": "user_id", "item_col": "video_id", "time_col": "date",
        "splits": {"train": [20220401, 20220404], "valid": [20220405, 20220407],
                   "test": [20220408, 20220410]},
        "metric": "within_user_ranking_gauc_ndcg5", "feature_cols": ["user_id"],
        "rationale": "x"}
v = DT.validate(GOOD, TMP)
check("a good draft validates", v["ok"], str(v["problems"]))
check("row counts are reported", sum(v["evidence"]["rows"].values()) > 0,
      str(v["evidence"].get("rows")))


def bad(mut, needle):
    d = json.loads(json.dumps(GOOD))
    for k, val in mut.items():
        if val is DELETE:
            d.pop(k, None)
        else:
            d[k] = val
    r = DT.validate(d, TMP)
    return (not r["ok"]) and any(needle in p for p in r["problems"]), r["problems"]


DELETE = object()
cases = [
    ({"label": "nope"}, "not a column"),
    ({"time_col": "nope"}, "not a column"),
    ({"group_by": "nope"}, "not a column"),
    ({"item_col": "nope"}, "not a column"),
    ({"feature_cols": ["nope"]}, "not a column"),
    ({"label": "dwell"}, "not binary"),
    ({"log_files": ["missing.csv"]}, "does not exist"),
    ({"log_files": ["../log.csv"]}, "bare filename"),
    ({"log_files": ["/etc/passwd"]}, "bare filename"),
    ({"log_files": []}, "non-empty"),
    ({"metric": "made_up_metric"}, "not implemented"),
    ({"metric": "within_user_ranking_gauc_ndcg5", "group_by": None}, "no group_by"),
    ({"splits": {"train": [1, 2], "valid": [3, 4]}}, "exactly"),
    ({"splits": {"train": [20220401, 20220406], "valid": [20220405, 20220407],
                 "test": [20220408, 20220410]}}, "overlap"),
    ({"splits": {"train": [20220405, 20220401], "valid": [20220406, 20220407],
                 "test": [20220408, 20220410]}}, "inverted"),
    ({"splits": {"train": [20220401, 20220402], "valid": [20220409, 20220410],
                 "test": [20220403, 20220404]}}, "future"),
    ({"splits": {"train": [20220401, 20220404], "valid": [20220405, 20220407],
                 "test": [20990101, 20990102]}}, "ZERO rows"),
    ({"dev_split": "test"}, "must differ"),
    ({"dev_split": "nonsense"}, "must be one of"),
    ({"label": DELETE}, "missing required"),
    ({"splits": DELETE}, "missing required"),
]
for mut, needle in cases:
    got, probs = bad(mut, needle)
    check(f"rejects {str(mut)[:52]} ({needle})", got, str(probs)[:150])

print("== positives are counted NUMERICALLY, not by string (adversarial review, bug 49)")
# `v not in ("", "0")` counted every other spelling of zero as a POSITIVE, so an all-zero
# float column reported positive_rate 1.0 and sailed past the constant-label guard -- which
# only fires on exactly 0.0 or 1.0, and 1.0 is what the bug produced.
write_csv("zeros.csv", ["user_id", "date", "lab"],
          [[f"u{i%50}", 20220401 + (i // 150), "0.0"] for i in range(600)])
ZD = {"slug": "t", "title": "t", "log_files": ["zeros.csv"], "label": "lab",
      "group_by": "user_id", "time_col": "date",
      "splits": {"train": [20220401, 20220401], "valid": [20220402, 20220402],
                 "test": [20220403, 20220403]},
      "metric": "within_user_ranking_gauc_ndcg5", "rationale": "x"}
vz = DT.validate(ZD, TMP)
check("an all-zero '0.0' column reports rate 0.0, not 1.0",
      all(r == 0.0 for r in vz["evidence"]["positive_rate"].values()),
      str(vz["evidence"]["positive_rate"]))
check("and is therefore rejected as a constant label",
      not vz["ok"] and any("constant" in p for p in vz["problems"]), str(vz["problems"]))
for spelling in ("00", " 0", "0.000", "-0.0"):
    write_csv("sp.csv", ["user_id", "date", "lab"],
              [[f"u{i%50}", 20220401 + (i // 150), spelling] for i in range(600)])
    r = DT.validate(dict(ZD, log_files=["sp.csv"]), TMP)["evidence"]["positive_rate"]
    check(f"{spelling!r} counts as zero", all(x == 0.0 for x in r.values()), str(r))
write_csv("mix.csv", ["user_id", "date", "lab"],
          [[f"u{i%50}", 20220401 + (i // 150), "1.0" if i % 3 == 0 else "0.0"]
           for i in range(600)])
rm = DT.validate(dict(ZD, log_files=["mix.csv"]), TMP)
check("a genuine float-spelled label still gets a real rate",
      abs(rm["evidence"]["positive_rate"]["train"] - 1/3) < 0.02,
      str(rm["evidence"]["positive_rate"]))
write_csv("txt.csv", ["user_id", "date", "lab"],
          [[f"u{i%50}", 20220401 + (i // 150), "yes" if i % 2 else ""] for i in range(600)])
rt = DT.validate(dict(ZD, log_files=["txt.csv"]), TMP)["evidence"]["positive_rate"]
check("a non-numeric label keeps the any-non-empty reading",
      abs(rt["train"] - 0.5) < 0.02, str(rt))

print("== group_by must actually GROUP (adversarial review, bug 50)")
# Measured on the real metric: per-row-unique groups give GAUC exactly 0.5000 and nDCG@5
# equal to the label rate. Only the EXISTENCE of group_by was checked before.
write_csv("grp.csv", ["rowid", "user_id", "date", "lab"],
          [[f"r{i}", f"u{i%50}", 20220401 + (i // 200), i % 2] for i in range(600)])
GD = dict(ZD, log_files=["grp.csv"], label="lab")
vg = DT.validate(dict(GD, group_by="rowid"), TMP)
check("a per-row-unique group_by is REJECTED",
      not vg["ok"] and any("does not group" in p for p in vg["problems"]),
      str(vg["problems"])[:150])
check("the refusal quotes rows-per-group",
      any("1.00 rows per group" in p for p in vg["problems"]))
check("the refusal explains the metric consequence",
      any("no-information 0.5" in p for p in vg["problems"]))
vgood = DT.validate(dict(GD, group_by="user_id"), TMP)
check("a real group_by on the same data passes", vgood["ok"], str(vgood["problems"]))
check("groups are reported per split", set(vgood["evidence"]["groups"]) == {"train","valid","test"})
write_csv("thin.csv", ["user_id", "date", "lab"],
          [[f"u{i//2}", 20220401 + (i // 200), i % 2] for i in range(600)])
vt = DT.validate(dict(GD, log_files=["thin.csv"], group_by="user_id"), TMP)
check("2 rows per group is a WARNING, not a rejection",
      vt["ok"] and any("rows per group" in w for w in vt["warnings"]),
      f"ok={vt['ok']} warns={vt['warnings']}")
check("an UNGROUPED metric is not subjected to the check",
      DT.validate(dict(GD, group_by="rowid", metric="global_auc"), TMP)["ok"]
      or "does not group" not in str(DT.validate(
          dict(GD, group_by="rowid", metric="global_auc"), TMP)["problems"]))

print("== validator warns without rejecting")
write_csv("log2.csv", ["user_id", "video_id", "date", "clicked", "dwell", "empty"],
          [[f"u{i%7}", f"v{i%11}", 20220401 + (i // 160), i % 2, i * 0.5, ""]
           for i in range(1600)])
d = json.loads(json.dumps(GOOD))
d["log_files"] = ["log.csv", "log2.csv"]
v = DT.validate(d, TMP)
check("two logs over one period is a WARNING, not a rejection", v["ok"], str(v["problems"]))
check("the overlap is warned about", any("both cover" in w for w in v["warnings"]),
      str(v["warnings"]))
check("the density difference is quantified",
      any("rows per time unit" in w for w in v["warnings"]), str(v["warnings"]))
check("density evidence recorded", "log_density" in v["evidence"])
d2 = json.loads(json.dumps(GOOD))
d2["splits"] = {"train": [20220401, 20220402], "valid": [20220403, 20220404],
                "test": [20220405, 20220406]}
v2 = DT.validate(d2, TMP)
check("rows outside every split are warned about",
      v2["ok"] and any("outside every split" in w for w in v2["warnings"]),
      str(v2["warnings"]))

print("== mismatched headers cannot be concatenated")
write_csv("other.csv", ["user_id", "date", "clicked"],
          [[f"u{i}", 20220401, i % 2] for i in range(30)])
d3 = json.loads(json.dumps(GOOD))
d3["log_files"] = ["log.csv", "other.csv"]
v3 = DT.validate(d3, TMP)
check("different headers rejected",
      not v3["ok"] and any("different header" in p for p in v3["problems"]),
      str(v3["problems"]))

print("== end to end on the synthetic dir")
prof = PD.profile_dir(TMP, sample=500)
dr = DT.heuristic_draft(prof, "predict whether the user clicked the video")
check("heuristic picks the binary label named in the goal", dr["label"] == "clicked",
      dr["label"])
check("heuristic picks the date column", dr["time_col"] == "date", dr["time_col"])
check("heuristic picks a group", dr["group_by"] == "user_id", str(dr["group_by"]))
check("heuristic picks an implemented metric", dr["metric"] in DT.METRICS)
check("heuristic draft validates against the real files",
      DT.validate(dr, TMP)["ok"], str(DT.validate(dr, TMP)["problems"]))
check("slug matches the schema pattern",
      __import__("re").match(DT.SCHEMA["properties"]["slug"]["pattern"], dr["slug"]),
      dr["slug"])
check("heuristic states it is a heuristic", "HEURISTIC" in dr["rationale"])

print("== schema and metric registry are self-consistent")
check("schema enumerates exactly the implemented metrics",
      set(DT.SCHEMA["properties"]["metric"]["enum"]) == set(DT.METRICS))
check("schema forbids extra properties", DT.SCHEMA["additionalProperties"] is False)
check("every metric declares its needs",
      all({"needs_group", "needs_binary_label", "impl", "desc"} <= set(v)
          for v in DT.METRICS.values()))
check("schema requires a rationale", "rationale" in DT.SCHEMA["required"])

print("== THE REAL OFFICIAL SPEC must validate with the organisers' exact row counts")
real = Path(__file__).resolve().parents[1] / "KuaiRand-Pure" / "data"
if real.is_dir():
    official = {
        "slug": "kuairand_pure", "title": "KuaiRand-Pure within-user ranking",
        "log_files": ["log_standard_4_08_to_4_21_pure.csv",
                      "log_standard_4_22_to_5_08_pure.csv"],
        "label": "long_view", "group_by": "user_id", "item_col": "video_id",
        "time_col": "date",
        "splits": {"train": [20220408, 20220421], "valid": [20220422, 20220428],
                   "test": [20220429, 20220508]},
        "metric": "within_user_ranking_gauc_ndcg5",
        "feature_cols": ["user_id", "video_id", "tab"],
        "rationale": "the organisers' declared conventions",
    }
    v = DT.validate(official, real)
    check("the official spec validates clean", v["ok"], str(v["problems"]))
    r = v["evidence"]["rows"]
    for k, want in [("train", 1_141_112), ("valid", 124_909), ("test", 170_588)]:
        check(f"{k} row count matches the organisers' published {want:,}",
              r.get(k) == want, f"got {r.get(k)}")
    check("no rows fall outside the official splits",
          v["evidence"]["rows_outside_every_split"] == 0,
          str(v["evidence"]["rows_outside_every_split"]))
    check("the label really is binary in the real data",
          v["evidence"]["label_values"] <= {0.0, 1.0},
          str(v["evidence"]["label_values"]))
    check("the two official logs do NOT overlap (so no warning)",
          not any("both cover" in w for w in v["warnings"]), str(v["warnings"]))
    check("valid split has the expected user count",
          v["evidence"]["groups"]["valid"] == 22_377,
          str(v["evidence"]["groups"].get("valid")))
else:
    print("  skip KuaiRand-Pure/data not present")

import shutil
shutil.rmtree(TMP, ignore_errors=True)
print(f"\n{ok} passed, {fail} failed")
raise SystemExit(1 if fail else 0)
