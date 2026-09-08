"""Freeze the shipped entry, verify the seed profiles, and police the repository's claims.

Three jobs, all of which fail loudly rather than drift:

  1. FREEZE. The submitted predictions, manifest and champion source are recorded here by
     sha256. New research infrastructure must never change them, and the fallback entry must
     stay usable even if every future architecture experiment fails.
  2. PROFILES. `minimal` must genuinely contain no ranking-loss implementation, and must
     reproduce the official pointwise baseline. `inherited` must still contain the prepared
     pairwise switch, because the recorded runs cannot be reproduced without it.
  3. CLAIMS. The repository must not assert things that are false now: superseded metrics as
     current, an independent 20-seed replication, an agent that implemented BPR, or
     architecture research that has not happened.
"""
import ast, hashlib, json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def shipped_files():
    """Return regular files Git would actually ship, excluding local ignored state."""
    raw = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode("utf-8")
    for rel in raw.split("\0"):
        if not rel:
            continue
        path = ROOT / rel
        if path.is_file() and not path.is_symlink():
            yield path

R = []
def check(n, c, extra=""):
    R.append((n, bool(c))); print(f"  {'PASS' if c else 'FAIL'}  {n}{('  '+extra) if extra else ''}")

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

# =============================================================== 1. THE FROZEN ENTRY
#: Every file of the shipped entry, by sha256, measured from the submission as built in the
#: original repository and copied here byte-for-byte. An earlier version of this table held
#: only three files and its hashes were correct for the first 16 characters and invented
#: after that -- so it could not have caught a change. These are the real digests, and all
#: eight files are asserted, including `champion_source/README.md`.
#:
#: If one of these ever fails, that is not a test to update: it means what gets submitted has
#: changed. Updating the table is a deliberate act that must accompany a new submission.
FROZEN = {
    "MANIFEST.json":
        "d230610e27c02af2a863a19bac6271c260117eb02441aad23544da401fcfd9bf",
    "champion_source/README.md":
        "ddbd1d2e67646ef04c47545fd0a1a38298e8e520b9f61f110c307d53dbb5b5df",
    "champion_source/config.json":
        "a4c929614d36259b131e83b24d14831dd8053549ce040a124bc5e41916c49c40",
    "champion_source/features.py":
        "9e3f88ffd89e4773d89c4f7f2e7dbaf21c8c1bb5b11772acd105462a23017ba7",
    "champion_source/model.py":
        "178d313c10586a567a9618cfc09649ffb2bff8909ddd8659089214a0681f9f90",
    "champion_source/train.py":
        "944115f8b3c62404add0dd47e3ec513ad5d4c2cd3eb4b5348ab514b7a83883a9",
    "submission_test.csv":
        "7cc68e2728ef3afb4c0881e28b1bf4c21215e2ce5db6eebc6f4b1b91b77de6d4",
    "valid_scores.npy":
        "ffe4fed193db4b33cb8b2b88817c518df5c11f90c1d12b12393af641a398cfbd",
}
CHAMPION_TREE = "fe985741a8368e082be4c71e0d9e3be06ca333f2aac1b26216203cbf2283c3b5"

sub = ROOT / "submission_compliant"
check("the frozen entry is present", sub.is_dir())
cs = sub / "champion_source"
check("the champion source ships with the entry", cs.is_dir())

# Every frozen file, asserted against its own recorded digest.
for rel, want in sorted(FROZEN.items()):
    f = sub / rel
    if not f.is_file():
        check(f"frozen file present: {rel}", False, "MISSING")
        continue
    got = sha(f)
    check(f"frozen and unchanged: {rel}", got == want,
          "" if got == want else f"got {got[:20]} want {want[:20]}")
check("the frozen table covers every shipped file, none added or removed",
      {str(p.relative_to(sub)) for p in sub.rglob("*") if p.is_file()} == set(FROZEN),
      str(sorted({str(p.relative_to(sub)) for p in sub.rglob("*") if p.is_file()}
                 ^ set(FROZEN))))
check("the table is not a stub", len(FROZEN) == 8, str(len(FROZEN)))

# The tree hash is the real freeze: it is derived from the shipped bytes, so it catches an
# edit to any of the four files without needing a hash table per file.
from core import artifacts as AR
tree = {p.name: p.read_text() for p in cs.iterdir() if p.suffix in (".py", ".json")}
got = AR.tree_hash(tree)
check("the champion source re-hashes to the recorded tree", got == CHAMPION_TREE,
      f"{got[:16]} vs {CHAMPION_TREE[:16]}")
check("that tree also verifies inside the artifact store",
      AR.verify(CHAMPION_TREE, root=ROOT / "runs" / "artifacts").tree == CHAMPION_TREE)
man = json.loads((sub / "MANIFEST.json").read_text())
check("the manifest names the same champion", man["champion"]["tree"] == CHAMPION_TREE)
check("the manifest is bound to the promoting score, not a refit",
      man["signed_validation"].get("kind") == "score"
      and "NOT a refit" in man["binding"]["how"])
check("the manifest holds no absolute paths",
      "/Users/" not in json.dumps(man), "found a machine-specific path")
check("the submission has one row per held-out row",
      sum(1 for _ in (sub / "submission_test.csv").open()) - 1 == 170_588)

# ============================================================ 2. THE SEED PROFILES
from agent import candidate as CAND

def fns_of(path):
    t = ast.parse(Path(path).read_text())
    return {n.name for n in ast.walk(t) if isinstance(n, ast.FunctionDef)}

MIN = ROOT / "research_workspace_minimal"
INH = ROOT / "research_workspace"
check("both seed profiles exist", MIN.is_dir() and INH.is_dir())
# The set is PINNED so that adding a seed profile is a deliberate, reviewed change rather
# than a silent one. `accelerated` was added by the execution-profile work; it is optional,
# PyTorch-only, and must never be mistaken for the profile the submission was built from.
check("the profiles are registered",
      set(CAND.PROFILES) == {"inherited", "minimal", "accelerated"},
      str(sorted(CAND.PROFILES)))
check("the accelerated profile exists on disk",
      (ROOT / "research_workspace_accelerated").is_dir())
_ap = CAND.profile_provenance("accelerated")
check("the accelerated profile is NOT labelled the official NumPy seed",
      _ap["official_numpy_seed"] is False, str(_ap))
check("the frozen submission is unaffected by it: champion source has no torch import",
      not any("import torch" in (cs / f).read_text()
              for f in ("model.py", "train.py", "features.py") if (cs / f).is_file()))

min_fns = fns_of(MIN / "model.py") | fns_of(MIN / "train.py")
inh_fns = fns_of(INH / "model.py") | fns_of(INH / "train.py")
check("MINIMAL has no pairwise implementation",
      not ({"step_pairwise", "pairs_within_user"} & min_fns), str(sorted(min_fns)))
check("MINIMAL has no listwise implementation", "step_listwise" not in min_fns)
check("MINIMAL keeps the pointwise step", "step_pointwise" in min_fns)
check("INHERITED still has the pairwise switch (the recorded runs need it)",
      bool({"step_pairwise", "pairs_within_user"} & inh_fns))
check("MINIMAL defines only the baseline FM class",
      [n.name for n in ast.walk(ast.parse((MIN / "model.py").read_text()))
       if isinstance(n, ast.ClassDef)] == ["FM"])
check("MINIMAL refuses a non-pointwise loss in its own words",
      "MINIMAL seed implements only 'pointwise'" in (MIN / "train.py").read_text())
check("both profiles expose the SAME immutable contract",
      sha(MIN / "contract.py") == sha(INH / "contract.py"))
check("MINIMAL is agent-editable through all four files",
      set(CAND._contract_files(MIN)) == {"config.json", "features.py", "model.py",
                                        "train.py"})
pv_min = CAND.profile_provenance("minimal")
pv_inh = CAND.profile_provenance("inherited")
check("provenance reports MINIMAL as pairwise-free",
      pv_min["pairwise_code_present"] is False)
check("provenance reports INHERITED as containing pairwise",
      pv_inh["pairwise_code_present"] is True)
check("provenance reports no supplied alternative architecture in either",
      not pv_min["alternative_architecture_code_present"]
      and not pv_inh["alternative_architecture_code_present"])
check("provenance is derived by PARSING, not by grepping prose "
      "(MINIMAL's docstring mentions pairwise)",
      "pairwise" in (MIN / "train.py").read_text()
      and pv_min["pairwise_code_present"] is False)
check("the two profiles are different candidates",
      CAND.validate_tree(CAND.seed_tree(MIN), MIN)["tree"]
      != CAND.validate_tree(CAND.seed_tree(INH), INH)["tree"])

# ==================================================== 3. REPOSITORY CLAIMS AND HYGIENE
OMITTED = ["make_submission", "agent.report", "core.watchdog",
           "experiments.rationale_quality", "strategy.mle_strategy" ]
live = [p for p in list((ROOT / "agent").glob("*.py")) + list((ROOT / "core").glob("*.py"))
        + list((ROOT / "strategy").glob("*.py")) + list(ROOT.glob("*.py"))]
bad_imports = []
for p in live:
    try: t = ast.parse(p.read_text())
    except SyntaxError: continue
    mods = set()
    for n in ast.walk(t):
        if isinstance(n, ast.Import): mods |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module: mods.add(n.module)
    for m in mods:
        if m in ("make_submission", "agent.report", "core.watchdog",
                 "experiments.rationale_quality"):
            bad_imports.append(f"{p.name} -> {m}")
check("no shipped module imports an omitted superseded module",
      not bad_imports, str(bad_imports))
for m in ("make_submission.py", "agent/report.py", "core/watchdog.py"):
    check(f"omitted and absent: {m}", not (ROOT / m).exists())
check("the replacement helper module is present", (ROOT / "core/report_md.py").is_file())

# ---- stale current-task claims
STALE = {
    "NDCG@10": "superseded metric",
    "Recall@50": "superseded metric",
    "full-catalogue retrieval": "superseded framing",
    "full-catalog retrieval": "superseded framing",
    "independent 20-seed replication": "the 20-seed analysis is an EXPANSION, not independent",
    "agent implemented BPR": "the agent activated an inherited switch",
    "model.py was never edited": "model.py was edited once across the collection",
}
#: A stale phrase is permitted when the LINE labels it, or when the FILE opens with an
#: explicit supersession banner -- a historical document should not have to repeat the
#: disclaimer on every row of a table. Anything else is an unlabelled current-tense claim.
LABELS = ("supersede", "SUPERSEDE", "obsolete", "no longer", "NOT ", "not an",
          "incorrect", "was wrong", "stale", "previously", "earlier draft",
          "corrected", "rather than", "historical", "at the time",
          "not scored", "is not the", "superseded")
BANNERS = ("SUPERSEDED", "NO LONGER", "superseded task", "previous task definition",
           "obsolete")
offenders = []
for p in sorted(ROOT.rglob("*.md")):
    if "runs/" in str(p): continue
    txt = p.read_text()
    head = "\n".join(txt.splitlines()[:40])
    file_labelled = any(b in head for b in BANNERS)
    # A WINDOW of the line plus its neighbours, because markdown wraps: the word
    # "superseded" routinely lands on the line before the metric it qualifies, and a
    # single-line check reads that as an unlabelled claim. Checking one line was the same
    # mistake as a grep window that stops short of the field it is looking for.
    lines = txt.splitlines()
    for i, line in enumerate(lines):
        window = " ".join(lines[max(0, i - 1):i + 2])
        low = line.lower()
        for phrase in STALE:
            at = low.find(phrase.lower())
            if at < 0 or file_labelled or any(l in window for l in LABELS):
                continue
            # A negation immediately before the phrase labels it: "*not* full-catalogue
            # retrieval" is a correct current statement, and markdown emphasis means the
            # word "not" may arrive as "*not*". Checked positionally, within 30 characters.
            before = low[max(0, at - 30):at]
            if any(neg in before for neg in
                   ("not", "never", "no longer", "isn't", "rather than", "instead of")):
                continue
            offenders.append(f"{p.relative_to(ROOT)}: {phrase!r} :: {line.strip()[:90]}")
check("no unlabelled stale claim in any markdown", not offenders,
      "\n      ".join(offenders[:6]))

# ---- secret scan
SECRET_PAT = [
    (re.compile(r"sk-[A-Za-z0-9_\-]{20,}"), "openai-style key"),
    (re.compile(r"AIza[0-9A-Za-z_\-]{30,}"), "google api key"),
    (re.compile(r"\bghp_[A-Za-z0-9]{20,}"), "github token"),
    (re.compile(r"^[0-9a-f]{64}$"), "bare 64-hex (possible signing key)"),
]
leaks = []
for p in sorted(shipped_files()):
    rel = str(p.relative_to(ROOT))
    if rel.startswith(("runs/", "KuaiRand-Pure")) or p.suffix in (".npy", ".csv", ".npz"):
        continue
    try: txt = p.read_text()
    except (UnicodeDecodeError, OSError): continue
    for pat, what in SECRET_PAT:
        for line in txt.splitlines():
            if pat.search(line.strip()):
                leaks.append(f"{rel}: {what}")
                break
check("no credential-shaped string in the shipped tree", not leaks, str(leaks[:5]))
# Checked on the PATH and the suffix, not the filename. `secrets/scorer.key` has the name
# "scorer.key", so a filename-only test missed exactly the file that matters most -- which
# `python3 -m core.scorer keygen` would create right here.
_bad = []
for q in shipped_files():
    rel = q.relative_to(ROOT)
    if q.name == ".env.example":
        continue
    if ("secrets" in rel.parts or q.name == ".env" or q.name.startswith(".env.")
            or q.suffix in (".key", ".pem")):
        _bad.append(str(rel))
check("no key or .env material anywhere in the tree, by PATH not filename",
      not _bad, str(_bad))
check("that check would catch secrets/scorer.key",
      "secrets" in Path("secrets/scorer.key").parts
      and Path("secrets/scorer.key").suffix == ".key")
check("the dataset itself is not shipped (a symlink is not a copy)",
      not (ROOT / "KuaiRand-Pure").is_dir() or (ROOT / "KuaiRand-Pure").is_symlink())

# ---- the 20-seed disclosure must be accurate AND ledger-derived
from core.seed_stats import champion_report, cross_check
L = ROOT / "runs" / "ledgers" / "research_trials.jsonl"
BASE = [p.name for p in (ROOT / "runs" / "artifacts").iterdir()
        if p.name.startswith("e1d994dedd11")][0]
rep = champion_report(L, CHAMPION_TREE, BASE)
check("the 5-seed original is derived from the ledger", rep["original_5"]["n"] == 5)
check("the 20-seed expansion is derived from the ledger", rep["expanded_20"]["n"] == 20)
check("the newly observed subset 5-19 is derived from the ledger",
      rep["newly_observed_5_19"]["n"] == 15 and rep["newly_observed_5_19"]["seeds"][0] == 5)
check("the disclosure says EXPANDED, not independent replication",
      "EXPANDED" in rep["disclosure"] and "not an independent" in rep["disclosure"])
check("the disclosure states seed 0 was used in selection",
      "selected on" in rep["disclosure"])
cc = cross_check(L, CHAMPION_TREE, BASE)
check("every per-seed value matches a signed score entry", cc["ok"],
      str(cc["unmatched"][:3]))
check("seed statistics are computed, not hard-coded constants",
      "0.001888" not in (ROOT / "core/seed_stats.py").read_text())

# ---- the confirmation CLI must be able to name GENUINELY UNSEEN seeds
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_cc", ROOT / "confirm_champion.py")
_cc = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_cc)
_bp = _cc.build_parser()
check("confirm_champion offers --seed-start",
      any(a.dest == "seed_start" for a in _bp._actions))
check("confirm_champion offers --seed-list",
      any(a.dest == "seed_list" for a in _bp._actions))

def _resolve(argv):
    a = _bp.parse_args(["--state", "x"] + argv)
    if a.seed_list:
        return sorted(int(x) for x in a.seed_list.replace(",", " ").split()), "seed_list"
    return list(range(a.seed_start, a.seed_start + a.seeds)), "range"

check("--seeds alone still means 0..n-1", _resolve(["--seeds", "5"])[0] == [0, 1, 2, 3, 4])
check("--seed-start moves past what has been measured",
      _resolve(["--seed-start", "20", "--seeds", "10"])[0] == list(range(20, 30)))
check("--seed-list names an explicit set",
      _resolve(["--seed-list", "20,21,22"]) == ([20, 21, 22], "seed_list"))
check("--seed-list sorts and tolerates spaces",
      _resolve(["--seed-list", "31, 7, 19"])[0] == [7, 19, 31])
check("seeds 20+ are genuinely unseen for the shipped champion",
      not (set(range(20, 30)) & set(rep["expanded_20"]["seeds"])),
      str(sorted(set(range(20, 30)) & set(rep["expanded_20"]["seeds"]))))
check("RUN.md documents unseen seeds as starting past the measured range",
      "--seed-start 20" in (ROOT / "RUN.md").read_text()
      and "already measured" in (ROOT / "RUN.md").read_text())
check("RUN.md no longer presents --seeds 20 as unseen confirmation",
      "--seeds 20 --baseline-tree" not in (ROOT / "RUN.md").read_text())
_src = (ROOT / "confirm_champion.py").read_text()
check("a repeated seed is refused", "repeats a seed" in _src)
check("the resolved seed list is recorded before fitting",
      "\"seeds\": seeds" in _src and "confirmation_protocol" in _src)
check("the protocol records HOW the seeds were chosen", "seed_selection" in _src)

# ---- every hash quoted in the docs must resolve to a real artifact
import re as _re
arts = {q.name for q in (ROOT / "runs" / "artifacts").iterdir()}
unresolved = []
for f in ("README.md", "RUN.md", "CLEANUP_AUDIT.md", "ARCHITECTURE.md"):
    fp = ROOT / f
    if not fp.is_file(): continue
    for h in set(_re.findall(r"\b[0-9a-f]{64}\b", fp.read_text())):
        if h not in arts:
            unresolved.append(f"{f}:{h[:16]}")
check("every 64-hex tree hash in the docs resolves to a shipped artifact",
      not unresolved, str(unresolved))
check("the docs quote at least one real tree hash", any(
    _re.search(r"\b[0-9a-f]{64}\b", (ROOT / f).read_text())
    for f in ("README.md", "RUN.md") if (ROOT / f).is_file()))

print()
bad = [n for n, ok in R if not ok]
print(f"{len(R)-len(bad)}/{len(R)} passed" + (f"  FAILURES: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
