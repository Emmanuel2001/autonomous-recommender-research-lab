"""Harness side of a code-first CANDIDATE: validate a tree, run it sandboxed, report.

Relationship to agent/patch.py
------------------------------
patch.py handles the single-module ProposePatch action and stays as it is, because the
typed driver still uses it. This module is its tree-shaped successor and deliberately
REUSES patch.py's `DISCOURAGED_IMPORTS`, `strip_comments` and dependency plumbing rather
than restating them -- one place to fix, and the same fast feedback the agent already gets.

WHAT IS DIFFERENT, AND WHY
--------------------------
* A tree, not a file. `train.py` may import `features` and `model`, so an architecture
  change and the feature change it requires are ONE candidate.
* `contract.py` cannot be shadowed. A tree containing it is refused, as is a module name
  that would shadow the stdlib, numpy, or a repo package -- the candidate directory goes
  first on sys.path, so `json.py` in a tree would otherwise break the runner itself.
* Computing the METRIC is allowed. patch.py refuses source mentioning GAUC, on the
  reasoning that a strategy computing its own score grades its own homework. That rule
  also made early stopping on the public validation split impossible, which is ordinary
  practice and what the official baseline does. The labels that decide the competition are
  not in the process at all, and the number that counts is the scorer's signed one, so the
  metric is exposed through `contract.primary` and the refusal is dropped -- knowingly.

The static checks here are FAST FEEDBACK, not the security boundary. core/isolation.py is.
"""
from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

from core import artifacts as AR

ROOT = Path(__file__).resolve().parents[1]
#: The default seed profile. A run may select another with `--seed-profile`; every function
#: below takes an explicit `workspace` so nothing silently mixes two profiles' contracts.
WORKSPACE = ROOT / "research_workspace"
WORKSPACE_MINIMAL = ROOT / "research_workspace_minimal"
PROFILES = {"inherited": WORKSPACE, "minimal": WORKSPACE_MINIMAL}


def workspace_for(profile: str) -> Path:
    if profile not in PROFILES:
        raise CandidateError(
            f"unknown seed profile {profile!r}; choose from {sorted(PROFILES)}")
    w = PROFILES[profile]
    if not (w / "contract.py").is_file():
        raise CandidateError(f"seed profile {profile!r} has no contract.py at {w}")
    return w


def profile_provenance(profile: str) -> dict:
    """What the seed ALREADY contains, recorded in run_start so an autonomy claim can be
    checked rather than taken on trust.

    Two precisions learned by getting it wrong:
      * detection is by PARSED function name, never by grepping prose -- the minimal
        profile's docstring explains that pairwise is absent, and a substring search reads
        that as presence;
      * only `model.py` and `train.py` are inspected. Scanning the whole workspace flagged
        `Data.columns` (a contract property) as an "alternative architecture".
    """
    import ast as _ast
    w = workspace_for(profile)

    def defs(fname):
        f = w / fname
        if not f.is_file():
            return set(), set()
        t = _ast.parse(f.read_text())
        fns = {n.name for n in _ast.walk(t) if isinstance(n, _ast.FunctionDef)}
        cls = {n.name for n in _ast.walk(t) if isinstance(n, _ast.ClassDef)}
        return fns, cls

    mfns, mcls = defs("model.py")
    tfns, _ = defs("train.py")
    all_fns = mfns | tfns

    pairwise = bool({"step_pairwise", "pairs_within_user"} & all_fns)
    listwise = "step_listwise" in all_fns
    #: the baseline FM's own surface. Anything else DEFINED IN model.py is a supplied
    #: alternative the agent did not have to write.
    BASELINE_FM = {"__init__", "logits", "predict", "_apply", "step_pointwise",
                   "state", "load", "sigmoid"}
    extra_model = sorted(mfns - BASELINE_FM - {"step_pairwise", "step_listwise"})
    extra_classes = sorted(mcls - {"FM"})
    return {"seed_profile": profile, "workspace": w.name,
            "pairwise_code_present": pairwise,
            "listwise_code_present": listwise,
            "alternative_architecture_code_present": bool(extra_model or extra_classes),
            "model_classes": sorted(mcls),
            "extra_model_functions": extra_model,
            "extra_model_classes": extra_classes,
            "model_py_functions": sorted(mfns),
            "train_py_functions": sorted(tfns)}


#: Reused verbatim from the single-module path.
from agent.patch import DISCOURAGED_IMPORTS, strip_comments   # noqa: E402

#: patch.py's list minus the metric rule (see the module docstring). Everything here is
#: also denied by the kernel; naming it cheaply tells the agent WHY.
CANDIDATE_PATTERNS = [
    (p, why) for p, why in __import__("agent.patch", fromlist=["x"]).FORBIDDEN_PATTERNS
    if "the metric belongs to the scorer" not in why
]

ENTRY_FILE = "train.py"
ENTRY_ARGS = 2                                   # fit_predict(data, params)


class CandidateError(ValueError):
    """A candidate tree that will not be run. Cheap, recorded, and always explains itself."""


def _contract_files(workspace: Path | None = None) -> tuple[str, ...]:
    """MUTABLE_FILES from the SELECTED profile's contract, loaded from its own file so the
    two profiles can never share one contract by import accident."""
    w = Path(workspace or WORKSPACE)
    import ast as _ast
    tree = _ast.parse((w / "contract.py").read_text())
    for node in tree.body:
        if isinstance(node, _ast.Assign) and any(
                getattr(t, "id", "") == "MUTABLE_FILES" for t in node.targets):
            return tuple(_ast.literal_eval(node.value))
    raise CandidateError(f"{w}/contract.py defines no MUTABLE_FILES")


def contract_sha256(workspace: Path | None = None) -> str:
    """Hash of the immutable contract, recorded beside every candidate's tree hash so a
    mid-run contract change is a visible fact rather than an argument."""
    import hashlib
    w = Path(workspace or WORKSPACE)
    return hashlib.sha256((w / "contract.py").read_bytes()).hexdigest()


def _reserved_module_names() -> set[str]:
    """Names a candidate module may not take, because the candidate dir is first on
    sys.path and would shadow them inside the runner."""
    std = set(getattr(sys, "stdlib_module_names", ()))
    repo = {p.stem for p in ROOT.iterdir()
            if p.is_dir() and (p / "__init__.py").exists() or p.suffix == ".py"}
    return std | repo | {"contract", "numpy", "np", "scipy", "pandas", "sklearn", "torch"}


def validate_tree(files: dict, workspace: Path | None = None) -> dict:
    """Static checks over a whole tree. Raises CandidateError with an actionable reason."""
    mutable = _contract_files(workspace)
    try:
        canon = AR.canon_tree(files)
    except AR.ArtifactError as e:
        raise CandidateError(str(e)) from None

    missing = [f for f in mutable if f not in canon]
    if missing:
        raise CandidateError(
            f"the tree is missing required files {missing}. A candidate consists of "
            f"{list(mutable)}; return every one, even if unchanged.")

    reserved = _reserved_module_names()
    for rel in sorted(canon):
        if "/" in rel:
            raise CandidateError(
                f"{rel!r}: candidate files are flat -- no subdirectories, because the "
                f"candidate directory is imported directly.")
        if not rel.endswith((".py", ".json")):
            raise CandidateError(f"{rel!r}: a candidate holds only .py and .json files")
        if rel == "contract.py":
            raise CandidateError(
                "contract.py is immutable and is supplied by the harness; a tree "
                "containing it would shadow the real contract. Edit train.py instead.")
        if rel.endswith(".py") and rel not in mutable and rel[:-3] in reserved:
            raise CandidateError(
                f"{rel!r} would shadow the module {rel[:-3]!r} that the runner imports. "
                f"Pick another name.")

    # config.json must parse, or every fit fails on the same line for no reason.
    try:
        cfgd = json.loads(canon["config.json"])
    except json.JSONDecodeError as e:
        raise CandidateError(f"config.json is not valid JSON: {e}") from None
    if not isinstance(cfgd, dict):
        raise CandidateError(f"config.json must be an object, got {type(cfgd).__name__}")

    imports: set[str] = set()
    functions: dict[str, set[str]] = {}
    for rel in sorted(canon):
        if not rel.endswith(".py"):
            continue
        src = canon[rel]
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            raise CandidateError(
                f"{rel} line {e.lineno}: syntax error: {e.msg}") from None
        fns = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        functions[rel] = fns
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imports |= {al.name.split(".")[0] for al in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module:
                imports.add(n.module.split(".")[0])

    if "fit_predict" not in functions.get(ENTRY_FILE, set()):
        raise CandidateError(
            f"{ENTRY_FILE} must define fit_predict(data, params). Found: "
            f"{sorted(functions.get(ENTRY_FILE, [])) or 'no functions'}. See contract.py.")
    for n in ast.walk(ast.parse(canon[ENTRY_FILE])):
        if isinstance(n, ast.FunctionDef) and n.name == "fit_predict":
            args = [x.arg for x in n.args.args]
            if len(args) != ENTRY_ARGS:
                raise CandidateError(
                    f"fit_predict takes exactly {ENTRY_ARGS} arguments (data, params); "
                    f"got {args}. `data` carries train/dev/heldout/aux/edges/spec.")

    for mod, why in DISCOURAGED_IMPORTS.items():
        if mod in imports:
            raise CandidateError(f"do not import {mod!r}: {why}")

    joined = "\n".join(strip_comments(canon[r]) for r in sorted(canon) if r.endswith(".py"))
    for pat, why in CANDIDATE_PATTERNS:
        m = pat.search(joined)
        if m:
            raise CandidateError(f"the tree references {m.group(0)!r}: {why}")

    return {"tree": AR.tree_hash(canon, already_canonical=True),
            "contract_sha256": contract_sha256(workspace),
            "bytes": sum(len(v.encode()) for v in canon.values()),
            "files": sorted(canon), "imports": sorted(imports),
            "config": cfgd}


def seed_tree(workspace: Path | None = None) -> dict:
    """The starting candidate: the selected profile's workspace, as it stands on disk."""
    w = Path(workspace or WORKSPACE)
    return AR.read_dir(w, include=_contract_files(w))


def run_candidate(files: dict, params: dict | None = None, *, run_dir: Path,
                  task: str = "kuairand_pure", backend: str = "seatbelt",
                  max_seconds: float = 900.0,
                  artifact_root: Path = AR.ARTIFACTS,
                  workspace: Path | None = None) -> dict:
    """Store, materialise, and run one candidate behind the sandbox.

    Never raises on a candidate that fails: a bad candidate is DATA about the agent, and
    the loop must survive it. Only a refused tree (CandidateError) and an unbuildable
    sandbox (IsolationError, returned as data) are distinguished.
    """
    from core import isolation as I

    ws = Path(workspace or WORKSPACE)
    meta = validate_tree(files, ws)                   # cheap refusal before any compute
    art = AR.put(files, root=artifact_root,
                 meta={"contract_sha256": meta["contract_sha256"], "task": task})

    # Dependencies resolve OUTSIDE the sandbox -- it has no network and never will.
    from agent import deps as D
    src_all = "\n".join(files[k] for k in sorted(files) if k.endswith(".py"))
    declared = D.parse_requires(src_all)
    try:
        dep_report = D.resolve(declared)
    except D.DependencyError as e:
        raise CandidateError(f"dependency refused: {e}") from None
    meta["dependencies"] = dep_report
    if dep_report["failed"]:
        return {"ok": False, "tree": art.tree, "error_type": "DependencyInstallFailed",
                "error": f"declared dependencies could not be provided: "
                         f"{dep_report['failed']}",
                "contract_violation": False, "static": meta}

    # resolve(): the sandbox profile is built from this path and seatbelt matches only
    # absolute ones. A relative run_dir produced a profile that granted nothing.
    run_dir = Path(run_dir).resolve()
    cand_dir = run_dir / "source"
    run_dir.mkdir(parents=True, exist_ok=True)
    AR.materialise(art.tree, cand_dir, root=artifact_root)

    # out_dir is THIS candidate's directory, so the sandbox's write grant is scoped to it.
    # The default (runs/iso) let any candidate write over a sibling's scores.
    sb = I.make_sandbox(backend, out_dir=run_dir)
    try:
        env_extra = {}
        if dep_report.get("pythonpath"):
            existing = os.environ.get("PYTHONPATH", "")
            env_extra["PYTHONPATH"] = (dep_report["pythonpath"]
                                       + (os.pathsep + existing if existing else ""))
        report = sb.run(["-m", "strategy.candidate_runner",
                         "--candidate", str(cand_dir),
                         "--workspace", str(ws),
                         "--task", task,
                         "--params", json.dumps(params or {}),
                         "--out-dir", str(run_dir),
                         "--max-seconds", str(max_seconds)],
                        timeout=max_seconds + 60, env_extra=env_extra)
    except I.IsolationError as e:
        return {"ok": False, "tree": art.tree, "error_type": "IsolationError",
                "error": str(e)[:900], "contract_violation": False, "static": meta}

    report["tree"] = art.tree
    report["static"] = meta
    report["dependencies"] = dep_report
    report["isolation"] = sb.backend
    report["run_dir"] = str(run_dir)
    report["dev_scores"] = str(run_dir / "dev_scores.npy")
    report["heldout_scores"] = str(run_dir / "heldout_scores.npy")
    report["train_subset_scores"] = str(run_dir / "train_subset_scores.npy")
    report["train_subset_index"] = str(run_dir / "train_subset_index.npy")
    return report
