"""Harness side of ProposePatch: validate, hash, run sandboxed, report.

DIVISION OF LABOUR -- worth being exact about, because it is easy to overclaim
----------------------------------------------------------------------------
    static checks here     FAST FEEDBACK. They cost microseconds and turn a 30-second
                           failed fit into an immediate, specific error the agent can
                           act on. They are NOT the security boundary.
    the sandbox            SECURITY. Reads of the held-out labels, the ledger, the
                           signing key and `.env` are denied by the kernel; there is no
                           network; writes go nowhere but the patch's own output dir;
                           and the policy is inherited across fork/exec. Measured, not
                           asserted: `python3 -m core.isolation probe`.

So a static check that a patch does not `import socket` is a courtesy to the agent, not
a defence. If the static checks were the defence this feature would be indefensible.

WHY PATCHES ARE HASHED
----------------------
The typed action space dedups by canonical config hash, which is what makes a repeated
proposal free instead of a 30-second refit (measured: 5 free rejections in one run).
Patches get the same treatment via a hash of their normalised source. Two textually
different patches that happen to be semantically identical will both run -- that is
accepted, and stated, rather than papered over with an equivalence check we cannot make
sound.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MAX_PATCH_BYTES = 24_000
REQUIRED_FN = "fit_predict"

#: Modules a strategy has no business importing. The sandbox already stops the damage;
#: this exists so the agent is told immediately and cheaply, with the reason.
DISCOURAGED_IMPORTS = {
    "socket": "no network is available in the sandbox, and a model that can reach the "
              "internet can fetch a label set",
    "urllib": "no network is available in the sandbox",
    "requests": "no network is available in the sandbox",
    "httpx": "no network is available in the sandbox",
    "subprocess": "the sandbox is inherited by child processes, so this cannot escape -- "
                  "but a strategy has no reason to spawn one",
    "shutil": "the strategy may only write to its own output directory",
    "ctypes": "calling native code directly sidesteps every Python-level assumption "
              "this harness makes about what a strategy does; the kernel sandbox still "
              "holds, but nothing above it would",
    "pickle": "loading a pickle is arbitrary code execution; build the model in code",
}
#: Paths a patch must not name. It cannot read them anyway; naming one is a sign the
#: patch is trying to score itself rather than model anything.
FORBIDDEN_PATTERNS = [
    (re.compile(r"log_standard_4_22_to_5_08"), "that file holds the eval-window labels"),
    (re.compile(r"secrets/|scorer\.key"), "the signing key is not available"),
    (re.compile(r"ledger.*\.jsonl|\.jsonl"), "the ledger holds every score already "
                                             "measured; reading it would let the model "
                                             "hill-climb on validation without the gate"),
    (re.compile(r"official_evaluate|core\.metrics|\bGAUC\b|ndcg_at_k"),
     "the metric belongs to the scorer. A strategy that computes its own score is "
     "grading its own homework -- return raw scores and let the scorer rank them"),
    (re.compile(r"\bheldout\.label\b|heldout\[.label.\]"),
     "held-out labels are None by construction; using them is the one thing that "
     "invalidates the submission. Build held-out features and user-group sizes without "
     "a label, and restrict labels and evaluation sets to train and dev"),
]


class PatchError(ValueError):
    """A patch that will not be run. Cheap, recorded, and always explains itself."""


def strip_comments(src: str) -> str:
    """Remove comments, keep string literals.

    The pattern checks below scan source text, and a COMMENT explaining that
    `heldout.label is None` is prose, not an attempt to use it -- the minimal example we
    show the model was refused by its own docstring until this existed.

    String literals are deliberately KEPT, because a filename appears as a string
    (`open("log_standard_...")`) and that is exactly the reference worth flagging.

    None of this is a security boundary: a patch can trivially evade a source scan with
    getattr(heldout, "lab" + "el"). The sandbox is what stops the damage; this is fast,
    specific feedback so the agent does not burn 30 seconds learning the same thing.
    """
    import io
    import tokenize
    lines = src.splitlines(keepends=True)
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return src                              # malformed: ast.parse will report it
    # Blank comments IN PLACE. Rebuilding by joining tokens with spaces turned
    # `heldout.label` into `heldout . label` and silently disabled every pattern below,
    # which is worse than the false positive it was fixing.
    for tok in toks:
        if tok.type != tokenize.COMMENT:
            continue
        (r1, c1), (r2, c2) = tok.start, tok.end
        if r1 != r2 or r1 - 1 >= len(lines):
            continue
        line = lines[r1 - 1]
        lines[r1 - 1] = line[:c1] + " " * (c2 - c1) + line[c2:]
    return "".join(lines)


def normalise(src: str) -> str:
    """Strip trailing whitespace and blank-line noise for hashing. Deliberately NOT a
    semantic normalisation -- see the module docstring."""
    return "\n".join(l.rstrip() for l in src.strip().splitlines() if l.strip())


def patch_hash(src: str) -> str:
    return hashlib.sha256(normalise(src).encode("utf-8")).hexdigest()


def validate(src: str) -> dict:
    """Static checks. Raises PatchError with a reason the agent can act on."""
    if not isinstance(src, str) or not src.strip():
        raise PatchError("patch code is empty")
    raw = src.encode("utf-8")
    if len(raw) > MAX_PATCH_BYTES:
        raise PatchError(f"patch is {len(raw):,} bytes; the cap is {MAX_PATCH_BYTES:,}. "
                         f"A strategy this large is probably doing several things at "
                         f"once -- propose one change.")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise PatchError(f"syntax error on line {e.lineno}: {e.msg}") from None

    fns = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if REQUIRED_FN not in fns:
        raise PatchError(
            f"the patch must define {REQUIRED_FN}(train, dev, heldout, params). "
            f"Found: {sorted(fns) or 'no functions'}. See strategy/patch_runner.py.")
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == REQUIRED_FN:
            args = [a.arg for a in n.args.args]
            if len(args) != 4:
                raise PatchError(
                    f"{REQUIRED_FN} takes exactly 4 arguments "
                    f"(train, dev, heldout, params); got {args}")

    imports = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imports |= {al.name.split(".")[0] for al in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            imports.add(n.module.split(".")[0])
    for mod, why in DISCOURAGED_IMPORTS.items():
        if mod in imports:
            raise PatchError(f"do not import {mod!r}: {why}")

    code_only = strip_comments(src)
    for pat, why in FORBIDDEN_PATTERNS:
        m = pat.search(code_only)
        if m:
            raise PatchError(f"patch references {m.group(0)!r}: {why}")

    return {"bytes": len(raw), "imports": sorted(imports), "functions": sorted(fns),
            "patch_hash": patch_hash(src)}


def run_sandboxed(src: str, params: dict, *, task: str = "kuairand_pure",
                  backend: str = "seatbelt", max_seconds: float = 900.0) -> dict:
    """Write the patch, run it behind the sandbox, return the runner's report.

    Never raises on a failing patch: a bad patch is DATA about the agent, and the loop
    must survive it. Only an isolation failure (a sandbox that cannot be built) raises.
    """
    from core import isolation as I

    meta = validate(src)                     # cheap refusal before any compute

    # Declared dependencies are resolved OUTSIDE the sandbox, before it is entered --
    # the sandbox has no network and never will. A refused declaration is a PatchError
    # (the agent asked for something not permitted); a failed INSTALL is not, because
    # that is infrastructure, and the honest response is to decline the patch rather than
    # run it without the library it said it needed.
    from agent import deps as D
    declared = D.parse_requires(src)
    try:
        dep_report = D.resolve(declared)
    except D.DependencyError as e:
        raise PatchError(f"dependency refused: {e}") from None
    meta["dependencies"] = dep_report
    if dep_report["failed"]:
        return {"ok": False, "patch_hash": meta["patch_hash"],
                "error_type": "DependencyInstallFailed",
                "error": f"declared dependencies could not be provided: "
                         f"{dep_report['failed']}",
                "contract_violation": False, "static": meta}

    h = meta["patch_hash"][:12]
    pdir = I.ISO_OUT / f"patch_{h}"
    pdir.mkdir(parents=True, exist_ok=True)
    pfile = pdir / "patch.py"
    pfile.write_text(src)

    # out_dir=pdir, NOT the default ISO_OUT: the write grant follows out_dir, so the
    # default let any patch write into runs/iso -- i.e. into a sibling candidate's
    # directory, over its scores. Narrowed to this candidate's own directory.
    sb = I.make_sandbox(backend, out_dir=pdir)
    try:
        env_extra = {}
        if dep_report.get("pythonpath"):
            existing = os.environ.get("PYTHONPATH", "")
            env_extra["PYTHONPATH"] = (dep_report["pythonpath"]
                                       + (os.pathsep + existing if existing else ""))
        report = sb.run(["-m", "strategy.patch_runner",
                         "--patch", str(pfile), "--task", task,
                         "--params", json.dumps(params or {}),
                         "--out-dir", str(pdir),
                         "--max-seconds", str(max_seconds)],
                        timeout=max_seconds + 60, env_extra=env_extra)
    except I.IsolationError as e:
        return {"ok": False, "patch_hash": meta["patch_hash"],
                "error_type": "IsolationError", "error": str(e)[:900],
                "contract_violation": False, "static": meta}
    report["patch_hash"] = meta["patch_hash"]
    report["static"] = meta
    report["dependencies"] = dep_report
    report["dev_scores"] = str(pdir / "dev_scores.npy")
    report["heldout_scores"] = str(pdir / "heldout_scores.npy")
    report["isolation"] = sb.backend
    return report
