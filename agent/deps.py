"""Dependencies a generated patch may declare, and where they are installed.

THE CONSTRAINT THAT SHAPES THIS
-------------------------------
The sandbox denies network. So a patch can never install anything itself -- installation
must happen OUTSIDE the sandbox, before the sandboxed run, in the harness. That is not a
limitation to work around; it is the correct order. A process that can install packages
can fetch a label set.

WHY AN ALLOWLIST AND NOT `pip install <whatever the model wrote>`
-----------------------------------------------------------------
The declaration comes from an LLM. Passing it to pip is remote code execution with the
package name chosen by the model: a typo lands on a typosquat, and a hallucinated name may
be a real malicious package. So names are matched against a fixed list, exactly, and
anything else is refused and recorded. The allowlist is short on purpose.

WHERE THINGS GO
---------------
`pip install --target runs/patch_libs`, never into the user's environment, and the
directory is added to the sandboxed child's PYTHONPATH. Two consequences worth stating:
  * running this cannot break the interpreter the harness itself needs;
  * the child only READS that directory, which the sandbox permits (it denies writes
    outside the patch output dir, and specific reads, not reads in general).

Anything already importable is used as-is and not reinstalled -- which is how `torch`
worked before this module existed. That was luck; this makes it a decision.
"""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "runs" / "patch_libs"

#: Declaration syntax, one name per directive:  `# requires: lightgbm`
REQUIRES_RE = re.compile(r"^\s*#\s*requires\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
#: A package name, and nothing else. No URLs, no VCS refs, no version specifiers, no
#: extras -- each of those is a way to smuggle something past a name-based allowlist.
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,40}$")

#: name -> (import name, why it is allowed). Short on purpose.
ALLOWLIST: dict[str, tuple[str, str]] = {
    "numpy":        ("numpy", "the baseline's only dependency"),
    "scipy":        ("scipy", "sparse matrices and linear algebra"),
    "pandas":       ("pandas", "dataframes; the kit deliberately avoids it, patches may not"),
    "scikit-learn": ("sklearn", "standard models and metrics"),
    "lightgbm":     ("lightgbm", "gradient boosting, a plausible ranker for this task"),
    "xgboost":      ("xgboost", "gradient boosting"),
    "torch":        ("torch", "neural rankers; the organisers' own reference used it"),
    "implicit":     ("implicit", "implicit-feedback matrix factorisation"),
}

MAX_DECLARATIONS = 4
INSTALL_TIMEOUT = 900


class DependencyError(ValueError):
    """A declaration that will not be honoured. Always explains itself."""


def parse_requires(src: str) -> list[str]:
    """Extract declared dependency names, in order, deduplicated.

    Only the directive form is recognised. An `import lightgbm` with no declaration is
    NOT auto-installed: a patch must say what it needs, so the ledger records an
    intention rather than the harness guessing one.
    """
    out: list[str] = []
    for raw in REQUIRES_RE.findall(src):
        for part in re.split(r"[,\s]+", raw.strip()):
            if part and part not in out:
                out.append(part)
    return out


def _importable(mod: str, extra_path: Path | None = None) -> bool:
    if extra_path is not None and str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _installed_version(mod: str) -> str | None:
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:                                     # pragma: no cover
        return None
    try:
        return version(mod)
    except Exception:
        return None


def validate_declarations(names: list[str]) -> None:
    """Refuse anything the allowlist does not name, before touching pip."""
    if len(names) > MAX_DECLARATIONS:
        raise DependencyError(
            f"{len(names)} dependencies declared; the cap is {MAX_DECLARATIONS}. A patch "
            f"needing more than that is probably doing several things at once.")
    for n in names:
        if not NAME_RE.match(n):
            raise DependencyError(
                f"{n!r} is not a bare package name. Version specifiers, URLs, VCS refs "
                f"and extras are all refused -- each is a way past a name-based "
                f"allowlist. Declare just the name.")
        if n not in ALLOWLIST:
            raise DependencyError(
                f"{n!r} is not on the allowlist. Permitted: {sorted(ALLOWLIST)}. The "
                f"declaration comes from a language model, so passing it to pip "
                f"unchecked would be remote code execution with the package name chosen "
                f"by the model.")


def resolve(names: list[str], *, install: bool = True) -> dict:
    """Ensure each declared dependency is importable by the sandboxed child.

    Returns a report for the ledger. Raises DependencyError only for a refused
    declaration; an install FAILURE is reported, not raised, so the caller can decline to
    run the patch rather than run it without the library it asked for.
    """
    validate_declarations(names)
    report = {"declared": list(names), "already_present": [], "installed": [],
              "failed": [], "target": str(TARGET), "pythonpath": None}
    if not names:
        return report

    need = []
    for n in names:
        mod = ALLOWLIST[n][0]
        if _importable(mod):
            report["already_present"].append(
                {"name": n, "import": mod, "version": _installed_version(n)})
        elif _importable(mod, TARGET):
            report["already_present"].append(
                {"name": n, "import": mod, "version": None, "from": "patch_libs"})
        else:
            need.append(n)

    if need and install:
        TARGET.mkdir(parents=True, exist_ok=True)
        for n in need:
            cmd = [sys.executable, "-m", "pip", "install", "--no-input",
                   "--disable-pip-version-check", "--target", str(TARGET), n]
            try:
                p = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=INSTALL_TIMEOUT)
            except subprocess.TimeoutExpired:
                report["failed"].append({"name": n, "error": "install timed out"})
                continue
            if p.returncode == 0 and _importable(ALLOWLIST[n][0], TARGET):
                report["installed"].append({"name": n, "import": ALLOWLIST[n][0]})
            else:
                tail = (p.stderr or p.stdout or "").strip().splitlines()[-3:]
                report["failed"].append({"name": n, "returncode": p.returncode,
                                         "error": " | ".join(tail)[:400]})
    elif need:
        report["failed"] = [{"name": n, "error": "not present and install disabled"}
                            for n in need]

    if TARGET.exists():
        report["pythonpath"] = str(TARGET)
    return report
