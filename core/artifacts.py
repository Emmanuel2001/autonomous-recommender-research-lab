"""Immutable, content-addressed store for candidate source trees.

WHY THE SOURCE IS NOT IN THE LEDGER
-----------------------------------
`core/scorer.py` caps a signed record at `MAX_RECORD_BYTES` (12,288) and refuses anything
larger. `agent/patch.py` permits a strategy up to `MAX_PATCH_BYTES` (24,000). So a legal
candidate cannot fit inside the record that is supposed to attest to it -- and in the
typed driver that refusal is SILENT (`stats["unrecorded"]` increments and the run carries
on), which is the worst of both worlds: a scored experiment with no pre-registration.

The resolution is not a bigger cap. The ledger records **hashes and a bounded diff**; the
bytes live here, addressed by the hash the ledger names. `verify()` recomputes the hash
from the stored bytes, so the signed chain covers the source without carrying it.

WHAT "CONTENT-ADDRESSED" BUYS
----------------------------
Two candidates with the same source are the same artifact, so a repeated idea is free
rather than a refit. A candidate always names its parent's hash, so lineage is a fact
about the data rather than a field someone remembered to set. And an artifact directory is
written once, atomically, and never mutated -- run OUTPUTS (diffs, logs, scores) live under
`runs/candidates/<run>/<iter>/`, deliberately not in here, so "immutable" needs no asterisk.

CANONICALISATION IS TEXTUAL, NOT SEMANTIC
-----------------------------------------
Files are canonicalised on the way in (CRLF -> LF, trailing whitespace per line removed,
exactly one final newline) and the canonical bytes are what gets both hashed and stored.
Two trees differing only in line endings or trailing blanks are therefore ONE artifact.
Two trees that are semantically identical but textually different are TWO -- the same
honest limit `agent/patch.py` states for its own hash. We do not claim an equivalence
check we cannot make sound.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "runs" / "artifacts"

#: Bumped only if the canonical serialisation below changes, which would invalidate every
#: previously recorded hash. Part of the hash input so old and new can never collide.
TREE_VERSION = b"track2-tree-v1\n"

#: A candidate tree is source, not a data dump. Both bounds exist so a runaway generation
#: fails here -- cheaply, with a readable reason -- rather than after a 30-second fit.
MAX_FILE_BYTES = 64_000
MAX_TREE_FILES = 32
MAX_TREE_BYTES = 256_000


class ArtifactError(ValueError):
    """A tree that will not be stored. Always explains itself."""


# ------------------------------------------------------------------ path safety
def safe_relpath(p: str) -> str:
    """Return `p` as a normalised relative POSIX path, or raise.

    Generated code names its own files, so this is the boundary that keeps a candidate
    inside its own tree. Absolute paths, `..`, drive letters, NUL bytes and empty
    components are all refused rather than normalised away -- silently rewriting a
    traversal into something harmless would hide the attempt.
    """
    if not isinstance(p, str) or not p.strip():
        raise ArtifactError("empty path in tree")
    if "\x00" in p:
        raise ArtifactError(f"path contains a NUL byte: {p!r}")
    if "\\" in p:
        raise ArtifactError(f"use forward slashes in tree paths, got {p!r}")
    pp = PurePosixPathCompat(p)
    if pp.is_absolute:
        raise ArtifactError(f"tree paths must be relative, got {p!r}")
    parts = [c for c in p.split("/") if c != ""]
    if not parts:
        raise ArtifactError(f"path resolves to nothing: {p!r}")
    for c in parts:
        if c in (".", ".."):
            raise ArtifactError(
                f"path {p!r} contains {c!r}; a candidate may only name files inside "
                f"its own tree")
    return "/".join(parts)


class PurePosixPathCompat:
    """Absolute-path test without importing pathlib flavours.

    `Path("C:/x").is_absolute()` is False on POSIX, so a Windows-style absolute path
    would pass a naive check. Testing the string directly is both simpler and portable.
    """

    def __init__(self, p: str):
        self.is_absolute = p.startswith("/") or (len(p) > 1 and p[1] == ":")


# ----------------------------------------------------------------- canonical text
def canon_text(s: str) -> str:
    """Textual canonicalisation -- see the module docstring for what this is not."""
    if not isinstance(s, str):
        raise ArtifactError(f"file contents must be str, got {type(s).__name__}")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    body = "\n".join(line.rstrip() for line in s.split("\n"))
    return body.rstrip("\n") + "\n" if body.strip() else ""


def canon_tree(files: Mapping[str, str]) -> dict[str, str]:
    """Normalise paths and contents, and enforce the size bounds."""
    if not isinstance(files, Mapping) or not files:
        raise ArtifactError("a tree must be a non-empty mapping of path -> source")
    if len(files) > MAX_TREE_FILES:
        raise ArtifactError(f"tree has {len(files)} files; the cap is {MAX_TREE_FILES}")
    out: dict[str, str] = {}
    total = 0
    for raw_path, raw_src in files.items():
        rel = safe_relpath(raw_path)
        if rel in out:
            raise ArtifactError(f"duplicate path after normalisation: {rel!r}")
        text = canon_text(raw_src)
        n = len(text.encode("utf-8"))
        if n > MAX_FILE_BYTES:
            raise ArtifactError(
                f"{rel} is {n:,} bytes; the per-file cap is {MAX_FILE_BYTES:,}")
        total += n
        out[rel] = text
    if total > MAX_TREE_BYTES:
        raise ArtifactError(f"tree is {total:,} bytes; the cap is {MAX_TREE_BYTES:,}")
    return out


def tree_hash(files: Mapping[str, str], *, already_canonical: bool = False) -> str:
    """sha256 over a length-prefixed, path-sorted serialisation.

    Length prefixes make the encoding injective: without them the pair
    ("ab", "c") and ("a", "bc") would hash identically, so two different trees could
    share an address.
    """
    canon = files if already_canonical else canon_tree(files)
    h = hashlib.sha256()
    h.update(TREE_VERSION)
    for rel in sorted(canon):
        pb = rel.encode("utf-8")
        db = canon[rel].encode("utf-8")
        h.update(len(pb).to_bytes(8, "big"))
        h.update(pb)
        h.update(len(db).to_bytes(8, "big"))
        h.update(db)
    return h.hexdigest()


# ------------------------------------------------------------------------- diffs
def tree_diff(parent: Mapping[str, str] | None, child: Mapping[str, str],
              *, context: int = 3) -> str:
    """Unified diff from `parent` to `child`, computed HERE.

    The diff is never taken from the model. An agent that reports a small tidy change
    while writing something else would otherwise produce a run log that reads correctly
    and is false -- and the run log is the artifact judges read. So the harness applies
    edits, then derives the diff from the two trees it holds.
    """
    p = {} if parent is None else canon_tree(parent)
    c = canon_tree(child)
    chunks: list[str] = []
    for rel in sorted(set(p) | set(c)):
        before = p.get(rel, "").splitlines(keepends=True)
        after = c.get(rel, "").splitlines(keepends=True)
        if before == after:
            continue
        a_label = f"a/{rel}" if rel in p else "/dev/null"
        b_label = f"b/{rel}" if rel in c else "/dev/null"
        chunks.extend(unified_diff(before, after, fromfile=a_label, tofile=b_label,
                                   n=context))
    text = "".join(chunks)
    # A unified_diff line only ends in \n if the source line did; a file with no final
    # newline yields a chunk that runs into the next one. Repair it so the diff applies.
    return "".join(l if l.endswith("\n") else l + "\n" for l in text.splitlines(True))


def diff_hash(diff_text: str) -> str:
    return hashlib.sha256(diff_text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------- store
@dataclass(frozen=True)
class Artifact:
    tree: str                       #: full sha256 of the canonical tree
    path: Path                      #: runs/artifacts/<tree>
    files: tuple[str, ...]          #: sorted relative paths
    bytes: int

    @property
    def short(self) -> str:
        return self.tree[:12]

    @property
    def source_dir(self) -> Path:
        return self.path / "source"

    def to_record(self) -> dict:
        """The subset that belongs in a signed ledger entry -- hashes, not source."""
        return {"tree": self.tree, "files": list(self.files), "bytes": self.bytes}


def _write_tree_into(canon: Mapping[str, str], dest: Path) -> None:
    for rel, text in canon.items():
        f = dest / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")


def put(files: Mapping[str, str], *, root: Path = ARTIFACTS,
        meta: Mapping | None = None) -> Artifact:
    """Store a tree immutably and return its address. Idempotent.

    Written to a temporary directory and moved into place with a single `os.replace`,
    so a crash mid-write cannot leave a half-tree at an address that claims to be
    complete. If the address already exists it is VERIFIED and returned rather than
    overwritten -- an artifact directory is written once.
    """
    canon = canon_tree(files)
    th = tree_hash(canon, already_canonical=True)
    root.mkdir(parents=True, exist_ok=True)
    final = root / th
    total = sum(len(v.encode("utf-8")) for v in canon.values())
    art = Artifact(th, final, tuple(sorted(canon)), total)

    if final.exists():
        verify(th, root=root)                 # refuse to trust an address blindly
        return art

    tmp = root / f".tmp_{os.getpid()}_{th[:12]}"
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        (tmp / "source").mkdir(parents=True)
        _write_tree_into(canon, tmp / "source")
        payload = {"tree": th, "tree_version": TREE_VERSION.decode().strip(),
                   "files": sorted(canon), "bytes": total}
        if meta:
            payload["meta"] = json.loads(json.dumps(dict(meta), allow_nan=False,
                                                    sort_keys=True))
        (tmp / "meta.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        try:
            os.replace(tmp, final)
        except OSError:
            # Another process won the race to this address. Content-addressed, so its
            # bytes are ours; verify rather than assume, then discard our copy.
            if not final.exists():
                raise
            verify(th, root=root)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return art


def read(tree: str, *, root: Path = ARTIFACTS) -> dict[str, str]:
    """Read a stored tree back as a path -> source mapping."""
    src = (root / tree) / "source"
    if not src.is_dir():
        raise ArtifactError(f"no artifact at {tree[:12]} under {root}")
    files: dict[str, str] = {}
    for f in sorted(src.rglob("*")):
        if f.is_dir():
            continue
        if f.is_symlink():
            raise ArtifactError(f"artifact {tree[:12]} contains a symlink: {f.name}")
        files[f.relative_to(src).as_posix()] = f.read_text(encoding="utf-8")
    if not files:
        raise ArtifactError(f"artifact {tree[:12]} has an empty source tree")
    return files


def verify(tree: str, *, root: Path = ARTIFACTS) -> Artifact:
    """Recompute the address from the stored bytes. Raises if they disagree.

    This is what lets the ledger carry a hash instead of the source: a signed entry
    naming a tree is only meaningful if the tree can be shown to be the one signed.
    """
    files = read(tree, root=root)
    got = tree_hash(files)
    if got != tree:
        raise ArtifactError(
            f"artifact {tree[:12]} does not match its contents (recomputed "
            f"{got[:12]}). The stored tree has been altered.")
    total = sum(len(canon_text(v).encode("utf-8")) for v in files.values())
    return Artifact(tree, root / tree, tuple(sorted(files)), total)


def materialise(tree: str, dest: Path, *, root: Path = ARTIFACTS) -> Path:
    """Copy a verified tree into `dest` for execution.

    Each candidate gets its OWN destination: the sandbox's write grant is narrowed to
    that directory, so one candidate cannot reach into another's working copy.
    """
    art = verify(tree, root=root)
    files = read(tree, root=root)
    dest.mkdir(parents=True, exist_ok=True)
    _write_tree_into(files, dest)
    return dest


def read_dir(src: Path, *, include: Iterable[str] | None = None) -> dict[str, str]:
    """Read a working directory into a tree mapping.

    `include` restricts to a known set of relative paths -- used so an editable workspace
    contributes exactly the files the contract declares mutable, and a stray `.pyc` or
    editor backup never changes a candidate's identity.
    """
    if not src.is_dir():
        raise ArtifactError(f"not a directory: {src}")
    want = None if include is None else {safe_relpath(p) for p in include}
    files: dict[str, str] = {}
    for f in sorted(src.rglob("*")):
        if f.is_dir() or f.is_symlink():
            continue
        rel = f.relative_to(src).as_posix()
        if want is not None and rel not in want:
            continue
        if want is None and ("__pycache__" in rel or rel.endswith((".pyc", ".pyo"))):
            continue
        files[rel] = f.read_text(encoding="utf-8")
    if want is not None:
        missing = sorted(want - set(files))
        if missing:
            raise ArtifactError(f"{src} is missing declared files: {missing}")
    return files
