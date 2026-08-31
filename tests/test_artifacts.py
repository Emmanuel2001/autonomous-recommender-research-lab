"""Tests for the content-addressed candidate store.

This store is what lets a signed ledger entry carry a HASH instead of the source it
attests to, so two properties matter more than the rest and are tested hardest:

  * the address is injective -- two different trees can never share one, including the
    "ab"+"c" / "a"+"bc" concatenation ambiguity that a naive hash walks straight into;
  * `verify()` actually recomputes, so an altered artifact is detected rather than
    trusted because its directory name looks right.
"""
import json, shutil, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import artifacts as A

R = []
def check(n, c, extra=""):
    R.append((n, bool(c))); print(f"  {'PASS' if c else 'FAIL'}  {n}{('  '+extra) if extra else ''}")

def raises(fn, *a, **k):
    """Returns the ArtifactError message, or "" if it did not raise. A test that
    accepts any exception would pass on a typo in the call itself."""
    try:
        fn(*a, **k)
    except A.ArtifactError as e:
        return str(e) or "raised"
    except Exception as e:
        return f"WRONG-TYPE:{type(e).__name__}"
    return ""

TMP = Path(tempfile.mkdtemp(prefix="track2_artifacts_"))
STORE = TMP / "artifacts"

BASE = {"model.py": "def f(x):\n    return x + 1\n",
        "config.json": '{\n  "lr": 0.05\n}\n'}

# 1 -- the address is a function of content, and injective ----------------------
check("the same tree hashes to the same address",
      A.tree_hash(BASE) == A.tree_hash(dict(BASE)))
check("different content gives a different address",
      A.tree_hash(BASE) != A.tree_hash({**BASE, "model.py": "def f(x):\n    return x\n"}))
check("a different path gives a different address",
      A.tree_hash({"a.py": "x = 1\n"}) != A.tree_hash({"b.py": "x = 1\n"}))
# The ambiguity a length-prefix exists to kill: without prefixes both serialise to
# the bytes "abc" and the two trees would collide at one address.
check("path/content boundary is unambiguous ('ab'+'c' vs 'a'+'bc')",
      A.tree_hash({"ab": "c"}) != A.tree_hash({"a": "bc"}))
check("key order does not affect the address",
      A.tree_hash({"a.py": "1\n", "b.py": "2\n"})
      == A.tree_hash({"b.py": "2\n", "a.py": "1\n"}))
check("the version prefix is part of the hash input",
      A.TREE_VERSION == b"track2-tree-v1\n")

# 2 -- canonicalisation is textual, and collapses only what it claims to --------
check("CRLF and LF are the same tree",
      A.tree_hash({"m.py": "a = 1\r\nb = 2\r\n"}) == A.tree_hash({"m.py": "a = 1\nb = 2\n"}))
check("a bare CR is normalised too",
      A.tree_hash({"m.py": "a = 1\rb = 2\r"}) == A.tree_hash({"m.py": "a = 1\nb = 2\n"}))
check("trailing whitespace on a line does not change the tree",
      A.tree_hash({"m.py": "a = 1   \nb = 2\t\n"}) == A.tree_hash({"m.py": "a = 1\nb = 2\n"}))
check("a missing final newline does not change the tree",
      A.tree_hash({"m.py": "a = 1"}) == A.tree_hash({"m.py": "a = 1\n"}))
check("extra blank lines at EOF do not change the tree",
      A.tree_hash({"m.py": "a = 1\n\n\n"}) == A.tree_hash({"m.py": "a = 1\n"}))
check("blank lines INSIDE the file are preserved (diffs stay readable)",
      A.tree_hash({"m.py": "a = 1\n\nb = 2\n"}) != A.tree_hash({"m.py": "a = 1\nb = 2\n"}))
# Stated limit, asserted so it cannot be quietly claimed otherwise later.
check("canonicalisation is NOT semantic: renaming a local makes a new tree",
      A.tree_hash({"m.py": "def f(a):\n    return a\n"})
      != A.tree_hash({"m.py": "def f(b):\n    return b\n"}))

# 3 -- path safety: a candidate stays inside its own tree -----------------------
for bad, why in [("../escape.py", "parent traversal"),
                 ("a/../../b.py", "traversal mid-path"),
                 ("/etc/passwd", "absolute posix"),
                 ("C:/windows/x.py", "absolute windows"),
                 ("a\\b.py", "backslash separator"),
                 ("./a.py", "explicit dot component"),
                 ("", "empty"),
                 ("   ", "whitespace only"),
                 ("a\x00b.py", "NUL byte")]:
    msg = raises(A.safe_relpath, bad)
    check(f"path refused: {why} ({bad!r})", msg and not msg.startswith("WRONG-TYPE"), msg[:60])
check("a nested legitimate path is accepted", A.safe_relpath("sub/dir/m.py") == "sub/dir/m.py")
check("duplicate slashes are normalised", A.safe_relpath("a//b.py") == "a/b.py")
check("a traversal inside a tree is refused, not silently rewritten",
      raises(A.canon_tree, {"../x.py": "1\n"}) != "")
check("two paths that normalise to one are refused rather than merged",
      "duplicate" in raises(A.canon_tree, {"a/b.py": "1\n", "a//b.py": "2\n"}))

# 4 -- size bounds --------------------------------------------------------------
check("a too-large file is refused",
      "per-file cap" in raises(A.canon_tree, {"m.py": "x" * (A.MAX_FILE_BYTES + 1)}))
check("too many files are refused",
      "the cap is" in raises(A.canon_tree,
                             {f"f{i}.py": "x\n" for i in range(A.MAX_TREE_FILES + 1)}))
check("an empty tree is refused", raises(A.canon_tree, {}) != "")
check("a non-str file body is refused",
      raises(A.canon_tree, {"m.py": 42}) != "")

# 5 -- store: write once, atomically, and verify on read ------------------------
art = A.put(BASE, root=STORE, meta={"note": "seed"})
check("put returns the content address", art.tree == A.tree_hash(BASE))
check("the artifact directory is named by its address", art.path.name == art.tree)
check("source lands under source/", (art.source_dir / "model.py").is_file())
check("meta.json is written", (art.path / "meta.json").is_file())
check("files are recorded sorted", art.files == ("config.json", "model.py"))
check("byte total matches the canonical bytes",
      art.bytes == sum(len(A.canon_text(v).encode()) for v in BASE.values()))
meta = json.loads((art.path / "meta.json").read_text())
check("meta.json records the tree and its files",
      meta["tree"] == art.tree and meta["files"] == list(art.files))
check("caller meta is preserved", meta.get("meta", {}).get("note") == "seed")
check("the artifact holds ONLY source and meta (outputs live with the run)",
      sorted(p.name for p in art.path.iterdir()) == ["meta.json", "source"])
check("no temp directory is left behind",
      not [p for p in STORE.iterdir() if p.name.startswith(".tmp_")])

rt = A.read(art.tree, root=STORE)
check("read round-trips the tree", A.tree_hash(rt) == art.tree)
check("read returns canonical text", rt["model.py"] == A.canon_text(BASE["model.py"]))

again = A.put(BASE, root=STORE)
check("put is idempotent (same address, no error)", again.tree == art.tree)
check("put of a canonically-equal tree hits the same address",
      A.put({"model.py": "def f(x):\r\n    return x + 1\r\n",
             "config.json": '{\n  "lr": 0.05\n}'}, root=STORE).tree == art.tree)

check("verify accepts an intact artifact", A.verify(art.tree, root=STORE).tree == art.tree)
check("verify refuses a missing artifact", raises(A.verify, "0" * 64, root=STORE) != "")

# The property the ledger's hash-carrying design rests on.
tampered = A.put({"model.py": "def f(x):\n    return x + 2\n"}, root=STORE)
(tampered.source_dir / "model.py").write_text("def f(x):\n    return x + 999\n")
msg = raises(A.verify, tampered.tree, root=STORE)
check("verify DETECTS an altered artifact", "does not match its contents" in msg, msg[:70])

# 6 -- diffs are derived by the harness, never taken from the model -------------
child = {**BASE, "model.py": "def f(x):\n    return x * 2\n"}
d = A.tree_diff(BASE, child)
check("a modified file produces a hunk", "@@" in d and "-    return x + 1" in d
      and "+    return x * 2" in d)
check("an unchanged file contributes nothing", "config.json" not in d)
check("identical trees produce an empty diff", A.tree_diff(BASE, BASE) == "")
check("an added file is shown against /dev/null",
      "--- /dev/null" in A.tree_diff(BASE, {**BASE, "new.py": "x = 1\n"}))
check("a removed file is shown as going to /dev/null",
      "+++ /dev/null" in A.tree_diff(BASE, {"config.json": BASE["config.json"]}))
check("parent=None presents the whole tree as added",
      A.tree_diff(None, BASE).count("--- /dev/null") == 2)
check("every diff line is newline-terminated (the diff stays applicable)",
      all(l.endswith("\n") for l in A.tree_diff(None, BASE).splitlines(True)))
check("the diff is a pure function of the two trees",
      A.diff_hash(A.tree_diff(BASE, child)) == A.diff_hash(A.tree_diff(BASE, child)))
check("a cosmetic-only edit produces no diff at all",
      A.tree_diff(BASE, {**BASE, "model.py": BASE["model.py"].rstrip() + "\n\n"}) == "")

# 7 -- materialise: one exclusive working copy per candidate --------------------
w1, w2 = TMP / "cand1", TMP / "cand2"
A.materialise(art.tree, w1, root=STORE)
A.materialise(art.tree, w2, root=STORE)
check("materialise writes the tree out", (w1 / "model.py").is_file())
check("two candidates get independent copies", (w2 / "model.py").is_file())
(w1 / "model.py").write_text("def f(x):\n    return 0\n")
check("editing one working copy does not touch the other",
      "return x + 1" in (w2 / "model.py").read_text())
check("editing a working copy does not touch the stored artifact",
      "return x + 1" in (art.source_dir / "model.py").read_text())
check("materialise refuses an unverifiable tree",
      raises(A.materialise, tampered.tree, TMP / "cand3", root=STORE) != "")

# 8 -- read_dir: identity comes from the DECLARED files only --------------------
wd = TMP / "workspace"
(wd / "__pycache__").mkdir(parents=True)
(wd / "__pycache__" / "m.cpython-311.pyc").write_text("junk")
(wd / "model.py").write_text(BASE["model.py"])
(wd / "config.json").write_text(BASE["config.json"])
(wd / "scratch.txt").write_text("notes the agent left behind\n")
declared = A.read_dir(wd, include=["model.py", "config.json"])
check("read_dir with include returns exactly the declared files",
      sorted(declared) == ["config.json", "model.py"])
check("an undeclared stray file cannot change candidate identity",
      A.tree_hash(declared) == art.tree)
check("read_dir refuses a missing declared file",
      "missing declared files" in raises(A.read_dir, wd, include=["model.py", "gone.py"]))
check("read_dir without include still skips __pycache__ and .pyc",
      not any("pycache" in k or k.endswith(".pyc") for k in A.read_dir(wd)))
check("read_dir refuses a non-directory", raises(A.read_dir, wd / "model.py") != "")

shutil.rmtree(TMP, ignore_errors=True)
print()
bad_l = [n for n, ok in R if not ok]
print(f"{len(R)-len(bad_l)}/{len(R)} passed" + (f"  FAILURES: {bad_l}" if bad_l else ""))
sys.exit(1 if bad_l else 0)
