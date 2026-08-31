# Champion source — tree `fe985741a8368e082be4c71e0d9e3be06ca333f2aac1b26216203cbf2283c3b5`

The exact agent-written strategy this submission's predictions came from, exported
from the content-addressed store at `runs/artifacts/fe985741a8368e082be4c71e0d9e3be06ca333f2aac1b26216203cbf2283c3b5/`.

Verify it is the tree the ledger names:

```bash
python3 -c "import sys; sys.path.insert(0,'.'); from core import artifacts as A; \
  print(A.tree_hash({p.name: p.read_text() for p in \
  __import__('pathlib').Path('submission_research/champion_source').glob('*.py')} | \
  {'config.json': open('submission_research/champion_source/config.json').read()}))"
```

which must print `fe985741a8368e082be4c71e0d9e3be06ca333f2aac1b26216203cbf2283c3b5`.

**Authorship, precisely:** `features.py` and `config.json` were written by the
agent. `model.py` and `train.py` are the seed implementations it inherited and
never edited — see `../../deliverables/AUTONOMY.md`.
