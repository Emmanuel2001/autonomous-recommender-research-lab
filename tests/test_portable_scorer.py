"""A real scorer process can serve and sign a portable task over its trust boundary."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.ledger import verify
from core.task import load

R = []


def check(name, condition, extra=""):
    ok = bool(condition)
    R.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


def request(sock_path: Path, payload: dict) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(10)
        s.connect(str(sock_path))
        s.sendall((json.dumps(payload) + "\n").encode())
        data = b""
        while not data.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
    return json.loads(data)


task_path = ROOT / "tests" / "fixtures" / "portable_recommender" / "task.json"
spec = load(task_path)
full, _ = spec.scorer_loader()

with tempfile.TemporaryDirectory(prefix="track2-portable-scorer-") as td:
    temp = Path(td)
    sock = temp / "scorer.sock"
    ledger = temp / "ledger.jsonl"
    score_path = temp / "scores.npy"
    np.save(score_path, np.linspace(-1.0, 1.0, len(full["valid"])))
    key = "portable-scorer-key-0123456789abcdef"
    env = dict(os.environ, TRACK2_SOCKET=str(sock), TRACK2_LEDGER=str(ledger),
               TRACK2_SIGNING_KEY=key, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "core.scorer", "serve",
         "--task-spec", str(task_path)], cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(100):
            if sock.exists():
                break
            if proc.poll() is not None:
                break
            time.sleep(0.05)
        check("portable scorer reaches listening state", sock.exists(),
              (proc.stdout.read() if proc.poll() is not None and proc.stdout else ""))
        if sock.exists():
            ping = request(sock, {"op": "ping"})
            check("ping identifies portable task", ping.get("task") == spec.slug)
            check("ping reports verified chain", ping.get("chain_verified_on_open") is True)
            scored = request(sock, {"op": "score_task", "scores": str(score_path),
                                     "split": "valid", "config_hash": "fixture-tree",
                                     "isolation": "none"})
            check("portable score request succeeds", scored.get("ok"), str(scored))
            if scored.get("ok"):
                payload = scored["entry"]["payload"]
                check("signed response carries task-owned primary",
                      payload["task"] == spec.slug and "primary" in payload["metrics"])
            wrong = request(sock, {"op": "score_official", "scores": str(score_path),
                                   "split": "valid", "config_hash": "wrong-op"})
            check("official scoring op is refused on portable scorer",
                  wrong.get("ok") is False and "legacy" in wrong.get("error", ""))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=5)
    check("portable ledger verifies after process shutdown", verify(ledger, key.encode()).ok)

print()
bad = [name for name, ok in R if not ok]
print(f"{len(R)-len(bad)}/{len(R)} passed" + (f"  FAILURES: {bad}" if bad else ""))
raise SystemExit(1 if bad else 0)
