"""Public provider thought summaries are optional, bounded, and visible in replay."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import backends as B
from agent import control as C

R = []


def check(name, condition):
    R.append((name, bool(condition)))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}")


response = {
    "status": "completed",
    "steps": [
        {"type": "thought", "signature": "opaque-secret-state",
         "summary": [{"type": "text", "text": "Compare the two loss families."}]},
        {"type": "thought", "signature": "opaque-2"},
        {"type": "model_output", "content": [{"type": "text", "text": "{}"}]},
    ],
    "usage": {"total_input_tokens": 10, "total_output_tokens": 2,
              "total_thought_tokens": 7},
}
summaries = B.extract_thought_summaries(response)
check("extracts provider public summary", summaries == ["Compare the two loss families."])
check("does not expose opaque thought signature", "opaque" not in repr(summaries))
check("empty thought blocks are tolerated", len(summaries) == 1)

captured = {}
original_post = B._post
B._post = lambda url, body, headers: (captured.update(body=body) or response)
try:
    enabled = B.GeminiBackend("test-key", "test-model", "high", True)
    _resp, usage = enabled.complete_json("system", "user", {"type": "object"})
    disabled = B.GeminiBackend("test-key", "test-model", None, False)
    captured_disabled = {}
    B._post = lambda url, body, headers: (captured_disabled.update(body=body) or response)
    disabled.complete_json("system", "user", {"type": "object"})
finally:
    B._post = original_post

gen = captured["body"]["generation_config"]
check("enabled request asks API for summaries", gen["thinking_summaries"] == "auto")
check("thinking level remains independent", gen["thinking_level"] == "high")
check("disabled request sends no summary switch",
      "generation_config" not in captured_disabled["body"])
check("usage carries summaries beside token counts",
      usage["thought_summaries"] == summaries and usage["thinking_tokens"] == 7)

print()
# The docstring above says "bounded". Nothing checked it: the bounding lived in a closure
# inside research_driver, so a provider returning many long summaries would build a record
# over core.scorer.MAX_RECORD_BYTES, which is REFUSED -- and the refusal was discarded.
import json
from agent.research_driver import bound_thought_summaries, THOUGHT_MAX_ITEMS
from core.scorer import MAX_RECORD_BYTES

many = [f"summary {i} " + ("x" * 1900) for i in range(20)]
kept, dropped = bound_thought_summaries(many)
# The CHAR budget binds before the item count here: each item truncates to 1800, so
# 6000 chars allows 3, not THOUGHT_MAX_ITEMS. Assert both bounds, not a guessed number.
check("many long summaries are capped", 0 < len(kept) <= THOUGHT_MAX_ITEMS)
check("the kept summaries respect the character budget",
      sum(len(k) for k in kept) <= 6000)
check("the dropped count is reported, not hidden", dropped == 20 - len(kept))
check("each summary is truncated to its per-item limit", all(len(k) <= 1800 for k in kept))
body = {"run_id": "r", "iteration": 1, "what": "llm_thought_summary", "role": "plan",
        "stage": "plan", "model": "m", "summaries": kept, "thinking_tokens": 7,
        "summaries_omitted_for_record_size": dropped,
        "note": "provider-generated public summary; not chain-of-thought"}
size = len(json.dumps(body).encode())
check(f"the resulting record fits under MAX_RECORD_BYTES ({size} <= {MAX_RECORD_BYTES})",
      size <= MAX_RECORD_BYTES)
check("blank and non-string summaries are dropped",
      bound_thought_summaries(["", "  ", None, "real"])[0] == ["real"])
check("no summaries yields no record", bound_thought_summaries(None) == ([], 0))

entries = [{"seq": 9, "payload": {"kind": "event", "body": {
    "what": "llm_thought_summary", "iteration": 2, "role": "judge",
    "stage": "judge", "model": "test-model", "thinking_tokens": 7,
    "summaries": summaries,
    "note": "provider-generated public summary; not chain-of-thought"}}}]
events = C.research_events(entries)
check("console turns summary into a research event",
      len(events) == 1 and events[0]["title"] == "Judge thought summary")
check("console labels the disclosure precisely",
      events[0]["detail"]["disclosure"].endswith("not chain-of-thought"))
check("console summary is readable without opening JSON",
      events[0]["summary"] == summaries[0])

print()
bad = [name for name, ok in R if not ok]
print(f"{len(R)-len(bad)}/{len(R)} passed" + (f"  FAILURES: {bad}" if bad else ""))
raise SystemExit(1 if bad else 0)
