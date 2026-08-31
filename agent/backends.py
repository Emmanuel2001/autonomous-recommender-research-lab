"""LLM backends for the proposer. Raw HTTP, stdlib only -- no new dependencies.

WHY RAW HTTP RATHER THAN VENDOR SDKs
------------------------------------
Total input+output tokens is 15% of the score and must be reported exactly. An SDK
that wraps usage in convenience properties is one more layer between us and the
number we are graded on -- the same reason LangChain was rejected. urllib is in the
standard library, the request bodies are a dozen lines each, and every byte of the
response is visible and loggable.

BACKENDS
  ollama            local, free, no key, no rate limit. Recommended: gpt-oss:20b (~12GB)
  openai_compat     Groq / Together / any OpenAI-shaped endpoint
  gemini            Google AI Studio free tier
  (anthropic lives in proposer.py because it uses the official SDK)

USAGE FIELD DISCOVERY
Gemini's token-count field names are not documented on the pages we could reach, so
`extract_usage` probes a list of plausible paths AND stores the raw usage object
verbatim. Nothing is silently lost, and `probe()` dumps a whole response so the
paths can be pinned against reality instead of guessed.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

TIMEOUT = 300.0          # seconds


class BackendError(RuntimeError):
    def __init__(self, msg: str, status: int | None = None, body: str = ""):
        super().__init__(msg)
        self.status, self.body = status, body


def _post(url: str, body: dict, headers: dict, timeout: float = TIMEOUT) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")[:1200]
        # 4xx is a request bug: never retry it, surface it loudly
        raise BackendError(f"HTTP {e.code} from {url}: {raw}", e.code, raw) from None
    except urllib.error.URLError as e:
        raise BackendError(f"could not reach {url}: {e.reason}") from None


# --------------------------------------------------------------- usage handling
# Gemini's interactions endpoint reports usage as (verified live, 2026-08-26,
# gemini-3.5-flash-lite):
#   usage.total_input_tokens / total_output_tokens / total_thought_tokens
#   usage.total_cached_tokens / total_tokens / total_tool_use_tokens
#   usage.raw_prompt_token   <- larger than total_input_tokens (126 vs 21 on a
#                               trivial call); kept in usage_raw, not summed
# OpenAI-compatible paths are retained for ollama/groq.
_USAGE_PATHS = {
    "input": [("usage", "total_input_tokens"),          # gemini interactions
              ("usage", "input_tokens"), ("usage", "prompt_tokens"),
              ("usage_metadata", "prompt_token_count"),
              ("usageMetadata", "promptTokenCount")],
    "output": [("usage", "total_output_tokens"),        # gemini interactions
               ("usage", "output_tokens"), ("usage", "completion_tokens"),
               ("usage_metadata", "candidates_token_count"),
               ("usageMetadata", "candidatesTokenCount")],
    "thinking": [("usage", "total_thought_tokens"),     # gemini interactions
                 ("usage_metadata", "thoughts_token_count"),
                 ("usageMetadata", "thoughtsTokenCount"),
                 ("usage", "reasoning_tokens")],
    "cached": [("usage", "total_cached_tokens"),
               ("usage", "cache_read_input_tokens")],
}


def _dig(d: dict, path: tuple):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur if isinstance(cur, (int, float)) else None


def extract_usage(resp: dict, backend: str) -> dict:
    """Find token counts wherever they live, and keep the raw object regardless.

    Reporting a token count we could not actually find would be worse than
    reporting that we could not find it, so `usage_fields_found` is explicit.
    """
    out = {"backend": backend, "input_tokens": 0, "output_tokens": 0,
           "thinking_tokens": 0, "cached_tokens": 0, "usage_fields_found": []}
    for kind, paths in _USAGE_PATHS.items():
        for p in paths:
            v = _dig(resp, p)
            if v is not None:
                out[f"{kind}_tokens"] = int(v)
                out["usage_fields_found"].append(".".join(p))
                break
    # THE reported figure. Thought tokens are generated tokens, so they belong in
    # the total -- omitting them would under-report the quantity the rubric scores.
    out["total_reported"] = out["input_tokens"] + out["output_tokens"] + out["thinking_tokens"]
    raw = resp.get("usage") or resp.get("usage_metadata") or resp.get("usageMetadata")
    out["usage_raw"] = raw if isinstance(raw, dict) else None
    if not out["usage_fields_found"]:
        out["warning"] = ("no token-usage field matched a known path; usage_raw is "
                          "stored verbatim. Run `python3 -m agent.backends probe` and "
                          "add the real path to _USAGE_PATHS.")
    return out


def extract_thought_summaries(resp: dict) -> list[str]:
    """Public provider summaries, never opaque signatures or hidden chain-of-thought."""
    summaries = []
    for step in resp.get("steps") or []:
        if not isinstance(step, dict) or step.get("type") != "thought":
            continue
        for block in step.get("summary") or []:
            if (isinstance(block, dict) and block.get("type") == "text"
                    and isinstance(block.get("text"), str) and block["text"].strip()):
                summaries.append(block["text"].strip())
    return summaries


# -------------------------------------------------------------------- backends
class GeminiBackend:
    """Google AI Studio. Free tier, no billing required."""
    name = "gemini"
    URL = "https://generativelanguage.googleapis.com/v1beta/interactions"

    def __init__(self, api_key: str, model: str = "gemini-3.5-flash-lite",
                 thinking_level: str | None = None,
                 thinking_summaries: bool = False):
        if not api_key:
            raise ValueError("GEMINI_API_KEY / TRACK2_LLM_API_KEY is empty")
        self.key, self.model, self.thinking = api_key, model, thinking_level
        self.thinking_summaries = bool(thinking_summaries)

    def complete_json(self, system: str, user: str, schema: dict) -> tuple[dict, dict]:
        gen: dict = {}
        if self.thinking:
            gen["thinking_level"] = self.thinking
        if self.thinking_summaries:
            # Interactions API public reasoning summary. This is intentionally not the
            # encrypted thought signature and does not expose private chain-of-thought.
            gen["thinking_summaries"] = "auto"
        body = {
            "model": self.model,
            "system_instruction": system,
            "input": user,
            # response_format is Gemini's structured-output field. additionalProperties
            # IS supported (verified in the structured-output docs), so our schema
            # passes through unchanged.
            "response_format": {"type": "text", "mime_type": "application/json",
                                "schema": schema},
            # do not accumulate server-side conversation state: every proposal is a
            # fresh stateless call, which is what keeps context flat and auditable
            "store": False,
        }
        if gen:
            body["generation_config"] = gen
        resp = _post(self.URL, body, {"x-goog-api-key": self.key})
        status = resp.get("status")
        if status and status != "completed":
            raise BackendError(f"interaction status={status!r}, not 'completed'; "
                               f"keys={sorted(resp)}")
        usage = extract_usage(resp, self.name)
        usage["thought_summaries"] = extract_thought_summaries(resp)
        return resp, usage

    @staticmethod
    def text_of(resp: dict) -> str:
        if isinstance(resp.get("output_text"), str):
            return resp["output_text"]
        # VERIFIED shape from the interactions endpoint: a steps[] list where the
        # answer is the last step of type "model_output"; a preceding "thought" step
        # carries only an opaque signature and must be skipped.
        steps = resp.get("steps")
        if isinstance(steps, list):
            for st in reversed(steps):
                if not isinstance(st, dict) or st.get("type") != "model_output":
                    continue
                for blk in st.get("content") or []:
                    if isinstance(blk, dict) and blk.get("type") == "text" \
                            and isinstance(blk.get("text"), str):
                        return blk["text"]
        # older / alternate shapes
        for path in (("output", 0, "content", 0, "text"),
                     ("candidates", 0, "content", "parts", 0, "text"),
                     ("output", 0, "text")):
            cur = resp
            try:
                for k in path:
                    cur = cur[k]
                if isinstance(cur, str):
                    return cur
            except (KeyError, IndexError, TypeError):
                continue
        raise BackendError(f"could not locate output text in response keys "
                           f"{sorted(resp)}; run `python3 -m agent.backends probe`")


class OpenAICompatBackend:
    """Ollama (local, free), Groq (free tier), Together, and anything OpenAI-shaped."""
    name = "openai_compat"

    def __init__(self, base_url: str, model: str, api_key: str = ""):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.key = model, api_key

    def complete_json(self, system: str, user: str, schema: dict) -> tuple[dict, dict]:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "action", "strict": True,
                                                "schema": schema}},
            "stream": False,
        }
        h = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        resp = _post(self.url, body, h)
        return resp, extract_usage(resp, self.name)

    @staticmethod
    def text_of(resp: dict) -> str:
        try:
            return resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise BackendError(f"unexpected response shape, keys {sorted(resp)}") from None


def make_backend(kind: str, **kw):
    if kind == "gemini":
        return GeminiBackend(kw["api_key"], kw.get("model") or "gemini-3.5-flash-lite",
                             kw.get("thinking_level"), kw.get("thinking_summaries", False))
    if kind == "ollama":
        return OpenAICompatBackend(kw.get("base_url") or "http://localhost:11434/v1",
                                   kw.get("model") or "gpt-oss:20b", "")
    if kind == "groq":
        return OpenAICompatBackend(kw.get("base_url") or "https://api.groq.com/openai/v1",
                                   kw.get("model") or "gpt-oss-20b", kw["api_key"])
    raise ValueError(f"unknown backend {kind!r}; choose gemini|ollama|groq")


def list_gemini_models(api_key: str) -> list[dict]:
    """Real model IDs from the API. Console display names ("Gemini 3.5 Flash Lite")
    are not the API ids, and guessing the mapping is how you get a 404 at 3am."""
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    req = urllib.request.Request(url, headers={"x-goog-api-key": api_key})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise BackendError(f"HTTP {e.code}: {e.read().decode('utf-8','replace')[:400]}") from None
    out = []
    for m in d.get("models", []):
        name = (m.get("name") or "").removeprefix("models/")
        methods = m.get("supportedGenerationMethods") or m.get("supported_generation_methods") or []
        out.append({"id": name, "display": m.get("displayName") or m.get("display_name") or "",
                    "in": m.get("inputTokenLimit") or m.get("input_token_limit"),
                    "out": m.get("outputTokenLimit") or m.get("output_token_limit"),
                    "methods": methods})
    return out


# ------------------------------------------------------------------- probe mode
def probe() -> None:
    """Make ONE minimal call and dump the whole response, so usage field names are
    pinned against reality rather than guessed."""
    import argparse, os, sys
    ap = argparse.ArgumentParser(prog="python3 -m agent.backends probe")
    ap.add_argument("probe", nargs="?")
    ap.add_argument("--backend", default="gemini", choices=["gemini", "ollama", "groq"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--list", action="store_true",
                    help="list real API model ids and exit (gemini only)")
    a = ap.parse_args()
    key = (os.environ.get("TRACK2_LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
           or os.environ.get("GROQ_API_KEY") or "")
    if a.backend != "ollama" and not key:
        sys.exit("set TRACK2_LLM_API_KEY (or GEMINI_API_KEY / GROQ_API_KEY)")
    if a.list:
        if a.backend != "gemini":
            sys.exit("--list is gemini only")
        rows = list_gemini_models(key)
        text = [m for m in rows if any("generate" in x.lower() or "interact" in x.lower()
                                       for x in m["methods"]) or not m["methods"]]
        print(f"{len(rows)} models visible to this key; text-capable first\n")
        print(f"  {'API MODEL ID':38s} {'in':>9s} {'out':>7s}  display")
        print("  " + "-" * 86)
        for m in sorted(text, key=lambda x: x["id"]):
            if "embedding" in m["id"] or "tts" in m["id"] or "image" in m["id"]:
                continue
            print(f"  {m['id']:38s} {str(m['in'] or '-'):>9s} {str(m['out'] or '-'):>7s}  {m['display']}")
        print("\nPick a *-flash-lite id for bulk iterations (500 RPD) and a full "
              "*-flash id for the few rationales you quote (20 RPD).")
        return
    be = make_backend(a.backend, api_key=key, model=a.model)
    schema = {"type": "object", "additionalProperties": False,
              "required": ["ok", "n"],
              "properties": {"ok": {"type": "boolean"}, "n": {"type": "integer"}}}
    resp, usage = be.complete_json("Reply only with the requested JSON.",
                                   'Return {"ok": true, "n": 7}', schema)
    print("=== RAW RESPONSE ===")
    print(json.dumps(resp, indent=2)[:4000])
    print("\n=== TEXT EXTRACTED ===")
    print(be.text_of(resp))
    print("\n=== USAGE EXTRACTED ===")
    print(json.dumps(usage, indent=2))
    if not usage["usage_fields_found"]:
        print("\n!! No usage path matched. Copy the real field names from the raw "
              "response above into _USAGE_PATHS in agent/backends.py.")


if __name__ == "__main__":
    probe()
