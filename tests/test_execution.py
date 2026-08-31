"""Portable execution profiles: CPU stays reproducible, accelerators are explicit."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.execution import (DeviceCapability, ExecutionError, ExecutionProfile,
                            ExperimentSpec, HardwareCapabilities, detect_capabilities,
                            resolve)

ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1; print(f"  ok   {name}")
    else:
        fail += 1; print(f"  FAIL {name}  {detail}")


def caps(*, mps=False, cuda=False, torch=True):
    return HardwareCapabilities(
        cpu=DeviceCapability(True, name="test cpu"),
        mps=DeviceCapability(mps, "no mps" if not mps else "", name="test mps"),
        cuda=DeviceCapability(cuda, "no cuda" if not cuda else "", name="test cuda",
                              memory_bytes=8 << 30 if cuda else None),
        torch_available=torch, torch_version="test" if torch else None,
        python="test", platform="test")


print("== capability probe is JSON-safe and truthful")
c = detect_capabilities()
raw = json.dumps(c.to_dict())
check("CPU is always available", c.cpu.available)
check("probe serialises without custom encoders", '"cpu"' in raw)
check("torch availability agrees with the current interpreter",
      isinstance(c.torch_available, bool))

print("== NumPy is an explicit CPU execution profile")
n = ExperimentSpec.from_candidate(task="x", config={}, imports=["numpy"],
                                  dependencies=[])
pn = resolve(n, caps(cuda=True, mps=True))
check("NumPy is inferred", n.framework == "numpy")
check("NumPy remains on CPU even when accelerators exist", pn.device == "cpu")
try:
    strict = ExperimentSpec.from_candidate(
        task="x", config={}, imports=["numpy"], dependencies=[],
        operator_profile={"framework": "numpy", "device": "cuda",
                          "allow_fallback": False})
    resolve(strict, caps(cuda=True)); raised = False
except ExecutionError:
    raised = True
check("a strict impossible NumPy/CUDA request is refused", raised)
try:
    resolve(ExperimentSpec.from_candidate(
        task="x", config={}, imports=["numpy"], dependencies=[],
        operator_profile={"accelerator_policy": "require"}), caps(cuda=True))
    raised = False
except ExecutionError:
    raised = True
check("NumPy cannot pretend to satisfy required acceleration", raised)

print("== torch resolves across CUDA, MPS and CPU")
t = ExperimentSpec.from_candidate(task="x", config={}, imports=["torch"],
                                  dependencies=["torch"])
check("torch is inferred from source/declaration", t.framework == "torch")
check("auto prefers CUDA", resolve(t, caps(cuda=True, mps=True)).device == "cuda")
check("auto uses MPS when CUDA is absent", resolve(t, caps(mps=True)).device == "mps")
check("auto falls back to CPU", resolve(t, caps()).device == "cpu")

req = ExperimentSpec.from_candidate(
    task="x", config={}, imports=["torch"], dependencies=["torch"],
    operator_profile={"framework": "torch", "device": "cuda",
                      "allow_fallback": True})
p = resolve(req, caps(mps=True))
check("an unavailable explicit device records its fallback",
      p.device == "mps" and p.fallback_used and "unavailable" in p.fallback_reason,
      str(p.to_dict()))

try:
    req2 = ExperimentSpec.from_candidate(
        task="x", config={}, imports=["torch"], dependencies=["torch"],
        operator_profile={"framework": "torch", "device": "cuda",
                          "allow_fallback": False})
    resolve(req2, caps()); raised = False
except ExecutionError:
    raised = True
check("strict unavailable CUDA is refused", raised)
try:
    conflict = ExperimentSpec.from_candidate(
        task="x", config={}, imports=["torch"], dependencies=["torch"],
        operator_profile={"device": "cpu", "accelerator_policy": "require"})
    resolve(conflict, caps(cuda=True)); raised = False
except ExecutionError:
    raised = True
check("required acceleration conflicts with explicit CPU", raised)

try:
    resolve(t, caps(torch=False)); raised = False
except ExecutionError:
    raised = True
check("a torch candidate is refused when torch is absent", raised)

print("== candidate and operator requests merge safely")
e = ExperimentSpec.from_candidate(
    task="x",
    config={"execution": {"framework": "torch", "device": "mps",
                           "max_memory_mb": 2048}},
    imports=["torch"], dependencies=["torch"],
    operator_profile={"device": "cpu", "accelerator_policy": "off"})
check("an explicit operator device overrides the candidate", e.execution.device == "cpu")
check("candidate resource limits survive", e.execution.max_memory_mb == 2048)
try:
    ExecutionProfile.from_value({"shell": "anything"}); raised = False
except ExecutionError:
    raised = True
check("unknown execution fields are refused", raised)

print("== a candidate may narrow operator constraints, never widen them")
# config.json is written by the LLM. Before this was enforced, a candidate could turn
# accelerator_policy 'off' into 'require', flip allow_fallback back on, and raise both
# resource ceilings -- i.e. generated content granting itself hardware.
esc = ExperimentSpec.from_candidate(
    task="x",
    config={"execution": {"allow_fallback": True, "accelerator_policy": "require",
                          "max_memory_mb": 65536, "max_fit_seconds": 99999}},
    imports=["torch"], dependencies=["torch"],
    operator_profile={"accelerator_policy": "off", "allow_fallback": False,
                      "max_memory_mb": 1024, "max_fit_seconds": 60})
x = esc.execution
check("a candidate cannot escalate accelerator_policy", x.accelerator_policy == "off",
      repr(x.accelerator_policy))
check("a candidate cannot re-enable operator-disabled fallback", x.allow_fallback is False)
check("a candidate cannot raise the memory ceiling", x.max_memory_mb == 1024,
      str(x.max_memory_mb))
check("a candidate cannot raise the fit-time ceiling", x.max_fit_seconds == 60,
      str(x.max_fit_seconds))
check("an operator that forbade accelerators still gets CPU when CUDA exists",
      resolve(esc, caps(cuda=True)).device == "cpu")
narrow = ExperimentSpec.from_candidate(
    task="x", config={"execution": {"accelerator_policy": "off", "max_memory_mb": 256}},
    imports=["torch"], dependencies=["torch"],
    operator_profile={"accelerator_policy": "prefer", "max_memory_mb": 1024})
check("narrowing in the safe direction is still allowed",
      narrow.execution.accelerator_policy == "off"
      and narrow.execution.max_memory_mb == 256)
nul = ExperimentSpec.from_candidate(
    task="x", config={"execution": {"max_memory_mb": None}},
    imports=["torch"], dependencies=["torch"], operator_profile={"max_memory_mb": 512})
check("a null resource request cannot erase an operator limit",
      nul.execution.max_memory_mb == 512, str(nul.execution.max_memory_mb))

print("== 'require' is never satisfied by CPU")
try:
    resolve(ExperimentSpec.from_candidate(
        task="x", config={}, imports=["torch"], dependencies=["torch"],
        operator_profile={"device": "mps", "accelerator_policy": "require",
                          "allow_fallback": True}), caps())
    raised = False
except ExecutionError:
    raised = True
check("required acceleration is refused rather than falling back to CPU", raised)
check("required acceleration still falls back between accelerators",
      resolve(ExperimentSpec.from_candidate(
          task="x", config={}, imports=["torch"], dependencies=["torch"],
          operator_profile={"device": "mps", "accelerator_policy": "require",
                            "allow_fallback": True}), caps(cuda=True)).device == "cuda")

env = resolve(t, caps(cuda=True)).environment
check("the child receives only declarative execution variables",
      set(env) == {"TRACK2_FRAMEWORK", "TRACK2_DEVICE", "TRACK2_EXECUTION_PROFILE"})

print("== Docker receives executable inputs and accelerator declaration")
from core import isolation as ISO
with tempfile.TemporaryDirectory(prefix="track2-docker-argv-") as td:
    root = Path(td)
    data = root / "data"; data.mkdir()
    (data / "events.csv").write_text("x\n1\n")
    (data / ISO.PORTABLE_SPEC_NAME).write_text("{}\n")
    candidate = root / "candidate"; candidate.mkdir()
    workspace = root / "workspace"; workspace.mkdir()
    out = root / "out"; out.mkdir()
    dr = ISO.DataRoot(data, ("events.csv", ISO.PORTABLE_SPEC_NAME), (), False)
    cmd = ISO._docker_argv(
        "cuda-image", dr, out,
        ["python", "-B", "-m", "strategy.candidate_runner",
         "--candidate", str(candidate), "--workspace", str(workspace),
         "--task-spec", str(data / ISO.PORTABLE_SPEC_NAME),
         "--out-dir", str(out)],
        {"TRACK2_FRAMEWORK": "torch", "TRACK2_DEVICE": "cuda",
         "TRACK2_EXECUTION_PROFILE": '{"device":"cuda"}'})
    joined = " ".join(cmd)
    check("CUDA container requests GPU passthrough", "--gpus all" in joined)
    check("candidate source is mounted read-only", f"{candidate}:/work/candidate:ro" in joined)
    check("immutable contract workspace is mounted read-only",
          f"{workspace}:/work/workspace:ro" in joined)
    check("portable task spec resolves inside redacted data root",
          cmd[cmd.index("--task-spec") + 1] == f"/work/data/{ISO.PORTABLE_SPEC_NAME}")
    check("writable host output is translated to container output",
          cmd[cmd.index("--out-dir") + 1] == ISO.CONTAINER_OUT)
    check("resolved device is forwarded into the container",
          "TRACK2_DEVICE=cuda" in joined)

print(f"\n{ok} passed, {fail} failed")
raise SystemExit(1 if fail else 0)
