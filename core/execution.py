"""Portable execution contracts for agent-written recommender experiments.

The research agent chooses an experiment; this module chooses where the numerical
training can run.  Keeping those decisions separate lets the same research loop use a
small NumPy CPU baseline, Apple Metal on a laptop, or CUDA on a Linux worker without
teaching the LLM host-specific shell commands.

Nothing here imports torch at module import time.  The minimal, reproducible submission
therefore remains NumPy-only; torch is an optional candidate dependency.
"""
from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field, fields
from typing import Any


FRAMEWORKS = ("auto", "numpy", "torch")
DEVICES = ("auto", "cpu", "mps", "cuda")
ACCELERATOR_POLICIES = ("off", "allow", "prefer", "require")
# How much acceleration each policy asks for. A candidate may move DOWN this
# ladder, never up: `config.json` is written by the LLM, so treating it as able
# to raise an operator ceiling would let generated content grant itself hardware.
_POLICY_LEVEL = {"off": 0, "allow": 1, "prefer": 2, "require": 3}


class ExecutionError(ValueError):
    """An execution request that this machine cannot satisfy safely."""


@dataclass(frozen=True)
class DeviceCapability:
    available: bool
    reason: str = ""
    name: str = ""
    memory_bytes: int | None = None


@dataclass(frozen=True)
class HardwareCapabilities:
    cpu: DeviceCapability
    mps: DeviceCapability
    cuda: DeviceCapability
    torch_available: bool
    torch_version: str | None
    python: str
    platform: str

    def to_dict(self) -> dict:
        return asdict(self)


def detect_capabilities() -> HardwareCapabilities:
    """Measure this process's devices.  Failure to import torch is data, not a crash."""
    cpu = DeviceCapability(True, name=platform.processor() or platform.machine() or "CPU")
    torch_available = False
    torch_version = None
    mps = DeviceCapability(False, "torch is not installed")
    cuda = DeviceCapability(False, "torch is not installed")
    try:
        import torch  # optional; deliberately local
        torch_available = True
        torch_version = str(getattr(torch, "__version__", "unknown"))

        mb = getattr(getattr(torch, "backends", None), "mps", None)
        if mb is None or not bool(mb.is_built()):
            mps = DeviceCapability(False, "this torch build has no MPS support")
        elif not bool(mb.is_available()):
            mps = DeviceCapability(
                False, "MPS is built but unavailable to this process/OS")
        else:
            mps = DeviceCapability(True, name="Apple Metal Performance Shaders")

        if not bool(torch.cuda.is_available()):
            cuda = DeviceCapability(False, "CUDA is unavailable to this process")
        else:
            idx = int(torch.cuda.current_device())
            prop = torch.cuda.get_device_properties(idx)
            cuda = DeviceCapability(
                True, name=str(torch.cuda.get_device_name(idx)),
                memory_bytes=int(getattr(prop, "total_memory", 0)) or None)
    except Exception as exc:  # an incompatible optional install must not break the harness
        mps = DeviceCapability(False, f"torch probe failed: {type(exc).__name__}: {exc}")
        cuda = DeviceCapability(False, f"torch probe failed: {type(exc).__name__}: {exc}")

    return HardwareCapabilities(
        cpu=cpu, mps=mps, cuda=cuda,
        torch_available=torch_available, torch_version=torch_version,
        python=platform.python_version(), platform=platform.platform())


@dataclass(frozen=True)
class ExecutionProfile:
    """Operator constraints.  A candidate can request less, never silently more."""
    framework: str = "auto"
    device: str = "auto"
    accelerator_policy: str = "prefer"
    allow_fallback: bool = True
    max_memory_mb: int | None = None
    max_fit_seconds: float | None = None
    dtype: str = "float32"

    def __post_init__(self):
        if self.framework not in FRAMEWORKS:
            raise ExecutionError(
                f"framework {self.framework!r} is not one of {list(FRAMEWORKS)}")
        if self.device not in DEVICES:
            raise ExecutionError(f"device {self.device!r} is not one of {list(DEVICES)}")
        if self.accelerator_policy not in ACCELERATOR_POLICIES:
            raise ExecutionError(
                f"accelerator_policy {self.accelerator_policy!r} is not one of "
                f"{list(ACCELERATOR_POLICIES)}")
        if self.max_memory_mb is not None and self.max_memory_mb <= 0:
            raise ExecutionError("max_memory_mb must be positive")
        if self.max_fit_seconds is not None and self.max_fit_seconds <= 0:
            raise ExecutionError("max_fit_seconds must be positive")
        if self.dtype not in ("float32", "float64", "bfloat16", "float16"):
            raise ExecutionError(f"unsupported dtype {self.dtype!r}")

    @classmethod
    def from_value(cls, value: Any) -> "ExecutionProfile":
        if value is None or value == "":
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                value = json.loads(value) if value.lstrip().startswith("{") else {
                    "device": value}
            except json.JSONDecodeError as exc:
                raise ExecutionError(f"execution profile is not valid JSON: {exc}") from None
        if not isinstance(value, dict):
            raise ExecutionError("execution profile must be an object, device name, or null")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ExecutionError(f"unknown execution-profile fields {unknown}")
        return cls(**value)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentSpec:
    """The machine-readable execution portion of one agent-proposed experiment."""
    task: str
    framework: str
    execution: ExecutionProfile
    dependencies: tuple[str, ...] = ()
    imports: tuple[str, ...] = ()
    hypothesis: str = ""
    lane: str = ""
    resources: dict = field(default_factory=dict)

    @classmethod
    def from_candidate(cls, *, task: str, config: dict, imports: list[str],
                       dependencies: list[str], operator_profile: Any = None,
                       hypothesis: str = "", lane: str = "") -> "ExperimentSpec":
        raw_requested = config.get("execution") or {}
        if isinstance(raw_requested, str):
            raw_requested = {"device": raw_requested}
        if not isinstance(raw_requested, dict):
            raise ExecutionError("config.execution must be an object or device name")
        requested = dict(raw_requested)
        operator = ExecutionProfile.from_value(operator_profile)
        # The operator's values are hard ceilings; the candidate may narrow them and declare
        # what its implementation needs, but may never widen them. Every branch below exists
        # because the un-narrowed version let a candidate turn `accelerator_policy: off` into
        # `require`, flip `allow_fallback` back on, and raise both resource ceilings.
        merged = operator.to_dict()
        for key, value in requested.items():
            if key not in merged:
                raise ExecutionError(f"candidate requested unknown execution field {key!r}")
            if key in ("framework", "device"):
                if merged[key] != "auto":
                    continue            # an explicit operator choice wins outright
                merged[key] = value
            elif key == "accelerator_policy":
                if value not in ACCELERATOR_POLICIES:
                    raise ExecutionError(
                        f"accelerator_policy {value!r} is not one of "
                        f"{list(ACCELERATOR_POLICIES)}")
                if _POLICY_LEVEL[value] < _POLICY_LEVEL[merged[key]]:
                    merged[key] = value
            elif key == "allow_fallback":
                merged[key] = bool(value) and bool(merged[key])
            elif key in ("max_memory_mb", "max_fit_seconds"):
                if value is None:
                    continue            # "no limit" cannot erase an operator limit
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ExecutionError(f"{key} must be a number, got {type(value).__name__}")
                merged[key] = value if merged[key] is None else min(value, merged[key])
            else:
                merged[key] = value
        inferred = "torch" if ("torch" in imports or "torch" in dependencies) else "numpy"
        if merged["framework"] == "auto":
            merged["framework"] = inferred
        profile = ExecutionProfile.from_value(merged)
        return cls(task=task, framework=profile.framework, execution=profile,
                   dependencies=tuple(dependencies), imports=tuple(imports),
                   hypothesis=hypothesis, lane=lane,
                   resources={k: v for k, v in requested.items()
                              if k in ("max_memory_mb", "max_fit_seconds")})

    def to_dict(self) -> dict:
        out = asdict(self)
        out["dependencies"] = list(self.dependencies)
        out["imports"] = list(self.imports)
        return out


@dataclass(frozen=True)
class ExecutionPlan:
    framework: str
    requested_device: str
    device: str
    accelerator_policy: str
    fallback_used: bool
    fallback_reason: str
    capabilities: HardwareCapabilities
    environment: dict

    def to_dict(self) -> dict:
        out = asdict(self)
        return out


def resolve(spec: ExperimentSpec,
            capabilities: HardwareCapabilities | None = None) -> ExecutionPlan:
    """Resolve a candidate request to one real device, or refuse it explicitly."""
    cap = capabilities or detect_capabilities()
    p = spec.execution
    framework = spec.framework
    requested = p.device

    if framework == "numpy":
        if p.accelerator_policy == "require":
            raise ExecutionError(
                "accelerator_policy='require' cannot be satisfied by the NumPy CPU "
                "candidate; use a torch implementation")
        if requested not in ("auto", "cpu") and not p.allow_fallback:
            raise ExecutionError(
                f"NumPy candidates execute on CPU; requested {requested!r} with fallback off")
        device = "cpu"
        fallback = requested not in ("auto", "cpu")
        reason = (f"NumPy has no {requested} execution backend in this harness"
                  if fallback else "")
    else:
        if not cap.torch_available:
            raise ExecutionError("the candidate requires torch, but torch is unavailable")

        available = {"cpu": cap.cpu.available, "mps": cap.mps.available,
                     "cuda": cap.cuda.available}
        if p.accelerator_policy == "require" and requested == "cpu":
            raise ExecutionError(
                "accelerator_policy='require' conflicts with device='cpu'")
        if p.accelerator_policy == "off":
            choices = ["cpu"]
        elif requested != "auto":
            choices = [requested]
        elif p.accelerator_policy in ("prefer", "require"):
            choices = ["cuda", "mps"] + ([] if p.accelerator_policy == "require" else ["cpu"])
        else:  # allow: choose the cheapest deterministic default unless explicitly asked
            choices = ["cpu", "cuda", "mps"]

        device = next((d for d in choices if available.get(d)), None)
        fallback = False
        reason = ""
        if device is None and requested != "auto" and p.allow_fallback:
            # The fallback pool must honour the policy that got us here: falling back to CPU
            # under accelerator_policy='require' silently satisfies a request for hardware
            # acceleration with no acceleration at all.
            pool = (("cuda", "mps") if p.accelerator_policy == "require"
                    else ("cuda", "mps", "cpu"))
            device = next((d for d in pool if available.get(d)), None)
            fallback = device is not None
            reason = f"requested {requested} is unavailable; resolved to {device}"
        if device is None:
            reasons = {"mps": cap.mps.reason, "cuda": cap.cuda.reason}
            raise ExecutionError(
                f"no device satisfies framework={framework}, requested={requested}, "
                f"policy={p.accelerator_policy}; availability={reasons}")
        if requested not in ("auto", device):
            fallback = True
            reason = reason or f"requested {requested} is unavailable; resolved to {device}"

    env = {
        "TRACK2_FRAMEWORK": framework,
        "TRACK2_DEVICE": device,
        "TRACK2_EXECUTION_PROFILE": json.dumps(p.to_dict(), sort_keys=True),
    }
    return ExecutionPlan(framework, requested, device, p.accelerator_policy,
                         fallback, reason, cap, env)


def public_capabilities() -> dict:
    """Small JSON-safe view for prompts and the local UI."""
    c = detect_capabilities().to_dict()
    # Environment paths and keys are intentionally absent; this is safe to show to an LLM.
    return c


if __name__ == "__main__":
    print(json.dumps(public_capabilities(), indent=2, sort_keys=True))
