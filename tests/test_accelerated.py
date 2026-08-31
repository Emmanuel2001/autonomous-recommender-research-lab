"""The optional PyTorch profile is a real CPU/MPS/CUDA candidate, not a UI label."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import candidate as C
from agent import deps as D
from core import execution as E
from research_workspace.model import FM

ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1; print(f"  ok   {name}")
    else:
        fail += 1; print(f"  FAIL {name}  {detail}")


ws = C.workspace_for("accelerated")
tree = C.seed_tree(ws)
meta = C.validate_tree(tree, ws)
src = "\n".join(tree.values())
declared = D.parse_requires(src)

print("== accelerated seed is a valid, disclosed candidate")
check("profile exists", ws.name == "research_workspace_accelerated")
check("tree validates", len(meta["tree"]) == 64)
check("the optional dependency is declared", declared == ["torch"], str(declared))
check("execution request is explicit",
      meta["config"]["execution"]["framework"] == "torch")
prov = C.profile_provenance("accelerated")
check("it is not mislabelled as the official NumPy seed",
      prov["official_numpy_seed"] is False)
check("the supplied alternative architecture is disclosed",
      prov["alternative_architecture_code_present"] is True)

print("== PyTorch FM forward is numerically the same architecture")
spec = importlib.util.spec_from_file_location("track2_accelerated_model", ws / "model.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
rng = np.random.default_rng(9)
x = rng.integers(0, 37, size=(41, 5), dtype=np.int64)
npfm = FM(37, k=8, seed=4)
tfm = mod.TorchFM(37, k=8, seed=4)
got = tfm.predict_numpy(x, "cpu")
want = npfm.predict(x)
check("CPU forward matches NumPy", np.allclose(got, want, atol=2e-7),
      f"max diff {np.max(np.abs(got-want))}")

print("== resolved device executes a real tensor operation")
experiment = E.ExperimentSpec.from_candidate(
    task="kuairand_pure", config=meta["config"], imports=meta["imports"],
    dependencies=declared)
plan = E.resolve(experiment)
check("framework resolves to torch", plan.framework == "torch")
check("resolved device is one of the enforced choices",
      plan.device in ("cpu", "mps", "cuda"), plan.device)
try:
    device_model = mod.TorchFM(37, k=8, seed=4).to(plan.device)
    device_out = device_model.predict_numpy(x, plan.device)
    executed = device_out.shape == (41,) and np.isfinite(device_out).all()
except Exception as exc:
    executed = False; detail = f"{type(exc).__name__}: {exc}"
else:
    detail = plan.device
check(f"a real {plan.device} forward pass succeeds", executed, detail)

print("== training is differentiable on the resolved device")
import torch
device_model = mod.TorchFM(37, k=8, seed=4).to(plan.device)
opt = torch.optim.Adam(device_model.parameters(), lr=0.01)
xt = torch.as_tensor(x, dtype=torch.long, device=plan.device)
y = torch.as_tensor(rng.integers(0, 2, len(x)), dtype=torch.float32, device=plan.device)
before = torch.nn.functional.binary_cross_entropy_with_logits(device_model(xt), y)
for _ in range(8):
    opt.zero_grad(set_to_none=True)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(device_model(xt), y)
    loss.backward(); opt.step()
after = torch.nn.functional.binary_cross_entropy_with_logits(device_model(xt), y)
check("optimizer changes the recommender parameters",
      float(after.detach()) < float(before.detach()),
      f"{float(before.detach()):.5f} -> {float(after.detach()):.5f}")

print(f"\n{ok} passed, {fail} failed")
raise SystemExit(1 if fail else 0)
