"""Scale gradients through the frozen FP16 H3 decoder before FP32 clipping."""
import math
import torch
from .preflight import require

POLICY = {"name":"native_fp16_backward_dynamic_scale_v1", "pixel_enabled":True,
          "latent_enabled":False, "init_scale":65536., "growth_factor":2.,
          "backoff_factor":.5,"growth_interval":2000,"max_consecutive_overflows":16}


def make_scaler(phase, device="cuda"):
    return torch.amp.GradScaler(device, enabled=phase in {"overfit","pixel"},
        init_scale=POLICY["init_scale"],growth_factor=POLICY["growth_factor"],
        backoff_factor=POLICY["backoff_factor"],growth_interval=POLICY["growth_interval"])


def unscale_and_check(model, optimizer, scaler):
    scaler.unscale_(optimizer)
    present = [p.grad for p in model.parameters() if p.grad is not None]
    require(present,"No trainable parameter received a gradient", "E_GRADIENT_CONTRACT")
    return all(bool(torch.isfinite(g).all()) for g in present)


def checked_scaler_step(optimizer, scaler, *, expected_update):
    """Observe actual optimizer invocation; AdamW.step() returning None is ambiguous."""
    calls=[]
    handle=optimizer.register_step_post_hook(lambda *_: calls.append(True))
    before=float(scaler.get_scale())
    try:
        scaler.step(optimizer)
    finally:
        handle.remove()
    require(len(calls)==int(expected_update),"Scaler update disagrees with finite-gradient decision", "E_GRADIENT_CONTRACT")
    scaler.update()
    after=float(scaler.get_scale())
    require(math.isfinite(after) and after>0,"Loss scale is not positive and finite", "E_NONFINITE_GRADIENT")
    if not expected_update:
        require(scaler.is_enabled() and after<before,"Nonfinite gradients require a skipped update and scale backoff", "E_NONFINITE_GRADIENT")
    return {"loss_scale_before":before,"loss_scale_after":after,"optimizer_updated":expected_update}
