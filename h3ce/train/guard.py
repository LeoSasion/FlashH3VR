"""An enforceable preparation-only scope, separate from the training entry."""
from contextlib import contextmanager
import torch
from h3ce.errors import H3CEError


@contextmanager
def no_training_guard():
    counts = {"optimizer_constructions": 0, "backward_calls": 0, "autograd_grad_calls": 0}
    targets = [(torch.optim.Optimizer, "__init__", "optimizer_constructions"),
               (torch.autograd, "backward", "backward_calls"),
               (torch.autograd, "grad", "autograd_grad_calls")]
    originals = []
    def forbidden(counter):
        def reject(*args, **kwargs):
            counts[counter] += 1
            raise H3CEError("E_CHECK_ONLY_VIOLATION", "Check-only cannot construct optimizers or run backward", counts)
        return reject
    try:
        for owner, name, counter in targets:
            originals.append((owner, name, getattr(owner, name)))
            setattr(owner, name, forbidden(counter))
        with torch.no_grad():
            yield counts
    finally:
        for owner, name, value in originals:
            setattr(owner, name, value)
