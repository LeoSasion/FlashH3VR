"""Instrumented native PyTorch L-BFGS; the numerical algorithm is unchanged."""
import torch
from contextlib import contextmanager
from h3ce.train.preflight import require


@contextmanager
def strict_fp32_refiner():
    old=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:yield
    finally:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=old


class CountedLBFGS(torch.optim.LBFGS):
    def __init__(self,params,**kwargs):
        super().__init__(params,**kwargs)
        self.trial_writes=0;self.accepted_writes=0;self.in_trial=False

    def _directional_evaluate(self,closure,x,t,d):
        require(not self.in_trial,'Nested line-search evaluation')
        self.in_trial=True
        try:return super()._directional_evaluate(closure,x,t,d)
        finally:self.in_trial=False

    def _add_grad(self,step_size,update):
        super()._add_grad(step_size,update)
        if self.in_trial:self.trial_writes+=1
        else:self.accepted_writes+=1


class FullBatchState:
    pending_window=False
    def __init__(self,indices):self.indices=list(indices);self.completed_calls=0
    def state_dict(self):return {'kind':'fixed_full_batch','indices':self.indices,'completed_calls':self.completed_calls}


def accumulate_full_batch(model,optimizer,indices,term_for):
    optimizer.zero_grad(set_to_none=True);rows=[];total=None
    for index in indices:
        loss=term_for(index)
        require(loss.ndim==0 and torch.isfinite(loss),'Nonfinite full-batch term')
        (loss/len(indices)).backward()
        rows.append({'index':index,'latent_mse':float(loss.detach())})
        total=loss.detach()/len(indices) if total is None else total+loss.detach()/len(indices)
    require(all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters()),'Nonfinite or disconnected full-batch gradient')
    return total,rows
