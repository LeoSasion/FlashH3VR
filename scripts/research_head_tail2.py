"""Shared frozen Restormer prefix and two pretrained trainable refinement blocks."""
import copy
from contextlib import contextmanager
import numpy as np,torch
from h3ce.train.losses import charbonnier,region_error,lighting_target_loss
from h3ce.train.pipeline import outer_box_feather
from scripts.detail_supervision_math import detail_loss,gaussian_lowpass_rectangle

@contextmanager
def strict_spatial():
    prior=(torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    try:yield
    finally:torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32=prior

class HeadTail(torch.nn.Module):
    def __init__(self,backbone):
        super().__init__();self.backbone=backbone.eval().requires_grad_(False)
        assert len(backbone.refinement)==4 and backbone.output.in_channels==96 and backbone.output.out_channels==3
        self.head=torch.nn.Sequential(copy.deepcopy(backbone.refinement[-2]),copy.deepcopy(backbone.refinement[-1]),copy.deepcopy(backbone.output)).float().requires_grad_(True)
        assert sum(p.numel() for p in self.head.parameters())==238334
    def features(self,x):
        captured=[];baseline=[]
        h1=self.backbone.refinement[-2].register_forward_pre_hook(lambda m,a:captured.append(a[0].detach()))
        h2=self.backbone.output.register_forward_hook(lambda m,a,y:baseline.append(y.detach()))
        try:
            with torch.no_grad(),strict_spatial():self.backbone(x)
        finally:h1.remove();h2.remove()
        assert len(captured)==len(baseline)==1
        return {'features':captured[0],'baseline':baseline[0]}
    def spatial(self,x,cache):
        with strict_spatial():return x+(self.head(cache['features'])-cache['baseline'])

def sample_for(case,device='cpu'):
    with np.load(case['path']) as a:
        x,y=(torch.from_numpy(a[k].copy()).permute(2,0,1)[None,:,None].to(device) for k in ('x','y'))
        valid,face=(torch.from_numpy(a[k].copy())[None,None,None].to(device) for k in ('valid','face_mask'))
    return {'x':x,'y':y,'valid':valid,'face_mask':face,'clean':case['clean'],'head_index':case['head_index'],'purpose':case['purpose']}

def compose(model,sample,cache):
    spatial=model.spatial(sample['x'][:,:,0],cache).unsqueeze(2)
    return sample['x']+outer_box_feather(sample['valid'])*(spatial-sample['x']),spatial

def objective(pred,sample):
    err=charbonnier(pred-sample['y']);valid=sample['valid'];face=valid*sample['face_mask']
    rgb=region_error(err,valid)+region_error(err,face)
    light=lighting_target_loss(pred,sample['y'],valid)
    detail=detail_loss(pred,sample['y'],valid,None,sample['face_mask'])
    raw=rgb+.2*light+.5*detail;weight=2. if sample['clean'] else 1.
    return {'rgb':rgb,'lighting':light,'detail':detail,'unweighted_total':raw,'sample_weight':raw.new_tensor(weight),'total':raw*weight}

def metrics(pred,sample):
    points=torch.nonzero(sample['valid'][0,0,0]);y0,x0=points.amin(0).tolist();y1,x1=(points.amax(0)+1).tolist()
    err=(pred-sample['y'])[:,:,0,y0:y1,x0:x1];mask=sample['face_mask'][:,:,0,y0:y1,x0:x1]
    high=err-gaussian_lowpass_rectangle(err,sigma=2*min(y1-y0,x1-x0)/512)
    dy,dx=err[:,:,1:]-err[:,:,:-1],err[:,:,:,1:]-err[:,:,:,:-1]
    return {'rgb_mae':float(err.abs().mean()),'face_mae':float((err.abs()*mask).sum()/(mask.sum()*3)),
            'high_mse':float(high.square().mean()),'face_high_mse':float((high.square()*mask).sum()/(mask.sum()*3)),
            'edge_mse':float((dy.square().sum()+dx.square().sum())/(dy.numel()+dx.numel()))}
