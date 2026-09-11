"""Official-style x4 tiled inference and explicit same-canvas image metrics."""
import torch
from torch.nn import functional as F
from h3ce.train.preflight import require
from h3ce.train.losses import region_error
from scripts.detail_supervision_math import detail_loss


def tile_grid(height,width,tile=256,overlap=32):
    hp=(height//8+1)*8;wp=(width//8+1)*8
    size=min(tile,hp,wp);require(size%8==0 and 0<=overlap<size,'Invalid official-style tile geometry')
    ys=list(range(0,hp-size,size-overlap))+[hp-size];xs=list(range(0,wp-size,size-overlap))+[wp-size]
    return {'original_hw':[height,width],'padded_hw':[hp,wp],'tile':size,'overlap':overlap,'positions':[[y,x] for y in ys for x in xs]}


def tiled_restore(model,x,budget_check=lambda:None,tile=256,overlap=32):
    require(x.ndim==4 and x.shape[:2]==(1,3) and x.dtype==torch.float32 and torch.isfinite(x).all(),'Expected one FP32 RGB image')
    h,w=x.shape[-2:];grid=tile_grid(h,w,tile,overlap);hp,wp=grid['padded_hw'];size=grid['tile']
    padded=torch.cat((x,torch.flip(x,[2])),2)[:,:,:hp,:]
    padded=torch.cat((padded,torch.flip(padded,[3])),3)[:,:,:,:wp]
    summed=torch.zeros(1,3,hp*4,wp*4,device=x.device,dtype=x.dtype);counts=torch.zeros_like(summed)
    for yy,xx in grid['positions']:
        budget_check();prediction=model(padded[...,yy:yy+size,xx:xx+size])
        require(prediction.shape==(1,3,size*4,size*4) and torch.isfinite(prediction).all(),'Official model output contract differs')
        summed[...,yy*4:(yy+size)*4,xx*4:(xx+size)*4].add_(prediction)
        counts[...,yy*4:(yy+size)*4,xx*4:(xx+size)*4].add_(1.)
    require((counts>0).all(),'Tiling left uncovered pixels')
    native=(summed/counts)[...,:h*4,:w*4].contiguous()
    working=F.interpolate(native,size=(h,w),mode='area')
    return working,native,grid


def image_metrics(prediction,sample):
    require(prediction.shape==sample['y'].shape and torch.isfinite(prediction).all(),'Invalid restored image')
    error=(prediction-sample['y']).float();valid=sample['valid'];result={}
    for name,mask in (('global',valid),('person',valid*sample['person_mask']),('face',valid*sample['face_mask'])):
        result['rgb_'+name+'_mae']=float(region_error(error.abs(),mask)) if mask.sum()>0 else None
    result['detail']=float(detail_loss(prediction,sample['y'],valid,sample['person_mask'],sample['face_mask']))
    result['out_of_range_fraction']=float(region_error(((prediction<0)|(prediction>1)).float(),valid))
    return result
