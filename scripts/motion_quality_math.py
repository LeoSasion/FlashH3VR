"""Target-relative frame and temporal-difference errors; no optical-flow claims."""
import numpy as np,cv2

def video_errors(prediction,target,foreground):
    assert prediction.shape==target.shape and prediction.ndim==4 and prediction.shape[-1]==3
    assert foreground.shape==prediction.shape[:3] and len(prediction)>1
    assert np.isfinite(prediction).all() and np.isfinite(target).all()
    totals={k:0. for k in ('abs','sq','fg_abs','high_sq','fg_high_sq','edge_sq','temporal_abs','fg_temporal_abs')}
    pixels=fg_pixels=edge_pixels=temporal_pixels=fg_temporal_pixels=0;prev_error=prev_mask=None
    sigma=2*min(prediction.shape[1:3])/512
    for p,y,m in zip(prediction,target,foreground):
        e=p.astype(np.float64)-y.astype(np.float64);mask=m.astype(bool)
        totals['abs']+=np.abs(e).sum();totals['sq']+=(e*e).sum();pixels+=e.size
        totals['fg_abs']+=np.abs(e[mask]).sum();fg_pixels+=int(mask.sum())*3
        hp=p-cv2.GaussianBlur(p,(0,0),sigma,borderType=cv2.BORDER_REFLECT_101)
        hy=y-cv2.GaussianBlur(y,(0,0),sigma,borderType=cv2.BORDER_REFLECT_101)
        he=hp.astype(np.float64)-hy.astype(np.float64)
        totals['high_sq']+=(he*he).sum();totals['fg_high_sq']+=(he[mask]**2).sum()
        for axis in (0,1):
            d=np.diff(e,axis=axis);totals['edge_sq']+=(d*d).sum();edge_pixels+=d.size
        if prev_error is not None:
            td=e-prev_error;union=mask|prev_mask
            totals['temporal_abs']+=np.abs(td).sum();temporal_pixels+=td.size
            totals['fg_temporal_abs']+=np.abs(td[union]).sum();fg_temporal_pixels+=int(union.sum())*3
        prev_error=e;prev_mask=mask
    assert fg_pixels>0 and fg_temporal_pixels>0
    return {'rgb_mae':totals['abs']/pixels,'rgb_mse':totals['sq']/pixels,'foreground_rgb_mae':totals['fg_abs']/fg_pixels,'high_mse':totals['high_sq']/pixels,'foreground_high_mse':totals['fg_high_sq']/fg_pixels,'edge_mse':totals['edge_sq']/edge_pixels,'temporal_difference_mae':totals['temporal_abs']/temporal_pixels,'foreground_temporal_difference_mae':totals['fg_temporal_abs']/fg_temporal_pixels}
