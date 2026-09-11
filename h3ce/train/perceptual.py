"""Single frozen LPIPS-Alex, with explicit offline loading and strict state coverage."""
from __future__ import annotations
import importlib.metadata
import importlib.util
from pathlib import Path
import torch
from h3ce.components import read_component_lock,inspect_component,component_path,sha256_file
from h3ce.errors import H3CEError

def require(value,message):
    if not value:raise H3CEError('E_PERCEPTUAL_COMPONENT',message)

def verify_perceptual_entry(entry,root):
    require(entry is not None and entry.get('id')=='lpips_alex','Missing LPIPS-Alex lock')
    require(not inspect_component(entry,root)['blockers'],'LPIPS-Alex calibration is not locally locked')
    require(entry['provider']=='richzhang/PerceptualSimilarity' and entry['architecture']=='lpips-alex'
            and entry['code_revision']=='lpips==0.1.4','Unexpected perceptual implementation')
    require(importlib.metadata.version('lpips')=='0.1.4'
            and importlib.metadata.version('torchvision')==entry['torchvision_version'],'Perceptual runtime version changed')
    backbone=entry.get('backbone',{});path=(root/str(backbone.get('path',''))).resolve()
    require(path.is_relative_to(root/'models') and path.is_file(),'Missing local perceptual backbone')
    require(backbone.get('source_url')=='https://download.pytorch.org/models/alexnet-owt-7be5be79.pth'
            and sha256_file(path)==backbone.get('sha256'),'Backbone source or checksum mismatch')
    package=Path(importlib.util.find_spec('lpips').origin).parent
    torchvision=Path(importlib.util.find_spec('torchvision').origin).parent
    expected={str(p) for p in [*sorted(package.glob('*.py')),torchvision/'models/alexnet.py']}
    hashes=entry.get('runtime_files_sha256',{})
    require(set(hashes)==expected,'Perceptual runtime file coverage differs')
    require(all(Path(p).is_file() and sha256_file(p)==h for p,h in hashes.items()),'Perceptual source checksum mismatch')
    return entry

def load_perceptual(config,root,*,device='cuda',force=False):
    if not config.training.losses.perceptual and not force:return None
    root=Path(root).resolve()
    entry=verify_perceptual_entry(read_component_lock(root/config.paths.components_lock).get('lpips_alex'),root)
    import lpips
    # Upstream constructors initialize before loading; isolate that RNG use from training.
    with torch.random.fork_rng(devices=[]):
        model=lpips.LPIPS(net='alex',version='0.1',pretrained=False,pnet_rand=True,
                          pnet_tune=False,use_dropout=True,eval_mode=True,verbose=False)
    backbone=torch.load(root/entry['backbone']['path'],map_location='cpu',weights_only=True)
    current=model.net.state_dict();mapped={}
    for key in current:
        parts=key.split('.');require(len(parts)==3 and parts[0] in {'slice1','slice2','slice3','slice4','slice5'},'Unknown upstream AlexNet slice layout')
        source='features.'+'.'.join(parts[1:]);require(source in backbone,'Missing pretrained feature tensor')
        mapped[key]=backbone[source]
    require(len(mapped)==10 and {'features.'+'.'.join(k.split('.')[1:]) for k in mapped}=={k for k in backbone if k.startswith('features.')},'Incomplete feature tensor coverage')
    model.net.load_state_dict(mapped,strict=True)
    calibration=torch.load(component_path(entry,root),map_location='cpu',weights_only=True)
    expected={f'lin{i}.model.1.weight' for i in range(5)}
    require(set(calibration)==expected,'Unknown LPIPS calibration tensor keys')
    for i in range(5):
        weight=calibration[f'lin{i}.model.1.weight']
        require(torch.isfinite(weight).all() and (weight>=0).all(),'Invalid calibration tensor')
        model.lins[i].load_state_dict({'model.1.weight':weight},strict=True)
    require(all(torch.isfinite(p).all() for p in model.parameters()),'Nonfinite perceptual parameters')
    model.pnet_rand=False  # Every feature parameter has now been replaced by verified pretrained weights.
    model.eval().requires_grad_(False)
    model.h3ce_component_sha256=entry['sha256']
    return model.to(device=device,dtype=torch.float32)

def application_loss(config,root,*,device='cuda'):
    from .losses import ApplicationLoss
    return ApplicationLoss(config.training.losses,perceptual=load_perceptual(config,root,device=device))
