# Publication change 2026-09-12: use repository-relative asset provenance.
"""Load the pinned official NAFNetLocal model without installing a second BasicSR."""
from pathlib import Path
import sys,ast,types,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,yaml
from h3ce.cache.keys import file_sha256
REV='2b4af71ebe098a92a75910c233a3965a3e93ede4'
SOURCE=ROOT/'third_party'/('NAFNet-'+REV);ASSET=ROOT/'models/nafnet_gopro32_baseline'

def load_model():
 provenance=json.loads((ROOT/'configs/nafnet_gopro32.provenance.json').read_text(encoding='utf-8'));assert provenance['revision']==REV
 for f in provenance['files']:assert file_sha256(ROOT/f['path'])==f['sha256'],f['path']
 # Keep official class bodies unchanged. Only their two BasicSR import bindings
 # are supplied locally, avoiding imports/installations unrelated to this model.
 norm_path=SOURCE/'basicsr/models/archs/arch_util.py';tree=ast.parse(norm_path.read_text());nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('LayerNormFunction','LayerNorm2d')];assert [n.name for n in nodes]==['LayerNormFunction','LayerNorm2d'];norm={'torch':torch,'nn':torch.nn,'__name__':'research_nafnet_norm'};exec(compile(ast.Module(nodes,type_ignores=[]),str(norm_path),'exec'),norm)
 local_path=SOURCE/'basicsr/models/archs/local_arch.py';local={'__name__':'research_nafnet_local'};exec(compile(local_path.read_text(),str(local_path),'exec'),local)
 arch_path=SOURCE/'basicsr/models/archs/NAFNet_arch.py';tree=ast.parse(arch_path.read_text());removed=[n for n in tree.body if isinstance(n,ast.ImportFrom) and n.module in ('basicsr.models.archs.arch_util','basicsr.models.archs.local_arch')];assert len(removed)==2;tree.body=[n for n in tree.body if n not in removed];scope={'__name__':'research_nafnet_arch','LayerNorm2d':norm['LayerNorm2d'],'Local_Base':local['Local_Base']};exec(compile(tree,str(arch_path),'exec'),scope)
 cfg=yaml.safe_load((SOURCE/'options/test/GoPro/NAFNet-width32.yml').read_text())['network_g'];assert cfg.pop('type')=='NAFNetLocal' and cfg=={'width':32,'enc_blk_nums':[1,1,1,28],'middle_blk_num':1,'dec_blk_nums':[1,1,1,1]}
 # Upstream TLC conversion performs one CPU shape-initialization forward using
 # constructor weights; it is not evaluated as a restoration or speed result.
 model=scope['NAFNetLocal'](**cfg);state=torch.load(ASSET/'NAFNet-GoPro-width32.pth',map_location='cpu',weights_only=True);assert 'params' in state;model.load_state_dict(state['params'],strict=True);model.eval().requires_grad_(False)
 assert all(v.dtype==torch.float32 for v in model.parameters())
 return model,{**provenance,'parameters':sum(p.numel() for p in model.parameters()),'constructor_shape_forwards_cpu':1,'implementation':'Official model, Local_Base and LayerNorm class bodies; two import bindings isolated, no numerical changes; fast_imp=False'}
