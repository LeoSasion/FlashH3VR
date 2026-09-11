"""CPU geometry/dispatch tests using a stand-in codec, not H3 inference."""
import unittest
from types import SimpleNamespace
import torch
from h3ce.data.head_bucket import head_transform
from h3ce.data.stable_head import stabilize_head_boxes
from h3ce.infer.head_region_native import restore_region
from h3ce.vae.bridge import FrameMeta


class Codec:
    def __init__(self):self.backend=SimpleNamespace(model=torch.nn.Identity());self.inputs=[]
    def current_codec_pack(self):return None
    def encode_rgb(self,x,meta):self.inputs.append(x.clone());return SimpleNamespace(tensor=x)
    def decode_latent(self,z,**kw):return z.tensor


class Zero(torch.nn.Module):
    def forward(self,x):return torch.zeros_like(x)


class RegionNativeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);self.x=torch.linspace(0,1,3*3*64*96).reshape(3,3,64,96)
        self.pts=[0.,.02,.04];self.meta=FrameMeta('video',tuple(self.pts),real_video=True)
        self.boxes=[[20+i,10,40+i*2,40] for i in range(3)]
        self.raw=[head_transform(b,(64,96),frame_index=i,pts=self.pts[i],side=256) for i,b in enumerate(self.boxes)]
        self.stable=stabilize_head_boxes(self.boxes,self.pts,[0]*3,(64,96))
    def test_four_routes_zero_correction_preserves_original_and_time_axis(self):
        for scope in ('full','head'):
            for stable in (False,True):
                with self.subTest(scope=scope,stable=stable):
                    codec=Codec();r=restore_region(self.x,self.meta,self.stable if stable else self.raw,codec,Zero(),stable=stable,native_scope=scope)
                    self.assertTrue(torch.equal(self.x,r['prediction']));self.assertEqual(r['pts'],self.meta.pts)
                    self.assertEqual(tuple(codec.inputs[0].shape),(1,3,3,64,96) if scope=='full' else (1,3,3,256,256))
    def test_full_h3_input_does_not_depend_on_smoothed_geometry(self):
        c=Codec()
        for stable in (False,True):restore_region(self.x,self.meta,self.stable if stable else self.raw,c,Zero(),stable=stable,native_scope='full')
        self.assertTrue(torch.equal(c.inputs[0],c.inputs[1]))
    def test_missing_or_cross_shot_tracks_rejected_before_h3(self):
        for records in ([None,*self.stable[1:]],stabilize_head_boxes(self.boxes,self.pts,[0,1,1],(64,96))):
            c=Codec()
            with self.assertRaises(ValueError):restore_region(self.x,self.meta,records,c,Zero(),stable=True,native_scope='head')
            self.assertEqual(c.inputs,[])


if __name__=='__main__':unittest.main()
