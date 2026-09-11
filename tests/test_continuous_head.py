"""CPU sampling contracts and known-motion evidence, not H3 GPU tests."""
import unittest
import numpy as np
import torch
from h3ce.data import continuous_head as new
from h3ce.data import head_bucket as old
from h3ce.data.stable_head import stabilize_head_boxes


def texture(hw=(256,384), dx=0., dy=0., zoom=1.):
    y,x=torch.meshgrid(torch.arange(hw[0],dtype=torch.float32),torch.arange(hw[1],dtype=torch.float32),indexing='ij')
    u=(x-dx)/zoom;v=(y-dy)/zoom
    return torch.stack([.5+.2*torch.sin(u*.12),.5+.2*torch.cos(v*.09),.5+.15*torch.sin((u+v)*.08)])[None]


class ContinuousHeadTests(unittest.TestCase):
    def test_uniform_affine_roundtrip_and_fractional_area(self):
        for side in (256,512):
            g=new.head_transform([100.24,61.81,181.6,168.09],(256,384),frame_index=0,pts=0.,side=side)
            np.testing.assert_allclose(np.array(g['bucket_to_original'])@g['original_to_bucket'],np.eye(3),atol=1e-13)
            self.assertEqual(*g['scale_xy'])
            self.assertAlmostEqual(new.valid_bucket(g).double().sum().item(),np.prod(g['resized_hw']),delta=.005)
            self.assertTrue(any(x!=round(x) for x in g['pad_lrtb']))

    def test_full_canvas_identity(self):
        x=texture((256,256))
        g=new.head_transform([0,0,256,256],(256,256),frame_index=0,pts=0.,side=256,expansion_xy=(1,1))
        self.assertTrue(torch.equal(x,new.pack_frame(x,g)))
        self.assertTrue(torch.equal(x,new.inverse_delta(x,g)))

    def test_zero_delta_and_outside_preserved_exactly(self):
        x=texture();g=new.head_transform([100.3,61.2,182.6,166.8],(256,384),frame_index=0,pts=0.)
        delta=torch.zeros(1,3,256,256)
        self.assertTrue(torch.equal(x,new.paste_delta(x,delta,g)))
        out=new.paste_delta(x,delta+.125,g);a,b,c,d=g['paste_xyxy']
        mask=torch.zeros_like(x,dtype=torch.bool);mask[...,b:d,a:c]=new.crop_feather(g)>0
        self.assertTrue(torch.equal(out[~mask],x[~mask]));self.assertGreater((out-x).max().item(),.12)

    def test_paired_geometry_and_gradient(self):
        x=texture().requires_grad_();g=new.head_transform([90.1,61.3,170.4,168.2],(256,384),frame_index=0,pts=0.)
        xx,yy,m=new.pack_training_pair(x,x,g)
        self.assertTrue(torch.equal(xx,yy));(xx*m).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all());self.assertGreater(x.grad.abs().sum().item(),0)

    def test_tiny_box_change_has_no_integer_crop_jump(self):
        x=texture();outputs={}
        for name,module in [('old',old),('new',new)]:
            outputs[name]=[]
            for dx in (-.0001,.0001):
                g=module.head_transform([100+dx,60,180+dx,180],(256,384),frame_index=0,pts=0.,side=256)
                outputs[name].append(module.pack_frame(x,g))
        a=(outputs['new'][0]-outputs['new'][1]).abs().max().item()
        b=(outputs['old'][0]-outputs['old'][1]).abs().max().item()
        self.assertLess(a,1e-5);self.assertGreater(b,1e-3)

    def test_known_scene_translation_tracks_without_erasing_real_motion(self):
        outputs={key:[] for key in ('old','new','fixed')}
        for dx,dy in [(0.,0.),(.4,.3),(1.1,.8),(2.3,1.4)]:
            x=texture(dx=dx,dy=dy)
            for name,module in [('old',old),('new',new),('fixed',new)]:
                mx,my=(0.,0.) if name=='fixed' else (dx,dy)
                g=module.head_transform([100+mx,60+my,180+mx,180+my],(256,384),frame_index=0,pts=0.,side=256)
                outputs[name].append(module.pack_frame(x,g)[...,40:210,80:170])
        errors={k:torch.stack(v).diff(dim=0).square().mean().item() for k,v in outputs.items()}
        self.assertLess(errors['new'],errors['old']*.01)
        self.assertGreater(errors['fixed'],1e-5)
        self.assertGreater(outputs['new'][0].std().item(),.1)

    def test_antialias_suppresses_above_nyquist_pattern(self):
        pattern=(torch.arange(512)%2).float()[None,None,None,:].expand(1,3,512,512)
        g=new.head_transform([0,0,512,512],(512,512),frame_index=0,pts=0.,side=128,expansion_xy=(1,1))
        out=new.pack_frame(pattern,g)
        self.assertLess((out[...,4:-4,4:-4]-.5).abs().max().item(),1e-6)

    def test_known_zoom_uses_dynamic_rectangle_and_preserves_texture(self):
        # The generated field and face geometry share a known similarity map.
        crops=[]
        for scale in (1.,1.03,1.08):
            g=new.head_transform([100*scale,60*scale,180*scale,180*scale],(512,512),frame_index=0,pts=0.,side=256)
            self.assertNotEqual(g['source_hw'][0],g['source_hw'][1])
            x=texture((512,512),dx=(scale-1)/2,dy=(scale-1)/2,zoom=scale)
            crops.append(new.pack_frame(x,g)[...,50:200,85:165])
        self.assertLess(torch.stack(crops).diff(dim=0).square().mean().item(),1e-6)
        self.assertGreater(crops[-1].std().item(),.1)

    def test_face_margin_does_not_claim_automatic_hat_coverage(self):
        g=new.head_transform([120,140,180,220],(432,768),frame_index=0,pts=0.)
        # Face is unchanged: known high-hat extent is outside the fixed expansion.
        hypothetical_hat_top=60
        self.assertGreater(g['crop_xyxy'][1],hypothetical_hat_top)
        self.assertEqual(g['bucket_hw'],[256,256])

    def test_native_gap_and_provenance_rejected(self):
        boxes=[[100,60,180,180]]*3;pts=[0.,.02,.5]
        records=stabilize_head_boxes(boxes,pts,[0]*3,(256,384),side=256)
        with self.assertRaisesRegex(ValueError,'PTS gaps'):new.from_stable_records(records,pts,(256,384))
        pts=[0.,.02,.04];records=stabilize_head_boxes(boxes,pts,[0]*3,(256,384),side=256)
        self.assertEqual(len(new.from_stable_records(records,pts,(256,384))),3)
        records[0]['raw_bbox_provenance']='source_detector'
        with self.assertRaises(ValueError):new.from_stable_records(records,pts,(256,384))

    def test_invalid_precision_is_not_silently_cast(self):
        x=texture().half();g=new.head_transform([100,60,180,180],(256,384),frame_index=0,pts=0.)
        with self.assertRaises(ValueError):new.pack_frame(x,g)


if __name__=='__main__':unittest.main()
