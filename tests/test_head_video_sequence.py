"""CPU planning and stand-in overlap contracts; no real H3 execution."""
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from h3ce.infer.head_video_sequence import prepare_sequence,native_sequence


def records(n):
    return [{'frame_index':i,'pts':i/60,'shot_id':0,'bbox_provenance':'input_detector',
        'face_xyxy':[100+i*.2,60,180+i*.2,180],'decision':'unique_face'} for i in range(n)]


class HeadSequenceTests(unittest.TestCase):
    def test_global_geometry_shared_by_overlap(self):
        rs=records(180);p=prepare_sequence(rs,[r['pts'] for r in rs],(432,768))
        self.assertEqual(len(p['chunks']),11);self.assertEqual(p['used_frames'],list(range(180)))
        self.assertEqual(p['chunks'][-1]['stop'],180)
        usage={i:[] for i in range(180)}
        for c in p['chunks']:
            for i in range(c['start'],c['stop']):usage[i].append(p['transforms'][i])
        self.assertTrue(all(v[0] is g for v in usage.values() for g in v))
        self.assertEqual(p['stable_records'][17]['contributors'],[15,16,17,18,19])
        self.assertNotEqual(p['transforms'][0]['crop_xyxy'],p['transforms'][-1]['crop_xyxy'])

    def test_shots_gaps_missing_and_isolated_do_not_share_context(self):
        rs=records(9)
        for i in (2,7):rs[i].update(face_xyxy=None,decision='missing')
        for i in range(5,9):rs[i]['pts']+=.5
        rs[4]['shot_id']=1
        p=prepare_sequence(rs,[r['pts'] for r in rs],(432,768))
        self.assertEqual([(c['start'],c['stop']) for c in p['chunks']],[(0,2),(5,7)])
        self.assertEqual(p['used_frames'],[0,1,5,6])
        self.assertEqual(p['skipped_frames'][4],'isolated_frame_no_video_context')

    def test_source_provenance_pts_and_eligibility_fail_closed(self):
        for field,value in [('bbox_provenance','source_detector'),('frame_index',8),('decision','ambiguous')]:
            rs=records(3);rs[1][field]=value
            with self.assertRaises(ValueError):prepare_sequence(rs,[r['pts'] for r in rs],(432,768))
        rs=records(3)
        with self.assertRaises(ValueError):prepare_sequence(rs,[0.,.1,.2],(432,768))

    def test_matching_source_frames_blend_different_contexts(self):
        rs=records(23);pts=[r['pts'] for r in rs];plan=prepare_sequence(rs,pts,(432,768))
        class StandIn:
            def __init__(self):self.inputs=[]
            def current_codec_pack(self):return 'frozen'
            def encode_rgb(self,x,meta):
                self.inputs.append((x.clone(),meta.pts))
                return SimpleNamespace(tensor=x+len(self.inputs)*.01,valid_frames=x.shape[2],padded_frames=22)
            def decode_latent(self,z,*,grad,codec_pack):
                assert not grad and codec_pack=='frozen'
                return z.tensor
        x=torch.arange(23,dtype=torch.float32)[:,None,None,None].expand(23,3,32,32)/100
        bridge=StandIn();out,w,context=native_sequence(x,pts,plan['chunks'],bridge)
        self.assertEqual(len(bridge.inputs),2)
        self.assertEqual(bridge.inputs[1][1],tuple(pts[17:23]))
        self.assertTrue(torch.equal(bridge.inputs[0][0][:,:,17:22],bridge.inputs[1][0][:,:,:5]))
        np.testing.assert_allclose(w.numpy(),1,atol=1e-7)
        for j,i in enumerate(range(17,22)):
            expected=x[i]+.01*((5-j)/6)+.02*((j+1)/6)
            torch.testing.assert_close(out[i],expected,atol=1e-7,rtol=0)
        self.assertEqual(context[-1]['valid_frames'],6)


if __name__=='__main__':unittest.main()
