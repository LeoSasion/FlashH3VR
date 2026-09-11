"""Known-motion CPU tests distinguish detection perturbations from real motion."""
import unittest
import numpy as np
from h3ce.data.stable_head import stabilize_head_boxes,validate_stable_records


def boxes_from(parameters):
    return [[cx-w/2,cy-h/2,cx+w/2,cy+h/2] for cx,cy,w,h in parameters]


def centers(records):return np.array([[(r['used_face_xyxy'][0]+r['used_face_xyxy'][2])/2,(r['used_face_xyxy'][1]+r['used_face_xyxy'][3])/2] for r in records])


class StableHeadTests(unittest.TestCase):
    def test_preserves_velocity_and_log_scale_on_irregular_pts(self):
        t=np.arange(21)/60+np.arange(21)**2*.00001
        b=boxes_from([(120+80*x,140+20*x,60*np.exp(x),80*np.exp(x*.4)) for x in t])
        rs=stabilize_head_boxes(b,t,[0]*len(t),(432,768))
        np.testing.assert_allclose([r['used_face_xyxy'] for r in rs],b,atol=1e-10)
    def test_reduces_known_measurement_jitter_without_frozen_box(self):
        t=np.arange(31)/60;true=np.array([(120+60*x,140+12*x,60*np.exp(x*.3),80*np.exp(x*.3)) for x in t])
        noisy=true.copy();noisy[:,0]+=(-1.)**np.arange(31)*.8;noisy[:,1]+=(-1.)**np.arange(31)*.5
        rs=stabilize_head_boxes(boxes_from(noisy),t,[0]*31,(432,768))
        self.assertLess(np.mean((centers(rs)-true[:,:2])**2),.3*np.mean((noisy[:,:2]-true[:,:2])**2))
        self.assertGreater(np.ptp(centers(rs)[:,0]),20)
    def test_shots_missing_frames_and_gap_do_not_bridge(self):
        a=[90,90,150,170];b=[390,190,450,270]
        boxes=[a,a,None,b,b,a,a];pts=[0.,.02,.04,.06,.08,.5,.52];shots=[0,0,0,1,1,2,2]
        rs=stabilize_head_boxes(boxes,pts,shots,(432,768))
        self.assertIsNone(rs[2])
        self.assertEqual(rs[1]['contributors'],[0,1]);self.assertEqual(rs[3]['contributors'],[3,4]);self.assertEqual(rs[5]['contributors'],[5,6])
    def test_validation_and_coverage(self):
        b=[[0,0,50,80],[1,1,51,81],[2,1,52,81]];t=[0.,.02,.04]
        rs=stabilize_head_boxes(b,t,[0]*3,(432,768));validate_stable_records(rs,t,(432,768),side=256)
        for box,r in zip(b,rs):
            a,c,d,e=r['transform']['crop_xyxy'];self.assertLessEqual(a,box[0]);self.assertLessEqual(c,box[1]);self.assertGreaterEqual(d,box[2]);self.assertGreaterEqual(e,box[3])
        rs[1]['raw_bbox_provenance']='source_detector'
        with self.assertRaises(ValueError):validate_stable_records(rs,t,(432,768),side=256)


if __name__=='__main__':unittest.main()
