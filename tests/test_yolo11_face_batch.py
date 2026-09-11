"""CPU contract stand-ins verify batch ownership/order; not detector quality tests."""
from types import SimpleNamespace
import unittest
import numpy as np
from h3ce.data.detect_yolo11 import Yolo11Detectors
from h3ce.data.yolo11_face_batch import detect_face_batch


class BatchFaceTests(unittest.TestCase):
    def adapter(self,predict):
        adapter=object.__new__(Yolo11Detectors)
        adapter.settings={'face':{'imgsz':960,'confidence':.25,'nms_iou':.5}}
        adapter.models={'face':SimpleNamespace(predict=predict)};adapter.class_ids={'face':0};adapter.device='cpu'
        return adapter

    def test_input_color_order_and_result_frame_order(self):
        frames=[np.full((40,60,3),[i+1,10,20],np.uint8) for i in range(3)]
        def predict(**kw):
            self.assertFalse(kw['half']);self.assertFalse(kw['augment']);self.assertEqual(kw['classes'],[0])
            out=[]
            for i,f in enumerate(kw['source']):
                np.testing.assert_array_equal(f[0,0],[20,10,i+1])
                out.append(SimpleNamespace(orig_shape=(40,60),boxes=SimpleNamespace(
                    xyxy=np.array([[i+1,2,30,35]],np.float32),conf=np.array([.9],np.float32),cls=np.array([0],np.float32))))
            return out
        result=detect_face_batch(self.adapter(predict),frames)
        self.assertEqual([r[0][0] for r in result],[1,2,3])

    def test_mixed_canvas_and_overlarge_batch_rejected_before_forward(self):
        def never(**kw):self.fail('Invalid batch reached model')
        d=self.adapter(never);x=np.zeros((40,60,3),np.uint8)
        for frames in ([x]*17,[x,np.zeros((41,60,3),np.uint8)],[],[x.astype(np.float32)]):
            with self.assertRaises(ValueError):detect_face_batch(d,frames)

    def test_result_count_cannot_drop_frames(self):
        d=self.adapter(lambda **kw:[])
        with self.assertRaises(ValueError):detect_face_batch(d,[np.zeros((40,60,3),np.uint8)])

    def test_only_verified_adapter_type(self):
        with self.assertRaises(TypeError):detect_face_batch(SimpleNamespace(),[np.zeros((40,60,3),np.uint8)])


if __name__=='__main__':unittest.main()
