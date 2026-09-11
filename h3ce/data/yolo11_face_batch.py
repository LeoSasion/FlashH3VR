"""Batched input-only face detection with an already verified YOLO11 adapter."""
import numpy as np
from .detect_yolo11 import Yolo11Detectors,_boxes_from_result


def detect_face_batch(detector,frames):
    if not isinstance(detector,Yolo11Detectors):raise TypeError('Verified YOLO11 adapter required')
    if not frames or len(frames)>16:raise ValueError('Explicit batch length must be 1..16')
    shapes={tuple(frame.shape) for frame in frames}
    if len(shapes)!=1:raise ValueError('This batch requires matching canvas dimensions')
    if any(f.ndim!=3 or f.shape[2]!=3 or f.dtype!=np.uint8 for f in frames):raise ValueError('Expected RGB uint8 HWC')
    h,w,_=frames[0].shape;settings=detector.settings['face']
    # Passing a same-sized list batches these images; do not resize/crop the face
    # before detection. Explicit full precision avoids hidden half-mode changes.
    results=detector.models['face'].predict(source=[np.ascontiguousarray(f[:,:,::-1]) for f in frames],
        imgsz=settings['imgsz'],conf=settings['confidence'],iou=settings['nms_iou'],classes=[detector.class_ids['face']],
        stream=False,save=False,verbose=False,augment=False,half=False,device=detector.device)
    if len(results)!=len(frames):raise ValueError('Batch source/result count changed')
    return [_boxes_from_result(r,class_id=detector.class_ids['face'],height=h,width=w,role='face') for r in results]
