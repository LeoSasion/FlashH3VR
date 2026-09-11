"""Equivalent YOLO11 execution reuse for the explicitly audited local runtime.

The first warmup forward already processes the exact first source tensor. Keep
that result for its immediately following inference, retaining upstream NMS
warmup. No results are shared between different tensors or predict calls.
"""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import hashlib

import numpy as np
import torch

from .detect_yolo11 import Yolo11Detectors, _boxes_from_result


def _check_runtime():
    from ultralytics.engine import predictor
    from ultralytics.nn import autobackend
    expected = [(predictor, 'c3726b297cee2e489634f6ee82e3bc9c824a92c356942a6bf1a86ba939b8923d'),
                (autobackend, 'b4423c84f90e6f7430dc3c71248dd98a1b9e3e074a685006ab446c9d7032dfd2')]
    for module, digest in expected:
        with Path(module.__file__).open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        if actual != digest:
            raise RuntimeError('Warmup reuse requires the reviewed predictor/backend source; no fallback')


@contextmanager
def reuse_actual_warmup(model):
    """Scope the patch to one owned predictor invocation and restore on errors."""
    _check_runtime()
    events, restorers = [], []
    pending = {}

    def on_start(predictor):
        if predictor.done_warmup:
            return
        backend = predictor.model
        if (backend.format != 'pt' or backend.fp16 or backend.device.type != 'cuda'
                or backend.backend.model.training or predictor.args.augment or predictor.args.embed):
            raise RuntimeError('Reuse is only audited for eval CUDA FP32 PyTorch detection without augmentation')
        original_forward, original_warmup = backend.forward, backend.warmup
        had_forward, had_warmup = 'forward' in backend.__dict__, 'warmup' in backend.__dict__
        active = False

        def forward(im, augment=False, embed=None, **kwargs):
            if augment or embed is not None or kwargs:
                raise RuntimeError('Unexpected inference arguments in the warmup reuse window')
            if active:
                if pending:
                    raise RuntimeError('Unexpected repeated model forward during warmup')
                snapshot = im.detach().clone()
                result = original_forward(im)
                pending.update(input=im, snapshot=snapshot, result=result)
                events.append(dict(event='actual_first_batch_forward', shape=list(im.shape),
                                   dtype=str(im.dtype), tensor_data_ptr=im.data_ptr(),
                                   guarded_input_bytes=im.numel() * im.element_size()))
                return result
            if pending:
                if im is not pending['input'] or not torch.equal(im, pending['snapshot']):
                    raise RuntimeError('Warmup result cannot be reused for a different or modified tensor')
                result = pending.pop('result')
                pending.clear()
                events.append(dict(event='consumed_for_identical_source_tensor', shape=list(im.shape),
                                   dtype=str(im.dtype), tensor_data_ptr=im.data_ptr()))
                return result
            return original_forward(im, augment=augment, embed=embed, **kwargs)

        def warmup(*args, **kwargs):
            nonlocal active
            if args or set(kwargs) != {'im'} or kwargs['im'] is None or pending:
                raise RuntimeError('Expected upstream warmup(im=actual_source_tensor)')
            active = True
            try:
                original_warmup(**kwargs)  # Includes the unchanged upstream NMS warmup.
            finally:
                active = False
            if not pending:
                raise RuntimeError('Upstream warmup did not perform the expected source forward')

        def restore():
            if had_forward:
                backend.forward = original_forward
            else:
                del backend.forward
            if had_warmup:
                backend.warmup = original_warmup
            else:
                del backend.warmup

        restorers.append(restore)
        backend.forward, backend.warmup = forward, warmup

    callbacks = model.callbacks['on_predict_start']
    callbacks.append(on_start)
    try:
        yield events
        if pending:
            raise RuntimeError('Unconsumed warmup result at end of predict call')
    finally:
        callbacks.remove(on_start)
        for restore in reversed(restorers):
            restore()
        pending.clear()


def batch_boxes_to_cpu(results, *, class_id, height, width):
    """One batch D2H copy, preserving the original bbox contract validation."""
    rows, sizes = [], []
    for result in results:
        if tuple(result.orig_shape) != (height, width):
            raise ValueError('Original result dimensions changed')
        data = result.boxes.data
        if data.ndim != 2 or data.shape[1] != 6:
            raise ValueError('Only untracked xyxy/conf/class detection rows are supported')
        rows.append(data)
        sizes.append(len(data))
    if not rows:
        return []
    data = torch.cat(rows, dim=0).detach().cpu().numpy()
    output, start = [], 0
    for size in sizes:
        part = data[start:start + size]
        boxes = SimpleNamespace(xyxy=part[:, :4], conf=part[:, 4], cls=part[:, 5])
        result = SimpleNamespace(orig_shape=(height, width), boxes=boxes)
        output.append(_boxes_from_result(result, class_id=class_id, height=height, width=width, role='face'))
        start += size
    return output


def detect_face_batch_reuse(detector, frames):
    if not isinstance(detector, Yolo11Detectors):
        raise TypeError('Verified YOLO11 adapter required')
    if not frames or len(frames) > 16:
        raise ValueError('Explicit batch length must be 1..16')
    if len({tuple(frame.shape) for frame in frames}) != 1:
        raise ValueError('This batch requires matching canvas dimensions')
    if any(f.ndim != 3 or f.shape[2] != 3 or f.dtype != np.uint8 for f in frames):
        raise ValueError('Expected RGB uint8 HWC')
    h, w, _ = frames[0].shape
    settings = detector.settings['face']
    with reuse_actual_warmup(detector.models['face']) as events:
        results = detector.models['face'].predict(
            source=[np.ascontiguousarray(f[:, :, ::-1]) for f in frames],
            imgsz=settings['imgsz'], conf=settings['confidence'], iou=settings['nms_iou'],
            classes=[detector.class_ids['face']], stream=False, save=False, verbose=False,
            augment=False, half=False, device=detector.device)
    if len(results) != len(frames):
        raise ValueError('Batch source/result count changed')
    if not hasattr(detector, 'h3ce_reuse_events'):
        detector.h3ce_reuse_events = []
    detector.h3ce_reuse_events.extend(events)
    return batch_boxes_to_cpu(results, class_id=detector.class_ids['face'], height=h, width=w)
