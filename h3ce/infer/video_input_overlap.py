"""Complete input validation and shot analysis overlapped with owned YOLO work."""
from pathlib import Path
import time

import av
import numpy as np

from h3ce.data.video import iter_video_frames, detect_shots
from h3ce.data.yolo11_face_executor import Yolo11BatchExecutor
from h3ce.infer.head_region_native import spatial_precision


def read_detect_video_overlap(source, detector, *, max_frames, long_edge=768,
                              reuse_execution=True, optimize_input=True):
    """Return only after EOF validation, all shot decisions and GPU jobs finish.

    First-use detector setup stays inside this call's timed file-open interval.
    H3 may start only after the caller receives the complete returned structure.
    """
    if not isinstance(detector, Yolo11BatchExecutor):
        raise ValueError('Input overlap requires the owned YOLO executor')
    if type(max_frames) is not int or max_frames < 2:
        raise ValueError('Explicit finite frame limit required')
    source = Path(source)
    started = time.perf_counter()
    with av.open(str(source)) as container:
        if len(container.streams.video) != 1:
            raise ValueError('Exactly one input video stream required')
        if len(container.streams.audio):
            raise ValueError('Research file path has no audio remux yet; refusing to drop audio')
        rate = container.streams.video[0].average_rate
        if not rate:
            raise ValueError('Missing source nominal frame rate')
    decoded, prepared = [], {}
    reader = iter_video_frames(source, working_long_edge_max=long_edge, max_frames=max_frames,
                               reuse_reformatter=reuse_execution, reuse_buffers=optimize_input)

    def frames():
        try:
            for frame in reader:
                decoded.append(frame)
                yield frame.rgb
        finally:
            reader.close()

    def finish_input():
        if len(decoded) < 2:
            raise ValueError('Real continuous input video required')
        arrays = np.stack([frame.rgb for frame in decoded])
        if arrays.shape[1] % 2 or arrays.shape[2] % 2:
            raise ValueError('This H264 writer requires even working dimensions')
        shots, shot_report = detect_shots(list(arrays), packed_channels=optimize_input)
        prepared.update(arrays=arrays, pts=[f.pts for f in decoded],
                        integer_pts=[f.pts_integer for f in decoded],
                        time_bases=[list(f.time_base) for f in decoded],
                        color=decoded[0].color_contract, shots=shots, shot_report=shot_report, rate=rate)

    source_frames = frames()
    try:
        # These flags are process-wide. Hold them until every owned worker has
        # drained, including when a later frame or the CPU callback fails.
        with spatial_precision():
            faces = detector.detect_stream(source_frames, on_input_complete=finish_input)
    finally:
        source_frames.close()
        reader.close()
    if len(faces) != len(prepared['arrays']):
        raise ValueError('Detector must return every source frame in order')
    prepared.update(faces=faces, overlap_seconds=time.perf_counter()-started)
    return prepared
