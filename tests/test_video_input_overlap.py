"""CPU ownership, bounded scheduling and real-media checks; no neural inference."""
from concurrent.futures import Future
from fractions import Fraction
import threading
from types import SimpleNamespace

import av
import numpy as np
import pytest
import torch

from h3ce.data import yolo11_face_executor as execution
from h3ce.data.video import iter_video_frames, detect_shots
from h3ce.infer import video_input_overlap as overlap
from h3ce.infer.head_region_native import spatial_precision


def immediate_workers(count, calls):
    def submit(job, frames, origin):
        calls.append((job, frames))
        result = Future()
        result.set_result(dict(job=job, faces=[float(frame[0, 0, 0]) for frame in frames]))
        return result
    return [SimpleNamespace(submit=submit) for _ in range(count)]


@pytest.mark.parametrize('profile,count', [('single16', 1), ('single32', 1), ('dual16', 2)])
def test_stream_submits_early_preserves_frame_identity_and_partial_tail(profile, count):
    arrays = [np.full((2, 2, 3), index, np.float32) for index in range(65)]
    calls, consumed, callbacks = [], [], []
    batch_size = execution.PROFILES[profile][0]
    def source():
        for index, frame in enumerate(arrays):
            if index == batch_size:
                assert calls[0][0].stop == batch_size
            consumed.append(index)
            yield frame
    rows, trace = execution.collect_stream_jobs(source(), immediate_workers(count, calls), profile, None,
        on_input_complete=lambda: callbacks.append(list(consumed)))
    assert [value for row in rows for value in row['faces']] == list(range(65))
    assert [job for job, _ in calls] == execution.plan_face_jobs(65, profile)
    assert all(frame is arrays[job.start+i] for job, frames in calls for i, frame in enumerate(frames))
    assert callbacks == [list(range(65))]
    assert trace['submissions'][0]['source_frames_read'] == batch_size


def test_stream_out_of_order_owners_and_callback_overlap():
    release_first, second_done = threading.Event(), threading.Event()
    completions, callback = [], []
    class Resource:
        def __init__(self):
            self.owner = threading.get_ident()
        def run(self, job, frames, origin):
            assert threading.get_ident() == self.owner
            if job.index == 0:
                assert release_first.wait(5)
            completions.append(job.index)
            if job.index == 1:
                second_done.set()
            return dict(job=job, faces=[float(frame[0, 0, 0]) for frame in frames])
    workers = [execution._OwnedThread(Resource, i) for i in range(2)]
    def finish():
        assert second_done.wait(5)
        callback.append(list(completions))
        release_first.set()
    try:
        rows, trace = execution.collect_stream_jobs(
            (np.full((1, 1, 3), i, np.float32) for i in range(60)), workers, 'dual16', None,
            on_input_complete=finish)
        assert 0 not in callback[0] and completions[0] == 1
        assert [x for row in rows for x in row['faces']] == list(range(60))
        assert trace['peak_pending_batches'] <= 4 and max(trace['peak_pending_by_owner']) <= 2
        assert [r['job'].worker for r in rows] == [0, 1, 0, 1]
    finally:
        release_first.set()
        for worker in workers:
            worker.close()


def test_long_source_enforces_per_owner_queue_bound_and_drains():
    waiting, release = threading.Event(), threading.Event()
    consumed, callbacks = [], []
    class Resource:
        def run(self, job, frames, origin):
            if job.index == 0:
                waiting.set()
                assert release.wait(5)
            return dict(job=job, faces=[float(frame[0, 0, 0]) for frame in frames])
    workers = [execution._OwnedThread(Resource, 0)]
    def source():
        for i in range(97):
            consumed.append(i)
            if i == 47:
                assert waiting.wait(5)
                # One running, one queued, one caller buffer: the third
                # submission cannot pass until an owner slot is released.
                release.set()
            yield np.full((1, 1, 3), i, np.float32)
    try:
        rows, trace = execution.collect_stream_jobs(source(), workers, 'single16', None,
            on_input_complete=lambda: callbacks.append(len(consumed)))
        assert [x for row in rows for x in row['faces']] == list(range(97))
        assert trace['peak_pending_by_owner'] == [2] and trace['peak_pending_batches'] == 2
        assert callbacks == [97]
    finally:
        release.set()
        workers[0].close()


@pytest.mark.parametrize('failure', ['reader', 'callback', 'worker'])
def test_failure_closes_source_drains_running_owner_and_restores_precision(failure):
    release, started, finished = threading.Event(), threading.Event(), threading.Event()
    calls, source_closed = [], []
    class Resource:
        def run(self, job, frames, origin):
            calls.append(job.index)
            assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
            started.set()
            assert release.wait(5)
            assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
            finished.set()
            if failure == 'worker':
                raise RuntimeError('worker failure')
            return dict(job=job, faces=[0.] * len(frames))
    worker = execution._OwnedThread(Resource, 0)
    def source():
        try:
            for i in range(16):
                yield np.zeros((1, 1, 3), np.float32)
            assert started.wait(5)
            if failure == 'reader':
                release.set()
                raise RuntimeError('reader failure')
        finally:
            source_closed.append(True)
    def callback():
        release.set()
        if failure == 'callback':
            raise RuntimeError('callback failure')
    prior = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    try:
        with pytest.raises(RuntimeError, match=f'{failure} failure'), spatial_precision():
            execution.collect_stream_jobs(source(), [worker], 'single16', None, on_input_complete=callback)
        assert finished.is_set() and calls == [0] and source_closed == [True]
        assert (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32) == prior
    finally:
        release.set()
        worker.close()


@pytest.mark.parametrize('bad', ['empty', 'dtype', 'shape'])
def test_stream_invalid_input_fails_without_later_dispatch(bad):
    calls = []
    frames = [] if bad == 'empty' else [np.zeros((1, 1, 3), np.float32)]
    if bad == 'dtype':
        frames.append(np.zeros((1, 1, 3), np.float64))
    if bad == 'shape':
        frames.append(np.zeros((2, 1, 3), np.float32))
    with pytest.raises(ValueError):
        execution.collect_stream_jobs(iter(frames), immediate_workers(2, calls), 'dual16', None)
    assert calls == []


def test_immediately_failed_worker_stops_consumption_and_rejects_wrong_identity():
    consumed, closed = [], []
    def source():
        try:
            for i in range(60):
                consumed.append(i)
                yield np.zeros((1, 1, 3), np.float32)
        finally:
            closed.append(True)
    def submit(job, frames, origin):
        future = Future()
        future.set_result(dict(job=job, faces=[]))
        return future
    with pytest.raises(ValueError, match='identity or result count'):
        execution.collect_stream_jobs(source(), [SimpleNamespace(submit=submit)] * 2, 'dual16', None)
    assert consumed == list(range(16)) and closed == [True]


def cpu_executor():
    detector = object.__new__(execution.Yolo11BatchExecutor)
    detector.submitted = []
    def detect(frames, *, on_input_complete):
        assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
        rows, detector.trace = execution.collect_stream_jobs(frames, immediate_workers(2, detector.submitted),
            'dual16', None, on_input_complete=on_input_complete)
        return [value for row in rows for value in row['faces']]
    detector.detect_stream = detect
    return detector


@pytest.mark.parametrize('long_edge', [96, 48])
def test_real_media_overlap_retains_nonuniform_pts_all_pixels_and_shots(tmp_path, long_edge):
    source = tmp_path/'vfr.nut'
    timestamps = [i*2 + (i//7) for i in range(35)]
    with av.open(str(source), 'w') as container:
        stream = container.add_stream('ffv1', rate=60)
        stream.width, stream.height, stream.pix_fmt = 96, 64, 'bgr0'
        stream.time_base = stream.codec_context.time_base = Fraction(1, 120)
        for field, value in [('colorspace', 1), ('color_primaries', 1), ('color_trc', 13), ('color_range', 2)]:
            setattr(stream.codec_context, field, value)
        for index, pts in enumerate(timestamps):
            rgb = np.full((64, 96, 3), 20 if index < 17 else 230, np.uint8)
            rgb[:, :10, 0] = index
            frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            frame.pts, frame.time_base = pts, Fraction(1, 120)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    expected = list(iter_video_frames(source, max_frames=35, working_long_edge_max=long_edge,
                                     reuse_buffers=True, reuse_reformatter=True))
    arrays = np.stack([frame.rgb for frame in expected])
    detector = cpu_executor()
    result = overlap.read_detect_video_overlap(source, detector, max_frames=35, long_edge=long_edge)
    assert result['arrays'].tobytes() == arrays.tobytes()
    assert result['pts'] == [float(Fraction(pts, 120)) for pts in timestamps]
    assert result['integer_pts'] == [f.pts_integer for f in expected]
    assert result['time_bases'] == [list(f.time_base) for f in expected]
    assert result['color'] == expected[0].color_contract
    shots, report = detect_shots(list(arrays), packed_channels=True)
    assert shots[-1] == 1 and (result['shots'], result['shot_report']) == (shots, report)
    assert [job.stop-job.start for job, _ in detector.submitted] == [16, 16, 3]
    with pytest.raises(Exception, match='limit|maximum|frame'):
        overlap.read_detect_video_overlap(source, cpu_executor(), max_frames=34, long_edge=long_edge)
    assert not torch.cuda.is_initialized()


def test_closed_stream_executor_rejects_before_cuda_and_consumption():
    detector = object.__new__(execution.Yolo11BatchExecutor)
    detector._closed, detector._call_lock = True, threading.Lock()
    with pytest.raises(RuntimeError, match='closed'):
        detector.detect_stream(iter(()))


def test_queued_batches_cancel_before_running_owners_are_released():
    release = threading.Event()
    started = [threading.Event(), threading.Event()]
    cancelled, ran = [], []
    class Resource:
        def run(self, job, frames, origin):
            ran.append(job.index)
            assert job.index < 2
            started[job.worker].set()
            assert release.wait(5)
            return dict(job=job, faces=[0.] * len(frames))
    class Worker(execution._OwnedThread):
        def submit(self, job, frames, origin):
            future = super().submit(job, frames, origin)
            def completed(value):
                if value.cancelled():
                    cancelled.append(job.index)
                    if len(cancelled) == 2:
                        release.set()
            future.add_done_callback(completed)
            return future
    workers = [Worker(Resource, i) for i in range(2)]
    def source():
        for _ in range(64):
            yield np.zeros((1, 1, 3), np.float32)
        assert all(event.wait(5) for event in started)
        raise RuntimeError('late source failure')
    try:
        with pytest.raises(RuntimeError, match='late source failure'):
            execution.collect_stream_jobs(source(), workers, 'dual16', None)
        assert sorted(cancelled) == [2, 3] and sorted(ran) == [0, 1]
    finally:
        release.set()
        for worker in workers:
            worker.close()


def test_file_entry_does_not_start_h3_or_create_output_after_input_failure(tmp_path, monkeypatch):
    from h3ce.infer import head_video_file
    destination = tmp_path/'uncommitted.mp4'
    calls = []
    def fail(*args, **kwargs):
        raise ValueError('late input validation failure')
    monkeypatch.setattr(head_video_file, 'read_detect_video_overlap', fail)
    monkeypatch.setattr(head_video_file, 'restore_head_sequence', lambda *a, **k: calls.append('h3'))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    with pytest.raises(ValueError, match='late input validation failure'):
        head_video_file.restore_video_file(tmp_path/'source.avi', destination, None, None,
            SimpleNamespace(device='cuda:0'), max_frames=60, overlap_input=True)
    assert calls == [] and not destination.exists()
