"""CPU scheduler/contract checks; these do not validate GPU output or speed."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from h3ce.data import yolo11_face_executor as execution


@pytest.mark.parametrize('profile,sizes,owners', [
    ('single16', [16, 16, 16, 12], [0, 0, 0, 0]),
    ('single32', [32, 28], [0, 0]),
    ('dual16', [16, 16, 16, 12], [0, 1, 0, 1]),
])
def test_exact_coverage_and_partial_last_batch(profile, sizes, owners):
    jobs = execution.plan_face_jobs(60, profile)
    assert [j.stop - j.start for j in jobs] == sizes
    assert [j.worker for j in jobs] == owners
    assert [i for j in jobs for i in range(j.start, j.stop)] == list(range(60))
    assert execution.plan_face_jobs(1, profile)[0].stop == 1


@pytest.mark.parametrize('count,profile', [(0, 'single16'), (True, 'single16'), (1, 'unknown')])
def test_invalid_plan_rejected(count, profile):
    with pytest.raises(ValueError):
        execution.plan_face_jobs(count, profile)


def test_out_of_order_completion_preserves_frame_order_and_ownership():
    release_first = threading.Event()
    completion = []

    class Resource:
        def __init__(self, index):
            self.owner, self.index = threading.get_ident(), index

        def run(self, job, frames, origin):
            assert threading.get_ident() == self.owner and job.worker == self.index
            if job.index == 0:
                assert release_first.wait(5)
            completion.append(job.index)
            if job.index == 1:
                release_first.set()
            return dict(job=job, faces=list(frames))

    workers = [execution._OwnedThread(lambda i=i: Resource(i), i) for i in range(2)]
    try:
        rows = execution.collect_ordered_jobs(execution.plan_face_jobs(60, 'dual16'), workers, np.arange(60), None)
        assert completion[0] == 1
        assert [value for row in rows for value in row['faces']] == list(range(60))
        assert len({w.resource.owner for w in workers}) == 2
    finally:
        for worker in workers:
            worker.close()


def test_failure_propagates_after_other_owner_drains():
    both_started = threading.Barrier(2)
    drained, calls = threading.Event(), []

    class Resource:
        def run(self, job, frames, origin):
            calls.append(job.index)
            both_started.wait(timeout=5)
            if job.worker == 0:
                raise RuntimeError('source batch failed')
            drained.set()
            return dict(job=job, faces=list(frames))

    workers = [execution._OwnedThread(Resource, i) for i in range(2)]
    try:
        with pytest.raises(RuntimeError, match='source batch failed'):
            execution.collect_ordered_jobs(execution.plan_face_jobs(32, 'dual16'), workers, np.arange(32), None)
        assert drained.is_set() and sorted(calls) == [0, 1]
    finally:
        for worker in workers:
            worker.close()


def test_wrong_result_identity_fails_before_next_batch():
    from concurrent.futures import Future
    calls = []

    def submit(job, frames, origin):
        calls.append(job.index)
        future = Future()
        future.set_result(dict(job=job, faces=[]))
        return future

    with pytest.raises(ValueError, match='identity or result count'):
        execution.collect_ordered_jobs(execution.plan_face_jobs(60, 'single16'),
            [SimpleNamespace(submit=submit)], np.arange(60), None)
    assert calls == [0]


def test_closed_executor_rejects_new_input_without_cuda():
    executor = object.__new__(execution.Yolo11BatchExecutor)
    executor.workers, executor._closed, executor._call_lock = [], False, threading.Lock()
    executor.close()
    executor.close()
    with pytest.raises(RuntimeError, match='closed'):
        executor.detect_frames(np.zeros((1, 2, 2, 3), np.float32))


def test_predictor_options_match_official_model_predict(monkeypatch):
    monkeypatch.setenv('YOLO_OFFLINE', 'true')
    monkeypatch.setenv('YOLO_AUTOINSTALL', 'false')
    from ultralytics.engine.model import Model

    class Predictor:
        def __init__(self, *, overrides, _callbacks):
            self.received = overrides

        def setup_model(self, **kwargs):
            pass

        def __call__(self, **kwargs):
            return self.received

    model = SimpleNamespace(predictor=None, overrides={'task': 'detect', 'imgsz': 640},
                            callbacks={}, model=object(), _smart_load=lambda key: Predictor)
    expected = Model.predict(model, source=[np.zeros((2, 2, 3), np.uint8)],
        imgsz=960, conf=.25, iou=.5, classes=[0], stream=False, save=False,
        verbose=False, augment=False, half=False, device='cuda:0')
    actual = execution._predictor_args(model, dict(imgsz=960, confidence=.25, nms_iou=.5), 0, 'cuda:0')
    assert actual == expected


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('preparation', ['legacy', 'direct_bgr'])
def test_repeated_video_reuses_owners_but_reports_only_current_input(monkeypatch, streaming, preparation):
    from h3ce.infer.head_region_native import spatial_precision

    class Event:
        clock = 0

        def __init__(self, **kwargs):
            Event.clock += 1
            self.time = Event.clock

        def record(self, stream):
            pass

        def elapsed_time(self, other):
            return other.time-self.time

    original_report = execution._GpuWorker.report

    class Resource:
        report = original_report

        def __init__(self, cfg, root, device, worker, byte_preparation='legacy'):
            self.owner, self.worker = threading.get_ident(), worker
            self.byte_preparation = byte_preparation
            self.stream = SimpleNamespace(cuda_stream=worker+1)
            self.detector = SimpleNamespace(contract_id='cpu-test-only')
            self.calls, self.reuse_events = [], []

        def run(self, job, frames, origin):
            assert self.owner==threading.get_ident()
            self.calls.append(dict(job=job.index, shape=[len(frames),3,1,1], start_event=Event(), end_event=Event()))
            if not self.reuse_events:
                self.reuse_events.extend(['first_use', 'consumed'])
            return dict(job=job, faces=[float(frame[0,0,0]) for frame in frames])

    monkeypatch.setattr(execution, '_GpuWorker', Resource)
    monkeypatch.setattr(execution.torch.cuda, 'Event', Event)
    monkeypatch.setattr(execution.torch.cuda, 'current_stream', lambda device: None)
    with execution.Yolo11BatchExecutor({}, '.', profile='dual16', byte_preparation=preparation) as executor, spatial_precision():
        owners=[id(w.resource) for w in executor.workers]
        for value in [0., .5]:
            frames=np.full((33,1,1,3), value, np.float32)
            assert (executor.detect_stream(iter(frames)) if streaming else executor.detect_frames(frames))==[value]*33
            assert executor.last_report['counts']==dict(forwards=3, exposures=33)
            assert executor.last_report['byte_preparation']==preparation
            assert all(w['byte_preparation']==preparation for w in executor.last_report['workers'])
            calls=[c for w in executor.last_report['workers'] for c in w['calls']]
            assert sorted(c['job'] for c in calls)==[0,1,2] and all(c['cuda_start_ms']>=0 for c in calls)
        assert owners==[id(w.resource) for w in executor.workers]
        assert all(not w['reuse_events'] for w in executor.last_report['workers'])
