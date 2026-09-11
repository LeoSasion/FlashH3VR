"""Owned YOLO11 workers, explicit CUDA streams and ordered frame results.

Only the three declared execution profiles are accepted. The locked upstream
preprocess/inference/postprocess functions are retained; its device-wide timing
barriers are replaced with stream events. Model and numerical settings stay fixed.
"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from pathlib import Path
import threading
import time

import numpy as np
import torch

from .detect_yolo11 import Yolo11Detectors
from .yolo11_face_reuse import _check_runtime, reuse_actual_warmup, batch_boxes_to_cpu
from .face_preparation import BYTE_PREPARATIONS, FaceBytePreparer


PROFILES = {'single16': (16, 1), 'single32': (32, 1), 'dual16': (16, 2)}


def _predictor_args(model, settings, class_id, device):
    custom = dict(conf=.25, batch=1, save=False, mode='predict', rect=True, embed=None)
    explicit = dict(imgsz=settings['imgsz'], conf=settings['confidence'], iou=settings['nms_iou'],
        classes=[class_id], save=False, verbose=False, augment=False, quantize=None, device=str(device))
    return {**model.overrides, **custom, **explicit}


@dataclass(frozen=True)
class FaceJob:
    index: int
    start: int
    stop: int
    worker: int


def plan_face_jobs(frame_count, profile):
    if type(frame_count) is not int or frame_count < 1 or profile not in PROFILES:
        raise ValueError('Positive frame count and an explicit supported face profile required')
    batch, workers = PROFILES[profile]
    return [FaceJob(i, a, min(a + batch, frame_count), i % workers)
            for i, a in enumerate(range(0, frame_count, batch))]


def collect_ordered_jobs(jobs, workers, frames, origin):
    """Keep at most one active batch per owner, propagate failures and drain."""
    expected = 0
    for i, job in enumerate(jobs):
        if (job.index != i or job.start != expected or job.stop <= job.start
                or not 0 <= job.worker < len(workers)):
            raise ValueError('Job plan must cover each input frame once, in order')
        expected = job.stop
    if expected != len(frames):
        raise ValueError('Job plan does not cover the complete input')
    owned = {i: iter([job for job in jobs if job.worker == i]) for i in range(len(workers))}
    pending, results = {}, {}

    def submit_next(owner):
        job = next(owned[owner], None)
        if job is not None:
            pending[workers[owner].submit(job, frames[job.start:job.stop], origin)] = job

    try:
        for owner in owned:
            submit_next(owner)
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            # Inspect all completed futures before issuing any more GPU work.
            ready = [(future, pending.pop(future)) for future in done]
            completed, failures = [], []
            for future, job in ready:
                try:
                    completed.append((job, future.result()))
                except BaseException as error:
                    failures.append(error)
            if failures:
                raise failures[0]
            for job, result in completed:
                if result['job'] != job or len(result['faces']) != job.stop - job.start:
                    raise ValueError('Worker changed its input identity or result count')
                results[job.index] = result
            for job, _ in completed:
                submit_next(job.worker)
    except BaseException:
        for future in pending:
            future.cancel()
        wait(pending)
        for future in pending:
            if not future.cancelled():
                future.exception()
        raise
    return [results[i] for i in range(len(jobs))]


def collect_stream_jobs(frames, workers, profile, origin, *, on_input_complete=None):
    """Dispatch immutable source frames as they arrive, with bounded owner queues.

    Each owned single-thread pool may hold one running and one queued batch.
    Reading and the CPU completion callback run on the caller's thread. Any
    source, callback or worker failure cancels queued work and drains active work
    before returning control, including before precision flags may be restored.
    """
    if profile not in PROFILES or len(workers) != PROFILES[profile][1]:
        raise ValueError('Streaming profile must match the owned worker count')
    batch_size, owner_count = PROFILES[profile]
    pending, results, jobs, buffered = {}, {}, [], []
    submitted, shape, frame_count = [], None, 0
    peak_pending, peak_by_owner = 0, [0] * owner_count
    iterator = iter(frames)
    started = time.perf_counter()

    def collect(done):
        completed, failures = [], []
        for future in done:
            job = pending.pop(future)
            try:
                result = future.result()
                if result['job'] != job or len(result['faces']) != job.stop - job.start:
                    raise ValueError('Worker changed its input identity or result count')
                completed.append((job, result))
            except BaseException as error:
                failures.append(error)
        if failures:
            raise failures[0]
        for job, result in completed:
            results[job.index] = result

    def check_ready():
        collect([future for future in pending if future.done()])

    def submit_buffer():
        nonlocal peak_pending
        owner = len(jobs) % owner_count
        check_ready()
        while sum(job.worker == owner for job in pending.values()) >= 2:
            collect(wait(pending, return_when=FIRST_COMPLETED)[0])
        start = jobs[-1].stop if jobs else 0
        job = FaceJob(len(jobs), start, frame_count, owner)
        dispatched = time.perf_counter()
        # A tuple retains the reader's original owned allocations; no batch stack
        # or reusable decode buffer can overwrite a running worker's input.
        future = workers[owner].submit(job, tuple(buffered), origin)
        pending[future] = job
        jobs.append(job)
        submitted.append(dict(job=job.index, wall_time=dispatched, source_frames_read=frame_count))
        buffered.clear()
        peak_pending = max(peak_pending, len(pending))
        for index in range(owner_count):
            peak_by_owner[index] = max(peak_by_owner[index], sum(j.worker == index for j in pending.values()))

    try:
        while True:
            check_ready()
            try:
                frame = next(iterator)
            except StopIteration:
                break
            check_ready()
            if (not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[-1] != 3
                    or frame.dtype != np.float32 or min(frame.shape) < 1):
                raise ValueError('Expected nonempty immutable FP32 HWC source frames')
            if shape is not None and frame.shape != shape:
                raise ValueError('Streaming source dimensions changed')
            shape = frame.shape
            buffered.append(frame)
            frame_count += 1
            if len(buffered) == batch_size:
                submit_buffer()
        exhausted = time.perf_counter()
        if frame_count == 0:
            raise ValueError('Expected at least one source frame')
        if buffered:
            submit_buffer()
        check_ready()
        callback_started = time.perf_counter()
        if on_input_complete is not None:
            on_input_complete()
        callback_ended = time.perf_counter()
        while pending:
            collect(wait(pending, return_when=FIRST_COMPLETED)[0])
    except BaseException:
        for future in pending:
            future.cancel()
        wait(pending)
        for future in pending:
            if not future.cancelled():
                future.exception()
        raise
    finally:
        close = getattr(iterator, 'close', None)
        if close is not None:
            close()
    return [results[index] for index in range(len(jobs))], dict(
        wall_start=started, input_exhausted_wall=exhausted,
        input_callback_start_wall=callback_started, input_callback_end_wall=callback_ended,
        all_jobs_drained_wall=time.perf_counter(), submissions=submitted,
        peak_pending_batches=peak_pending, peak_pending_by_owner=peak_by_owner,
        per_owner_pending_limit=2, total_pending_limit=2*owner_count,
        note='Input and worker wall intervals overlap and must not be added as exclusive stages')


class _GpuWorker:
    def __init__(self, cfg, root, device, worker, byte_preparation='legacy'):
        self.owner = threading.get_ident()
        self.worker, self.device = worker, torch.device(device)
        self.byte_preparation = byte_preparation
        self.preparer = None if byte_preparation == 'legacy' else FaceBytePreparer(byte_preparation)
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.Stream(device=self.device)
        self.detector = Yolo11Detectors(cfg, root, device=device)
        _check_runtime()  # The verified adapter sets offline flags before importing Ultralytics.
        self.calls, self.reuse_events = [], []
        self.current_job = None
        self.origin = None
        # setup_model deep-copies its source module. Free closures keep the owned
        # logger identity; bound methods would also deepcopy this worker/stream.
        self.detector.models['face'].model.register_forward_pre_hook(lambda m, a: self._before(m, a))
        self.detector.models['face'].model.register_forward_hook(lambda *a: self._after(*a))
        self.detector.models['person'].model.register_forward_pre_hook(self._reject_person)

    @staticmethod
    def _reject_person(*_):
        raise RuntimeError('Unexpected person model forward in face-only execution')

    def _before(self, module, args):
        if threading.get_ident() != self.owner or self.current_job is None:
            raise RuntimeError('Model invoked outside its owned worker job')
        event = torch.cuda.Event(enable_timing=True)
        event.record(self.stream)
        self.calls.append(dict(job=self.current_job.index, worker=self.worker, shape=list(args[0].shape),
            dtype=str(args[0].dtype), thread_id=self.owner, stream=int(self.stream.cuda_stream),
            start_event=event, completed=False, wall_start=time.perf_counter()))

    def _after(self, *_):
        event = torch.cuda.Event(enable_timing=True)
        event.record(self.stream)
        self.calls[-1].update(end_event=event, completed=True, wall_end=time.perf_counter())

    def _predict(self, byte_frames, *, already_bgr=False):
        model = self.detector.models['face']
        settings = self.detector.settings['face']
        if model.predictor is None:
            # Match Model.predict's resolved defaults; half=False maps to None.
            args = _predictor_args(model, settings, self.detector.class_ids['face'], self.device)
            model.predictor = model._smart_load('predictor')(overrides=args, _callbacks=model.callbacks)
            model.predictor.setup_model(model=model.model, verbose=False)
        p = model.predictor
        if (p.model.fp16 or p.model.format != 'pt' or p.args.augment or p.args.embed
                or p.args.visualize or p.args.save or p.args.save_txt or p.args.save_crop or p.args.show or p.args.compile
                or p.args.task != 'detect' or p.model.backend.model.training):
            raise RuntimeError('Unsupported predictor state for fixed FP32 face execution')
        bgr = byte_frames if already_bgr else [np.ascontiguousarray(frame[:, :, ::-1]) for frame in byte_frames]
        p.setup_source(bgr)
        dataset = iter(p.dataset)
        p.batch = next(dataset)
        if next(dataset, None) is not None or len(p.batch[1]) != len(byte_frames):
            raise RuntimeError('Expected exactly one complete NumPy source batch')
        p.seen, p.results = 0, None
        with reuse_actual_warmup(model) as events:
            p.run_callbacks('on_predict_start')
            p.run_callbacks('on_predict_batch_start')
            im = p.preprocess(p.batch[1])
            if im.dtype != torch.float32:
                raise RuntimeError('Unexpected detector precision')
            if not p.done_warmup:
                p.model.warmup(im=im)
                p.done_warmup = True
            preds = p.inference(im)
            p.results = p.postprocess(preds, im, p.batch[1])
            p.run_callbacks('on_predict_postprocess_end')
            p.seen = len(byte_frames)
            p.run_callbacks('on_predict_batch_end')
            p.run_callbacks('on_predict_end')
        self.reuse_events.extend(events)
        h, w, _ = byte_frames[0].shape
        return batch_boxes_to_cpu(p.results, class_id=self.detector.class_ids['face'], height=h, width=w)

    def run(self, job, frames, origin):
        if threading.get_ident() != self.owner or job.worker != self.worker:
            raise RuntimeError('Worker/model ownership mismatch')
        before = len(self.calls)
        self.current_job, self.origin = job, origin
        started = time.perf_counter()
        try:
            with torch.cuda.device(self.device), torch.cuda.stream(self.stream), torch.inference_mode():
                self.stream.wait_event(origin)
                byte = (self.preparer.prepare_bgr(frames) if self.preparer is not None else
                        [np.uint8(frame.clip(0, 1) * 255 + .5) for frame in frames])
                prepared = time.perf_counter()
                faces = self._predict(byte, already_bgr=self.preparer is not None)
        finally:
            try:
                self.stream.synchronize()
            finally:
                self.current_job = None
        ended = time.perf_counter()
        if len(self.calls) - before != 1 or not self.calls[-1]['completed']:
            raise RuntimeError('Each source batch must execute the face model exactly once')
        return dict(job=job, faces=faces, prepare_seconds=prepared-started,
                    predict_seconds=ended-prepared, wall_start=started, wall_end=ended)

    def report(self, origin, first_call=0, first_reuse=0):
        calls = []
        for row in self.calls[first_call:]:
            data = {k: v for k, v in row.items() if k not in ('start_event', 'end_event')}
            data.update(cuda_start_ms=origin.elapsed_time(row['start_event']),
                        cuda_end_ms=origin.elapsed_time(row['end_event']),
                        cuda_interval_ms=row['start_event'].elapsed_time(row['end_event']))
            calls.append(data)
        return dict(worker=self.worker, thread_id=self.owner, stream=int(self.stream.cuda_stream),
                    byte_preparation=self.byte_preparation,
                    detector_contract=self.detector.contract_id, calls=calls, reuse_events=self.reuse_events[first_reuse:])


class _OwnedThread:
    def __init__(self, factory, index):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f'h3ce-face-{index}')
        try:
            self.resource = self.pool.submit(factory).result()
        except BaseException:
            self.pool.shutdown(wait=True, cancel_futures=True)
            raise

    def submit(self, job, frames, origin):
        return self.pool.submit(self.resource.run, job, frames, origin)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


class Yolo11BatchExecutor:
    """Persistent owned workers; caller controls one fixed precision scope.

    A call returns results in source order and fully drains its streams. Calls on
    one executor cannot overlap. CPU frames must remain immutable until return.
    """
    def __init__(self, cfg, project_root, *, profile='single16', device='cuda:0', byte_preparation='legacy'):
        if profile not in PROFILES or not device.startswith('cuda:') or byte_preparation not in BYTE_PREPARATIONS:
            raise ValueError('Explicit supported profile and CUDA index required')
        self.profile, self.device = profile, torch.device(device)
        self.byte_preparation = byte_preparation
        self.workers, self.last_report = [], None
        self._closed, self._call_lock = False, threading.Lock()
        try:
            for i in range(PROFILES[profile][1]):
                self.workers.append(_OwnedThread(lambda i=i: _GpuWorker(cfg, Path(project_root), device, i,
                                                                      byte_preparation=byte_preparation), i))
        except BaseException:
            self.close()
            raise

    def detect_frames(self, frames):
        if (not isinstance(frames, np.ndarray) or frames.ndim != 4 or frames.shape[-1] != 3
                or frames.dtype != np.float32 or min(frames.shape) < 1):
            raise ValueError('Expected nonempty FP32 THWC frames')
        if self._closed or not self._call_lock.acquire(blocking=False):
            raise RuntimeError('Executor is closed or already processing another input')
        try:
            if torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32:
                raise RuntimeError('Caller must hold fixed strict FP32 flags until workers finish')
            jobs = plan_face_jobs(len(frames), self.profile)
            first = [(len(w.resource.calls), len(w.resource.reuse_events)) for w in self.workers]
            origin = torch.cuda.Event(enable_timing=True)
            origin.record(torch.cuda.current_stream(self.device))
            started = time.perf_counter()
            rows = collect_ordered_jobs(jobs, self.workers, frames, origin)
            seconds = time.perf_counter()-started
            faces = [face for row in rows for face in row['faces']]
            reports = [worker.resource.report(origin, *position) for worker, position in zip(self.workers, first)]
            actual_calls = [call for report in reports for call in report['calls']]
            self.last_report = dict(profile=self.profile, byte_preparation=self.byte_preparation,
                frames=len(frames), wall_seconds=seconds,
                jobs=[dict(index=r['job'].index, start=r['job'].start, stop=r['job'].stop, worker=r['job'].worker,
                           **{k:v for k,v in r.items() if k not in ('job','faces')}) for r in rows],
                workers=reports, counts=dict(forwards=len(actual_calls), exposures=sum(c['shape'][0] for c in actual_calls)),
                note='CUDA event intervals include queued work/gaps; overlap is not a kernel-level utilization proof')
            return faces
        finally:
            self._call_lock.release()

    def detect_stream(self, frames, *, on_input_complete=None):
        """Read an immutable frame iterator while the owned detectors execute."""
        if self._closed or not self._call_lock.acquire(blocking=False):
            raise RuntimeError('Executor is closed or already processing another input')
        try:
            self.last_report = None
            if torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32:
                raise RuntimeError('Caller must hold fixed strict FP32 flags until workers finish')
            first = [(len(w.resource.calls), len(w.resource.reuse_events)) for w in self.workers]
            origin = torch.cuda.Event(enable_timing=True)
            origin.record(torch.cuda.current_stream(self.device))
            started = time.perf_counter()
            rows, schedule = collect_stream_jobs(frames, self.workers, self.profile, origin,
                                                on_input_complete=on_input_complete)
            seconds = time.perf_counter()-started
            faces = [face for row in rows for face in row['faces']]
            reports = [worker.resource.report(origin, *position) for worker, position in zip(self.workers, first)]
            actual_calls = [call for report in reports for call in report['calls']]
            self.last_report = dict(profile=self.profile, byte_preparation=self.byte_preparation,
                frames=len(faces), wall_seconds=seconds,
                jobs=[dict(index=r['job'].index, start=r['job'].start, stop=r['job'].stop, worker=r['job'].worker,
                           **{k:v for k,v in r.items() if k not in ('job','faces')}) for r in rows],
                workers=reports, counts=dict(forwards=len(actual_calls), exposures=sum(c['shape'][0] for c in actual_calls)),
                input_overlap=schedule,
                note='Whole read/callback/detection overlap interval; CUDA intervals include queued work/gaps')
            return faces
        finally:
            self._call_lock.release()

    def close(self):
        with self._call_lock:
            if not self._closed:
                self._closed = True
                for worker in self.workers:
                    worker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
