"""CPU control-flow checks, not GPU detection or performance tests."""
from types import SimpleNamespace

import pytest
import torch

from h3ce.data import yolo11_face_reuse as reuse


class Backend(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.format, self.fp16 = 'pt', False
        self.device = SimpleNamespace(type='cuda')
        self.backend = SimpleNamespace(model=torch.nn.Identity().eval())
        self.calls = 0

    def forward(self, x, **kwargs):
        self.calls += 1
        return x * 2

    def warmup(self, *, im):
        self.forward(im)


def setup(monkeypatch):
    monkeypatch.setattr(reuse, '_check_runtime', lambda: None)
    backend = Backend()
    model = SimpleNamespace(callbacks={'on_predict_start': []})
    predictor = SimpleNamespace(model=backend, done_warmup=False,
                                args=SimpleNamespace(augment=False, embed=None))
    return model, predictor, backend


def start(model, predictor):
    for callback in model.callbacks['on_predict_start']:
        callback(predictor)


def check_restored(model, backend):
    assert not model.callbacks['on_predict_start']
    assert 'forward' not in backend.__dict__ and 'warmup' not in backend.__dict__


def test_source_reused_once_and_later_input_runs(monkeypatch):
    model, predictor, backend = setup(monkeypatch)
    x = torch.arange(8, dtype=torch.float32)
    with reuse.reuse_actual_warmup(model) as events:
        start(model, predictor)
        backend.warmup(im=x)
        assert torch.equal(backend(x), x * 2) and backend.calls == 1
        assert torch.equal(backend(x + 1), (x + 1) * 2) and backend.calls == 2
        assert len(events) == 2
    check_restored(model, backend)


@pytest.mark.parametrize('change', ['different_object', 'in_place'])
def test_changed_input_fails_and_restores(monkeypatch, change):
    model, predictor, backend = setup(monkeypatch)
    x = torch.arange(8, dtype=torch.float32)
    with pytest.raises(RuntimeError, match='different or modified'):
        with reuse.reuse_actual_warmup(model):
            start(model, predictor)
            backend.warmup(im=x)
            if change == 'different_object':
                x = x.clone()
            else:
                x.add_(1)
            backend(x)
    check_restored(model, backend)


def test_unconsumed_result_fails_and_restores(monkeypatch):
    model, predictor, backend = setup(monkeypatch)
    with pytest.raises(RuntimeError, match='Unconsumed'):
        with reuse.reuse_actual_warmup(model):
            start(model, predictor)
            backend.warmup(im=torch.ones(2))
    check_restored(model, backend)


def test_batch_results_keep_empty_images_and_class_checks():
    def result(data):
        return SimpleNamespace(orig_shape=(32, 40), boxes=SimpleNamespace(data=torch.tensor(data).reshape(-1, 6).float()))
    rows = [result([[1, 2, 10, 20, .5, 0]]), result([]), result([[3, 4, 8, 9, 1, 0]])]
    assert reuse.batch_boxes_to_cpu(rows, class_id=0, height=32, width=40) == [
        [[1., 2., 10., 20., .5]], [], [[3., 4., 8., 9., 1.]]]
