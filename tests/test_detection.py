"""CPU adapter contract tests; no fake runtime result is a YOLO/H3 acceptance result."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from h3ce.components import component_inventory, read_component_lock
from h3ce.data import detect_yolo11 as detection
from h3ce.errors import H3CEError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def locked_project(tmp_path):
    config = yaml.safe_load((PROJECT_ROOT / "configs/project.v2.yaml").read_text(encoding="utf-8"))
    entries = read_component_lock(PROJECT_ROOT / "components.lock.json")
    for role in ("person", "face"):
        # These bytes deliberately are not model weights and are never unpickled.
        path = tmp_path / entries[role]["local_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"CPU contract fixture for {role}".encode()
        path.write_bytes(payload)
        entries[role]["sha256"] = hashlib.sha256(payload).hexdigest()
        entries[role]["code_revision"] = "ultralytics==8.3.0"
        entries[role].pop("runtime_files_sha256", None)
    lock_path = tmp_path / "components.lock.json"

    def write():
        lock_path.write_text(json.dumps({"schema_version": 1, "components": list(entries.values())}), encoding="utf-8")

    write()
    return config, tmp_path, entries, write


def fake_detector(role, *, box=None):
    scale = "s" if role == "person" else "m"
    graph = [[copy.deepcopy(source), repeats, name, []] for source, repeats, name in detection._GRAPH]
    layers = []
    for source, _, name in detection._GRAPH:
        namespace = "torch.nn.modules.upsampling" if name == "nn.Upsample" else "ultralytics.nn.modules.block"
        cls = type(name.removeprefix("nn."), (), {"__module__": namespace})
        instance = cls()
        instance.f = copy.deepcopy(source)
        layers.append(instance)
    layers[-1].nc, layers[-1].nl, layers[-1].reg_max, layers[-1].end2end = 1, 3, 16, False
    detector = SimpleNamespace(
        task="detect", names={0: role},
        model=SimpleNamespace(yaml={"backbone": graph[:11], "head": graph[11:], "scale": scale, "scales": copy.deepcopy(detection._SCALES)}, model=layers),
        calls=[],
    )

    def predict(**kwargs):
        detector.calls.append(kwargs)
        h, w = kwargs["source"].shape[:2]
        coordinates = [[1, 2, w - 1, h - 1]] if box is None else box
        xyxy = np.asarray(coordinates, dtype=np.float32).reshape(-1, 4)
        return [SimpleNamespace(orig_shape=(h, w), boxes=SimpleNamespace(xyxy=xyxy, conf=np.full(len(xyxy), 0.9), cls=np.zeros(len(xyxy))))]

    detector.predict = predict
    return detector


def patch_runtime(monkeypatch, models=None):
    models = models or {role: fake_detector(role) for role in ("person", "face")}
    loaded = []

    def factory(path, task):
        assert Path(path).is_absolute() and Path(path).is_file() and task == "detect"
        role = "face" if "face" in Path(path).name else "person"
        loaded.append(role)
        return models[role]

    monkeypatch.setattr(detection, "_get_yolo_class", lambda revision: factory)
    return models, loaded


def test_t01_person_face_independent_cpu_contract(locked_project, monkeypatch):
    config, root, _, _ = locked_project
    models, loaded = patch_runtime(monkeypatch)
    detectors = detection.Yolo11Detectors(config, root)
    rgb = np.zeros((12, 20, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255
    result = detectors.detect(rgb)
    assert loaded == ["person", "face"]
    assert result["bbox_provenance"] == "source_detector"
    assert len(result["person_boxes"]) == len(result["face_boxes"]) == 1
    assert models["person"].calls[0]["source"][0, 0].tolist() == [0, 0, 255]
    assert models["person"].calls[0]["device"] == "cpu"
    assert models["face"].calls[0]["classes"] == [0]


def test_t01_coco_cannot_impersonate_face_cpu_contract(locked_project, monkeypatch):
    config, root, _, _ = locked_project
    models, _ = patch_runtime(monkeypatch)
    models["face"].names = {0: "person"}
    with pytest.raises(H3CEError, match="E_FACE_WEIGHTS"):
        detection.Yolo11Detectors(config, root)


def test_t01_same_weight_bytes_rejected_before_model_load(locked_project, monkeypatch):
    config, root, entries, write = locked_project
    payload = (root / entries["person"]["local_path"]).read_bytes()
    (root / entries["face"]["local_path"]).write_bytes(payload)
    entries["face"]["sha256"] = entries["person"]["sha256"]
    write()
    _, loaded = patch_runtime(monkeypatch)
    with pytest.raises(H3CEError, match="E_FACE_WEIGHTS"):
        detection.Yolo11Detectors(config, root)
    assert not loaded


@pytest.mark.parametrize("problem", ["missing_hash", "bad_hash", "missing_file", "missing_code", "unapproved_source", "mismatched_path"])
def test_unlocked_or_wrong_artifact_rejected_before_loading(locked_project, monkeypatch, problem):
    config, root, entries, write = locked_project
    if problem == "missing_hash":
        entries["face"]["sha256"] = None
    elif problem == "bad_hash":
        entries["face"]["sha256"] = "0" * 64
    elif problem == "missing_file":
        (root / entries["face"]["local_path"]).unlink()
    elif problem == "missing_code":
        entries["face"]["code_revision"] = None
    elif problem == "unapproved_source":
        entries["face"]["source_url"] = "https://example.com/yolov11m-face.pt"
    else:
        config["data"]["detection"]["face"]["weights"] = "models/another-face.pt"
    write()
    _, loaded = patch_runtime(monkeypatch)
    with pytest.raises(H3CEError, match="E_FACE_WEIGHTS"):
        detection.Yolo11Detectors(config, root)
    assert not loaded


@pytest.mark.parametrize("mutation", ["yaml_graph", "live_graph", "scale", "head_generation", "head_dfl", "module_origin"])
def test_renaming_other_architecture_yolo11_does_not_pass(locked_project, monkeypatch, mutation):
    config, root, _, _ = locked_project
    models, _ = patch_runtime(monkeypatch)
    face = models["face"]
    if mutation == "yaml_graph":
        face.model.yaml["backbone"][2][2] = "C2f"
    elif mutation == "live_graph":
        face.model.model[2].f = 0
    elif mutation == "scale":
        face.model.yaml["scale"] = "n"
    elif mutation == "head_generation":
        face.model.model[-1].end2end = True
    elif mutation == "head_dfl":
        face.model.model[-1].reg_max = 1
    else:
        type(face.model.model[2]).__module__ = "unknown_runtime"
    with pytest.raises(H3CEError, match="E_FACE_WEIGHTS"):
        detection.Yolo11Detectors(config, root)


@pytest.mark.parametrize("box", [[[0, 0, 21, 12]], [[3, 2, 1, 8]], [[0, 0, 0, 8]], [[-1, 0, 8, 8]], [[0, 0, float("nan"), 8]]])
def test_invalid_xyxy_rejected_instead_of_clipped(locked_project, monkeypatch, box):
    config, root, _, _ = locked_project
    patch_runtime(monkeypatch, {"person": fake_detector("person"), "face": fake_detector("face", box=box)})
    detectors = detection.Yolo11Detectors(config, root)
    with pytest.raises(H3CEError, match="E_FACE_WEIGHTS"):
        detectors.detect(np.zeros((12, 20, 3), dtype=np.uint8))


def test_missing_face_returns_empty_face_and_keeps_person(locked_project, monkeypatch):
    config, root, _, _ = locked_project
    patch_runtime(monkeypatch, {"person": fake_detector("person"), "face": fake_detector("face", box=[])})
    detectors = detection.Yolo11Detectors(config, root)
    result = detectors.detect(np.zeros((12, 20, 3), dtype=np.uint8))
    assert result["person_boxes"] and result["face_boxes"] == []


def test_contract_hash_changes_with_detector_settings_and_weights(locked_project, monkeypatch):
    config, root, entries, write = locked_project
    patch_runtime(monkeypatch)
    first = detection.Yolo11Detectors(config, root).contract_id
    assert first == detection.Yolo11Detectors(config, root).contract_id
    config["data"]["detection"]["face"]["confidence"] += 0.1
    second = detection.Yolo11Detectors(config, root).contract_id
    assert second != first
    face_path = root / entries["face"]["local_path"]
    face_path.write_bytes(b"another controlled CPU fixture")
    entries["face"]["sha256"] = hashlib.sha256(face_path.read_bytes()).hexdigest()
    write()
    assert detection.Yolo11Detectors(config, root).contract_id != second


def test_inventory_reports_unresolved_components_without_importing_runtime(monkeypatch, tmp_path):
    config = yaml.safe_load((PROJECT_ROOT / "configs/project.v2.yaml").read_text(encoding="utf-8"))
    document = json.loads((PROJECT_ROOT / "components.lock.json").read_text(encoding="utf-8"))
    for entry in document["components"]:
        entry["sha256"] = None
    (tmp_path / "components.lock.json").write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(detection, "_get_yolo_class", lambda *_: pytest.fail("inventory must not load a model"))
    result = component_inventory(config, tmp_path)
    assert not result["ready"]
    assert result["components"]
    assert all(item["model_acceptance"] == "not_run" for item in result["components"])


def test_contract_hash_changes_when_adapter_code_or_numpy_changes(locked_project, monkeypatch):
    config, root, _, _ = locked_project
    patch_runtime(monkeypatch)
    initial = detection.Yolo11Detectors(config, root).contract_id
    monkeypatch.setattr(detection, "sha256_file", lambda _: "f" * 64)
    changed_code = detection.Yolo11Detectors(config, root).contract_id
    assert changed_code != initial
    monkeypatch.setattr(detection.np, "__version__", "controlled-test-runtime-version")
    assert detection.Yolo11Detectors(config, root).contract_id != changed_code


def test_runtime_version_mismatch_rejected_before_import(monkeypatch):
    monkeypatch.setattr(detection.importlib.metadata, "version", lambda _: "9.9.9")
    with pytest.raises(H3CEError, match="E_COMPONENT_CODE"):
        detection._get_yolo_class("ultralytics==8.3.0")


@pytest.mark.real_model
def test_t01_real_yolo11_person_face_acceptance():
    """Opt-in actual-model smoke test on a user-supplied image containing person+face.

    This is CPU detector acceptance only. It cannot establish H3 GPU acceptance.
    Set H3CE_REAL_DETECTOR_IMAGE after locking local detector files/runtime.
    """
    image_path = os.environ.get("H3CE_REAL_DETECTOR_IMAGE")
    if not image_path:
        pytest.skip("Real YOLO11 acceptance not run: H3CE_REAL_DETECTOR_IMAGE is unset.")
    if importlib.util.find_spec("ultralytics") is None:
        pytest.skip("Real YOLO11 acceptance not run: Ultralytics is not installed.")
    config_path = Path(os.environ.get("H3CE_REAL_CONFIG", PROJECT_ROOT / "configs/project.v2.yaml"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    from PIL import Image, ImageOps

    detectors = detection.Yolo11Detectors(config, PROJECT_ROOT)
    with Image.open(image_path) as image:
        rgb = np.array(ImageOps.exif_transpose(image).convert("RGB"))
    result = detectors.detect(rgb)
    assert result["person_boxes"], "Acceptance image must produce a real person detection."
    assert result["face_boxes"], "Acceptance image must produce a real face detection."
