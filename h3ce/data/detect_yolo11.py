"""YOLO11 person/face adapter with offline locks and fail-closed contracts.

The module contains no detector approximation. Tests may inject a fake runtime to
exercise validation; those tests are CPU contract tests, not model acceptance.
"""

from __future__ import annotations

import importlib.metadata
import os
from copy import deepcopy
from pathlib import Path

import numpy as np

from h3ce.components import canonical_hash, config_dict, read_component_lock, require_locked_component, sha256_file
from h3ce.errors import H3CEError


# From Ultralytics' YOLO11 detection YAML. Compare both the serialized graph and
# live graph; a filename, a claimed family field, or one C3k2 block is insufficient.
_GRAPH = [
    (-1, 1, "Conv"), (-1, 1, "Conv"), (-1, 2, "C3k2"),
    (-1, 1, "Conv"), (-1, 2, "C3k2"), (-1, 1, "Conv"),
    (-1, 2, "C3k2"), (-1, 1, "Conv"), (-1, 2, "C3k2"),
    (-1, 1, "SPPF"), (-1, 2, "C2PSA"), (-1, 1, "nn.Upsample"),
    ([-1, 6], 1, "Concat"), (-1, 2, "C3k2"), (-1, 1, "nn.Upsample"),
    ([-1, 4], 1, "Concat"), (-1, 2, "C3k2"), (-1, 1, "Conv"),
    ([-1, 13], 1, "Concat"), (-1, 2, "C3k2"), (-1, 1, "Conv"),
    ([-1, 10], 1, "Concat"), (-1, 2, "C3k2"), ([16, 19, 22], 1, "Detect"),
]
_SCALES = {"n": [0.5, 0.25, 1024], "s": [0.5, 0.5, 1024], "m": [0.5, 1.0, 512], "l": [1.0, 1.0, 512], "x": [1.0, 1.5, 512]}


def _runtime_versions() -> dict:
    versions = {}
    for name in ("ultralytics", "torch", "torchvision", "opencv-python"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _get_yolo_class(expected_code_revision: str):
    expected_version = expected_code_revision.removeprefix("ultralytics==")
    try:
        actual = importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError as exc:
        raise H3CEError("E_DEPENDENCY", "The locked Ultralytics runtime is not installed.", {"required": expected_code_revision}) from exc
    if actual != expected_version:
        raise H3CEError("E_COMPONENT_CODE", "Ultralytics runtime differs from the component lock.", {"expected": expected_version, "actual": actual})
    # Set before importing: upstream computes these flags at import time.
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    try:
        from ultralytics import YOLO
        import ultralytics.utils as utilities
    except ImportError as exc:
        raise H3CEError("E_DEPENDENCY", "The locked Ultralytics runtime or a required dependency cannot be imported.", {"reason": str(exc)}) from exc
    # A previously imported runtime may have cached its flags; fail closed.
    if getattr(utilities, "AUTOINSTALL", True) or getattr(utilities, "ONLINE", True):
        raise H3CEError("E_COMPONENT_CODE", "Ultralytics was initialized with online/automatic install enabled; restart with YOLO_OFFLINE=true and YOLO_AUTOINSTALL=false.")
    return YOLO


def _validate_architecture(detector, entry: dict, role: str) -> tuple[dict[int, str], int]:
    code = "E_FACE_WEIGHTS" if role == "face" else "E_PERSON_WEIGHTS"
    try:
        model = detector.model
        yaml = model.yaml
        graph = yaml["backbone"] + yaml["head"]
        serialized_graph = [(item[0], item[1], item[2]) for item in graph]
        if serialized_graph != _GRAPH:
            raise ValueError("serialized graph differs from the verified YOLO11 detection topology")
        scale = yaml.get("scale")
        if scale != entry["architecture"][6] or list(yaml["scales"][scale]) != _SCALES[scale]:
            raise ValueError("YOLO11 model scale differs from the locked architecture")
        if getattr(detector, "task", None) != "detect":
            raise ValueError("runtime task is not detection")
        layers = list(model.model)
        if len(layers) != len(_GRAPH):
            raise ValueError("live module count differs from the verified YOLO11 graph")
        for index, (layer, (source, _, module_name)) in enumerate(zip(layers, _GRAPH)):
            if type(layer).__name__ != module_name.removeprefix("nn.") or layer.f != source:
                raise ValueError(f"live module signature differs at graph layer {index}")
            namespace = type(layer).__module__
            expected_namespace = "torch.nn.modules.upsampling" if module_name == "nn.Upsample" else "ultralytics.nn.modules."
            if not namespace.startswith(expected_namespace):
                raise ValueError(f"unexpected module implementation at graph layer {index}")
        names = detector.names
        if isinstance(names, list):
            names = dict(enumerate(names))
        if not isinstance(names, dict) or any(not isinstance(key, int) or not isinstance(value, str) for key, value in names.items()):
            raise ValueError("runtime class table is not an integer-to-name mapping")
        if set(names) != set(range(len(names))):
            raise ValueError("runtime class IDs are not contiguous")
        class_ids = [index for index, name in names.items() if name.casefold() == role]
        if len(class_ids) != 1:
            raise ValueError(f"runtime class table must contain one {role} class")
        head = layers[-1]
        if head.nc != len(names) or head.nl != 3 or head.reg_max != 16 or getattr(head, "end2end", False):
            raise ValueError("detection head does not match YOLO11 DFL/P3-P5 output semantics")
        return names, class_ids[0]
    except (AttributeError, KeyError, TypeError, IndexError, ValueError) as exc:
        raise H3CEError(code, f"{role} model failed the YOLO11 runtime architecture/class contract.", {"reason": str(exc)}) from exc


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _boxes_from_result(result, *, class_id: int, height: int, width: int, role: str) -> list[list[float]]:
    code = "E_FACE_WEIGHTS" if role == "face" else "E_PERSON_WEIGHTS"
    try:
        if tuple(result.orig_shape) != (height, width):
            raise ValueError("bbox original image shape differs from source geometry")
        boxes = result.boxes
        xyxy, confidence, classes = _numpy(boxes.xyxy), _numpy(boxes.conf), _numpy(boxes.cls)
        if xyxy.ndim != 2 or xyxy.shape[1] != 4 or confidence.shape != (len(xyxy),) or classes.shape != (len(xyxy),):
            raise ValueError("decoder must return xyxy[N,4], confidence[N], class[N]")
        if not all(np.isfinite(value).all() for value in (xyxy, confidence, classes)):
            raise ValueError("bbox outputs contain nonfinite values")
        if np.any(classes != np.floor(classes)) or np.any(classes != class_id):
            raise ValueError("decoder returned class IDs outside the requested class")
        if np.any((confidence < 0) | (confidence > 1)):
            raise ValueError("confidence outside [0,1]")
        if len(xyxy) and (np.any(xyxy[:, 0] < 0) or np.any(xyxy[:, 1] < 0) or np.any(xyxy[:, 2] > width) or np.any(xyxy[:, 3] > height) or np.any(xyxy[:, 2] <= xyxy[:, 0]) or np.any(xyxy[:, 3] <= xyxy[:, 1])):
            raise ValueError("xyxy boxes must have positive area within the source image")
        return np.column_stack((xyxy, confidence)).astype(float).tolist()
    except (AttributeError, TypeError, ValueError) as exc:
        raise H3CEError(code, f"{role} detector returned an invalid bbox contract.", {"reason": str(exc)}) from exc


class Yolo11Detectors:
    """Separate local person and face YOLO11 weights; no fallback or download."""

    def __init__(self, config, project_root: str | Path, *, device: str = "cpu"):
        config = config_dict(config)
        self.settings = deepcopy(config["data"]["detection"])
        if device != "cpu" and not (device.isdigit() or device.startswith("cuda:")):
            raise H3CEError("E_DETECT_DEVICE", "Detector device must be cpu or an explicit CUDA index.")
        self.device = device
        if self.settings["family"] != "yolo11" or self.settings["fallback_other_family"]:
            raise H3CEError("E_FACE_WEIGHTS", "Only YOLO11 with separate person and face weights is supported.")
        root = Path(project_root).resolve()
        lock_path = Path(config["paths"]["components_lock"])
        entries = read_component_lock(lock_path if lock_path.is_absolute() else root / lock_path)
        self.entries = {}
        paths = {}
        for role in ("person", "face"):
            if role not in entries:
                raise H3CEError("E_FACE_WEIGHTS" if role == "face" else "E_PERSON_WEIGHTS", f"Missing {role} component lock.")
            self.entries[role] = entries[role]
            if self.settings[role]["class_name"] != role:
                raise H3CEError("E_FACE_WEIGHTS" if role == "face" else "E_PERSON_WEIGHTS", "Configured detector class must match its role.")
            paths[role] = require_locked_component(entries[role], root, role=role, configured_weights=self.settings[role]["weights"])
        if paths["person"] == paths["face"] or entries["person"]["sha256"].lower() == entries["face"]["sha256"].lower():
            raise H3CEError("E_FACE_WEIGHTS", "Person and face detectors must use independent weight artifacts.")
        if entries["person"]["code_revision"] != entries["face"]["code_revision"]:
            raise H3CEError("E_COMPONENT_CODE", "Person and face locks must agree on the installed runtime revision.")
        self.contract_id = canonical_hash({
            "adapter": "h3ce_yolo11_detection_v1", "input": "RGB-uint8-HWC", "output": "xyxy_conf",
            "adapter_code_sha256": sha256_file(Path(__file__)), "numpy": np.__version__,
            "components": self.entries, "detection": self.settings, "device": self.device,
            "runtime_dependencies": _runtime_versions(),
        })
        from h3ce.acquire import verify_detector_runtime_sources
        for entry in self.entries.values():
            verify_detector_runtime_sources(entry)
        factory = _get_yolo_class(entries["person"]["code_revision"])
        self.models, self.class_ids = {}, {}
        for role in ("person", "face"):
            try:
                detector = factory(str(paths[role]), task="detect")
            except Exception as exc:
                raise H3CEError("E_FACE_WEIGHTS" if role == "face" else "E_PERSON_WEIGHTS", f"Failed to load the verified local {role} artifact; no fallback is available.", {"reason": str(exc)}) from exc
            _, self.class_ids[role] = _validate_architecture(detector, entries[role], role)
            self.models[role] = detector

    def detect(self, rgb: np.ndarray) -> dict:
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
            raise H3CEError("E_DETECT_INPUT", "Detector input must be a nonempty RGB uint8 HWC array.")
        height, width = rgb.shape[:2]
        # Ultralytics' NumPy API expects BGR. Do not accidentally invert face colors.
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        output = {"bbox_provenance": "source_detector"}
        for role in ("person", "face"):
            settings = self.settings[role]
            try:
                results = self.models[role].predict(
                    source=bgr, imgsz=settings["imgsz"], conf=settings["confidence"],
                    iou=settings["nms_iou"], classes=[self.class_ids[role]],
                    stream=False, save=False, verbose=False, augment=False, device=self.device,
                )
                if len(results) != 1:
                    raise ValueError("Expected exactly one result for one source frame")
            except Exception as exc:
                raise H3CEError("E_FACE_WEIGHTS" if role == "face" else "E_PERSON_WEIGHTS", f"{role} detector inference failed; no fallback is available.", {"reason": str(exc)}) from exc
            output[f"{role}_boxes"] = _boxes_from_result(results[0], class_id=self.class_ids[role], height=height, width=width, role=role)
        return output
