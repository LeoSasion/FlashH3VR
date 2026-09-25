"""Synthetic CPU checks for the new full-scene Dense release path."""

from fractions import Fraction
import json

import av
import numpy as np
import pytest
import torch

from flashh3vr import DenseRestorer
from flashh3vr.full_video import plan_full_video, restore_full_video_frames, restore_full_video_file
from h3ce.infer.head_video_file import encode_srgb_h264


def _faces(count):
    return [[[20.0, 20.0, 100.0, 100.0, 0.95]] for _ in range(count)]


def test_short_h3_padding_is_internal_and_trims_to_real_frames(monkeypatch):
    import flashh3vr.inference as inference

    class FakeH3:
        def __init__(self):
            self.model = torch.nn.Module().eval()
            self.model.register_buffer("latents_mean", torch.zeros(24))
            self.model.register_buffer("latents_std", torch.ones(24))
            self.context = []

        def encode_mean_raw(self, x):
            self.context.append(x.shape[2])
            return torch.zeros(1, 24, {5: 2, 22: 7}[x.shape[2]], 16, 16)

        def decode_raw(self, z):
            return torch.zeros(1, 3, {2: 5, 7: 22}[z.shape[2]], 256, 256)

    monkeypatch.setattr(inference, "native_tiled_dense_inter", lambda z, _dense, _model: (z, {"inter_module_calls": 1}))
    restorer = DenseRestorer.__new__(DenseRestorer)
    restorer.device = torch.device("cpu")
    restorer.h3 = FakeH3()
    restorer.dense = torch.nn.Identity().eval()
    restorer.last_plan = None
    for real, context in ((2, 5), (5, 5), (6, 22), (22, 22)):
        rgb = torch.full((1, 3, real, 256, 256), 0.4)
        output = restorer.restore_video_window(rgb, pts=[i / 25 for i in range(real)])
        assert output.shape == rgb.shape
        assert restorer.h3.context[-1] == context
        assert restorer.last_plan["valid_frames"] == real
        assert restorer.last_plan["h3_context_frames"] == context
        assert restorer.last_plan["padded_frames"] == context
        assert restorer.last_plan["padding_frames_added"] == context - real


def test_planner_keeps_real_pts_and_only_same_frame_overlap():
    count = 44
    pts = [i / 25 for i in range(count)]
    plan = plan_full_video(_faces(count), pts, [0] * count, (128, 128), side=256)
    assert plan["used_frames"] == list(range(count))
    assert [(row["start"], row["stop"], row["valid_frames"], row["padded_frames"])
            for row in plan["chunks"]] == [(0, 22, 22, 22), (17, 39, 22, 22), (34, 44, 10, 22)]
    assert plan["pts"] == pts
    assert all(row["padding_frames_added"] >= 0 for row in plan["chunks"])


def test_cut_gap_missing_and_ambiguous_faces_split_context():
    count = 12
    pts = [i / 25 for i in range(count)]
    pts[10:] = [value + .5 for value in pts[10:]]
    faces = _faces(count)
    faces[3] = []
    faces[6] = [faces[6][0], [21, 20, 99, 100, .8]]
    shots = [0] * 9 + [1] * 3
    plan = plan_full_video(faces, pts, shots, (128, 128), side=448)
    assert [(c["start"], c["stop"]) for c in plan["chunks"]] == [(0, 3), (4, 6), (7, 9), (10, 12)]
    assert plan["skipped_frames"] == {3: "missing_face", 6: "ambiguous_faces",
                                       9: "isolated_face_no_video_context"}
    assert [c["padded_frames"] for c in plan["chunks"]] == [5, 5, 5, 5]


def test_full_scene_pastes_only_head_delta_and_blends_overlap():
    count = 23
    frames = torch.full((count, 3, 128, 128), 0.4)
    frames[:, :, 0, 0] = .2
    pts = [i / 25 for i in range(count)]

    class ConstantRestorer:
        device = torch.device("cpu")
        last_plan = {"repair_steps": 1}

        def restore_video_window(self, value, *, pts):
            assert value.shape[2] == len(pts)
            return value + .1

    result = restore_full_video_frames(frames, pts, _faces(count), [0] * count,
                                       ConstantRestorer(), side=256)
    output = result["prediction"]
    assert len(result["window_reports"]) == 2
    assert [(w["valid_frames"], w["padded_frames"]) for w in result["window_reports"]] == [(22, 22), (6, 22)]
    assert torch.equal(output[:, :, 0, 0], frames[:, :, 0, 0])
    assert (output[:, :, 60, 60] > frames[:, :, 60, 60]).all()
    assert torch.allclose(output[17:22, :, 60, 60], output[16:17, :, 60, 60].expand(5, -1), atol=1e-5)


def test_file_pipeline_decodes_detects_pastes_encodes_and_keeps_pts(tmp_path, monkeypatch):
    import flashh3vr.full_video as full_video

    source, destination = tmp_path / "source.mp4", tmp_path / "restored.mp4"
    frames = torch.full((6, 3, 128, 128), .35)
    pts_integer = [0, 2, 5, 7, 11, 17]
    encode_srgb_h264(source, frames, pts_integer, [[1, 120]] * 6, rate=Fraction(60))

    class Detector:
        def __init__(self, *_args, **_kwargs):
            pass

        def detect_frames(self, rgb):
            assert rgb.shape == (6, 128, 128, 3)
            return _faces(6)

    class Restorer:
        device = torch.device("cpu")
        last_plan = {"repair_steps": 1}

        def __init__(self, **_kwargs):
            pass

        def restore_video_window(self, value, *, pts):
            assert len(pts) == 6
            return value + .02

    monkeypatch.setattr(full_video, "PinnedFaceDetector", Detector)
    monkeypatch.setattr(full_video, "DenseRestorer", Restorer)
    result = restore_full_video_file(source, destination, h3_weights="stub-h3",
                                     dense_weights="stub-dense", face_weights="stub-face",
                                     side=256, max_frames=6, working_long_edge=0, device="cpu")
    assert result["restored_frames"] == 6
    with av.open(str(destination)) as container:
        actual_pts = [frame.pts * frame.time_base for frame in container.decode(video=0)]
    assert actual_pts == [Fraction(value, 120) for value in pts_integer]
    receipt = json.loads((tmp_path / "restored.flashh3vr.json").read_text(encoding="utf-8"))
    assert receipt["source_canvas_hw"] == receipt["working_output_canvas_hw"] == [128, 128]
    assert receipt["window_reports"][0]["valid_frames"] == 6
    assert receipt["window_reports"][0]["padded_frames"] == 22
    with pytest.raises(Exception, match="frame limit|frame_limit|Source exceeds"):
        restore_full_video_file(source, tmp_path / "too_short_limit.mp4",
                                h3_weights="stub-h3", dense_weights="stub-dense",
                                face_weights="stub-face", side=256, max_frames=5,
                                working_long_edge=0, device="cpu")
    assert not (tmp_path / "too_short_limit.mp4").exists()
