"""Strict parsing and safety invariants from AGENT_SPEC sections 0/3/5/6/10."""

from copy import deepcopy
import math
from pathlib import Path

import pytest
import yaml

from h3ce.config import ProjectConfig, load_config, project_root, write_resolved
from h3ce.errors import H3CEError


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def source():
    return yaml.safe_load((ROOT / "configs" / "project.v2.yaml").read_text(encoding="utf-8"))


def write_change(tmp_path, source, dotted, value):
    data = deepcopy(source)
    keys = dotted.split(".")
    parent = data
    for key in keys[:-1]:
        parent = parent[key]
    parent[keys[-1]] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_shipped_yaml_complete_and_roundtrips(tmp_path, source):
    config = load_config(ROOT / "configs" / "project.v2.yaml")
    assert config.model_dump(mode="json") == source
    output = write_resolved(config, tmp_path / "run" / "resolved.yaml")
    assert load_config(output) == config
    assert list(output.parent.glob("*.partial")) == []
    assert config.data.degradation.variants_per_image == 4
    assert config.data.degradation.resolution.ratio_range == [0.25, 0.75]


@pytest.mark.parametrize("dotted", [
    "unknown", "data.unknown", "data.degradation.blur.unknown",
    "data.detection.face.unknown", "model.scene_context.unknown",
    "training.stages.character_lora.unknown", "lora.routing.coefficients.close.unknown",
])
def test_unknown_fields_at_every_depth_rejected(tmp_path, source, dotted):
    with pytest.raises(H3CEError) as error:
        load_config(write_change(tmp_path, source, dotted, True))
    assert error.value.code == "E_CONFIG"
    assert "Extra inputs" in str(error.value)


@pytest.mark.parametrize("dotted,value", [
    ("data.degradation.variants_per_image", 0),
    ("data.degradation.variants_per_image", -1),
    ("data.degradation.variants_per_image", True),
    ("data.degradation.variants_per_image", 4.0),
    ("data.degradation.variants_per_clip", "2"),
    ("project.budget_seconds", True),
    ("training.microbatch", 1.5),
    ("native_temporal.last_n_blocks", 0),
    ("data.degradation.resolution.short_edge_range", [True, 512]),
    ("data.degradation.resolution.ratio_range", [0, 0.5]),
    ("data.degradation.resolution.ratio_range", [0.9, 0.5]),
    ("data.degradation.resolution.ratio_range", [0.5, 1.01]),
    ("data.degradation.resolution.ratio_range", [0.5]),
    ("data.degradation.resolution.short_edge_range", [513, 512]),
    ("data.degradation.blur.sigma_reference_px_range", [-1, 1]),
    ("data.degradation.blur.sigma_reference_px_range", [2, 1]),
    ("data.degradation.blur.probability", 1.01),
    ("data.degradation.noise.probability", -0.01),
    ("data.degradation.noise.sigma_255_range", [3, 2]),
    ("data.degradation.compression.jpeg_quality_range", [101, 102]),
    ("data.degradation.compression.h264_crf_range", [52, 53]),
    ("data.degradation.compression.jpeg_quality_range", [90, 80]),
    ("data.degradation.compression.probability", 0.5),
    ("data.detection.face.confidence", math.nan),
    ("lora.routing.coefficients.close.face", math.inf),
    ("training.stages.bootstrap_pixel.lr", math.inf),
    ("model.output.strength", -math.inf),
    ("data.sampling.debug_buckets_hw", [[257, 256]]),
    ("inference.overlap_frames", 22),
    ("inference.chunk_frames", 6),
    ("native_temporal.train_clip_frames", 1),
    ("native_temporal.image_replay_fraction", 1.0),
])
def test_invalid_counts_ranges_probabilities_and_finite_numbers(tmp_path, source, dotted, value):
    with pytest.raises(H3CEError) as error:
        load_config(write_change(tmp_path, source, dotted, value))
    assert error.value.code == "E_CONFIG"


@pytest.mark.parametrize("dotted,value", [
    ("vae.encoder_trainable", True),
    ("vae.preserve_pts", False),
    ("vae.clamp_before_loss", True),
    ("native_temporal.freeze_refiner_and_character_loras", False),
    ("native_temporal.cache_trainable_decoder_outputs", True),
    ("native_temporal.target_suffixes", ["attn.to_q"]),
    ("native_temporal.rank", 8),
    ("model.custom_temporal_modules", True),
    ("model.scene_context.cross_time_attention", True),
    ("model.widths", [64, 128, 256]),
    ("lora.normalize_mix_weights", True),
    ("lora.spatial_mutual_exclusion", True),
    ("data.detection.family", "yolo26"),
    ("data.detection.fallback_other_family", True),
    ("data.detection.allow_identity_embeddings", True),
    ("data.sampling.fullbody_include_head", False),
    ("data.sampling.scene_from", "hq_full_frame"),
    ("data.degradation.resample_each_epoch", True),
    ("data.degradation.recipe_order", ["downsample", "blur", "noise", "compression", "upsample"]),
    ("data.degradation.resolution.allow_superresolution_of_target", True),
    ("training.decoder_input_gradient", False),
    ("backend.require_trained_base_for_lora", False),
    ("cache.verify_checksums", False),
    ("cache.pin_active_manifests", 1),
    ("cache.prune_default", "delete"),
    ("inference.fps_target", 30),
    ("evaluation.native_codec_release.require_human_approval", False),
])
def test_fixed_contract_cannot_be_disabled(tmp_path, source, dotted, value):
    with pytest.raises(H3CEError) as error:
        load_config(write_change(tmp_path, source, dotted, value))
    assert error.value.code == "E_CONFIG"


@pytest.mark.parametrize("dotted,value", [
    ("codec_mode", "frozen"),
    ("data.learned_temporal", False),
    ("model.scene_context.kernel_thw", [3, 3, 3]),
    ("training.train_only_temporal_branch", True),
    ("data.detection.YuNet", {}),
    ("data.detection.family", "YuNet"),
    ("lora.routing.adapters", [{"nested": {"codec_mode": "old"}}]),
    ("schema_version", 1),
])
def test_legacy_fields_have_migration_error_at_any_depth(tmp_path, source, dotted, value):
    with pytest.raises(H3CEError) as error:
        load_config(write_change(tmp_path, source, dotted, value))
    assert error.value.code == "E_CONFIG_V1"


def test_weights_must_be_independent(tmp_path, source):
    with pytest.raises(H3CEError) as error:
        load_config(write_change(tmp_path, source, "data.detection.face.weights", source["data"]["detection"]["person"]["weights"]))
    assert error.value.code == "E_CONFIG"


@pytest.mark.parametrize("dotted,value", [
    ("paths.tmp", "dataset/raw/cache"),
    ("paths.tmp", "runs"),
    ("paths.exports", "runs/exports"),
    ("paths.runs", "."),
])
def test_raw_cache_runs_export_paths_cannot_overlap(tmp_path, source, dotted, value):
    with pytest.raises(H3CEError):
        load_config(write_change(tmp_path, source, dotted, value))


def test_arbitrary_finite_mix_weights_are_preserved(tmp_path, source):
    config = load_config(write_change(tmp_path, source, "lora.routing.coefficients.medium", {"fullbody": -2.5, "face": 7.0}))
    assert config.lora.routing.coefficients.medium.fullbody == -2.5
    assert config.lora.routing.coefficients.medium.face == 7.0


def test_root_is_invocation_cwd_not_config_parent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = ProjectConfig(schema_version=2)
    assert project_root(config, tmp_path / "configs" / "project.v2.yaml") == tmp_path
    config = ProjectConfig(schema_version=2, project={"root": "work"})
    assert project_root(config, "elsewhere/config.yaml") == tmp_path / "work"


def test_aligned_pair_mode_requires_manifest(tmp_path, source):
    with pytest.raises(H3CEError):
        load_config(write_change(tmp_path, source, "data.mode", "aligned_pairs"))


def test_duplicate_yaml_keys_are_not_silently_ignored(tmp_path):
    path = tmp_path / "duplicates.yaml"
    path.write_text("schema_version: 2\nproject:\n  seed: 42\n  seed: 99\n", encoding="utf-8")
    with pytest.raises(H3CEError) as error:
        load_config(path)
    assert error.value.code == "E_CONFIG"
    assert "duplicate mapping key" in str(error.value)


@pytest.mark.parametrize("content", ["", "[]", "schema_version: 2\ndata: [", "schema_version: true"])
def test_invalid_yaml_and_non_mapping_root_are_configuration_errors(tmp_path, content):
    path = tmp_path / "invalid.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(H3CEError) as error:
        load_config(path)
    assert error.value.code == "E_CONFIG"


def test_missing_file_is_configuration_error(tmp_path):
    with pytest.raises(H3CEError) as error:
        load_config(tmp_path / "missing.yaml")
    assert error.value.code == "E_CONFIG"
