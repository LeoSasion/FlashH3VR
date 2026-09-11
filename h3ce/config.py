"""Strict v2 configuration and reproducible, fully expanded YAML output.

Relative ``project.root`` is resolved against the invocation's current working
directory, including when the configuration file lives in ``configs/``. All
other project paths are relative to that resolved root. Moving a YAML file does
not silently change the project to which it applies.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    ValidationError,
    model_validator,
)
import yaml

from h3ce.errors import H3CEError


def _true(value: Any) -> bool:
    if value is not True:
        raise ValueError("the v2 contract requires true")
    return value


def _false(value: Any) -> bool:
    if value is not False:
        raise ValueError("the v2 contract requires false")
    return value


def _integer(value: Any) -> int:
    if type(value) is not int:
        raise ValueError("an integer is required (booleans are not integers)")
    return value


AlwaysTrue = Annotated[Literal[True], BeforeValidator(_true)]
AlwaysFalse = Annotated[Literal[False], BeforeValidator(_false)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
Seed = Annotated[int, Field(strict=True, ge=0, le=2**64 - 1)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
PositiveFloat = Annotated[FiniteFloat, Field(gt=0)]
NonnegativeFloat = Annotated[FiniteFloat, Field(ge=0)]
Probability = Annotated[FiniteFloat, Field(ge=0, le=1)]
Ratio = Annotated[FiniteFloat, Field(gt=0, le=1)]
NonemptyString = Annotated[str, Field(min_length=1)]
HW = Annotated[list[PositiveInt], Field(min_length=2, max_length=2)]
FloatPair = Annotated[list[NonnegativeFloat], Field(min_length=2, max_length=2)]
PositivePair = Annotated[list[PositiveFloat], Field(min_length=2, max_length=2)]
IntPair = Annotated[list[PositiveInt], Field(min_length=2, max_length=2)]

_LEGACY_KEYS = {
    "codec_mode",
    "learned_temporal",
    "train_only_temporal_branch",
    "kernel_thw",
}


def _reject_legacy(value: Any, location: str = "") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}" if location else str(key)
            normalized = str(key).casefold()
            if normalized in _LEGACY_KEYS or "yunet" in normalized:
                raise H3CEError(
                    "E_CONFIG_V1",
                    f"Legacy field {child_location!r} is unsupported; migrate to v2 explicitly.",
                )
            if isinstance(child, str) and "yunet" in child.casefold():
                raise H3CEError(
                    "E_CONFIG_V1",
                    f"YuNet configuration at {child_location!r} is unsupported in v2.",
                )
            _reject_legacy(child, child_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_legacy(child, f"{location}[{index}]")


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False, validate_default=True
    )


class ProjectSettings(StrictModel):
    name: NonemptyString = "single_character"
    root: NonemptyString = "."
    seed: Seed = 42
    budget_seconds: PositiveInt = 172800
    vram_target_gib: PositiveFloat = 24
    offline_by_default: AlwaysTrue = True


class PathsConfig(StrictModel):
    raw: NonemptyString = "dataset/raw"
    pairs_manifest: NonemptyString | None = None
    tmp: NonemptyString = "tmp"
    models: NonemptyString = "models"
    runs: NonemptyString = "runs"
    exports: NonemptyString = "exports"
    components_lock: NonemptyString = "components.lock.json"


class BackendConfig(StrictModel):
    name: Literal["h3_vae_small_refiner"] = "h3_vae_small_refiner"
    require_trained_base_for_lora: AlwaysTrue = True
    native_h3_dit_lora_compatible: AlwaysFalse = False
    require_component_lock: AlwaysTrue = True


class VAEConfig(StrictModel):
    implementation: Literal["aitoolkit_h3_pinned", "aitoolkit_h3_int8_dequant_fp16", "aitoolkit_h3_int8_mm"] = "aitoolkit_h3_pinned"
    weights: NonemptyString = "models/minimax_h3_video_vae_fp16.safetensors"
    external_range: Literal["zero_one"] = "zero_one"
    latent_normalization: Literal["bridge_owned"] = "bridge_owned"
    posterior: Literal["mean"] = "mean"
    encoder_trainable: AlwaysFalse = False
    precision_policy: Literal["upstream_verified"] = "upstream_verified"
    tiling: Literal["upstream_locked"] = "upstream_locked"
    input_dispatch: Literal["auto"] = "auto"
    preserve_pts: AlwaysTrue = True
    frame_policy: Literal["h3_17n_plus_5"] = "h3_17n_plus_5"
    tail_padding: Literal["repeat_last"] = "repeat_last"
    clamp_before_loss: AlwaysFalse = False


class NativeLosses(StrictModel):
    reconstruction: PositiveFloat = 1.0
    motion_reconstruction: NonnegativeFloat = 0.1
    teacher_anchor: NonnegativeFloat = 0.05


class NativeTemporalConfig(StrictModel):
    mode: Literal["frozen", "finetune"] = "frozen"
    method: Literal["decoder_attention_lora"] = "decoder_attention_lora"
    decoder_adapter: NonemptyString | None = None
    last_n_blocks: PositiveInt = 4
    target_suffixes: list[Literal["attn.to_out"]] = Field(
        default_factory=lambda: ["attn.to_out"], min_length=1, max_length=1
    )
    rank: Annotated[Literal[4], BeforeValidator(_integer)] = 4
    alpha: PositiveFloat = 4
    learning_rate: PositiveFloat = 0.00001
    max_steps: PositiveInt = 500
    require_real_video: AlwaysTrue = True
    train_clip_frames: PositiveInt = 5
    promote_clip_frames: PositiveInt = 22
    image_replay_fraction: Probability = 0.25
    checkpoint_decoder: StrictBool = True
    freeze_refiner_and_character_loras: AlwaysTrue = True
    cache_trainable_decoder_outputs: AlwaysFalse = False
    losses: NativeLosses = Field(default_factory=NativeLosses)

    @model_validator(mode="after")
    def legal_video_lengths(self) -> NativeTemporalConfig:
        for name in ("train_clip_frames", "promote_clip_frames"):
            count = getattr(self, name)
            if count < 5 or (count - 5) % 17:
                raise ValueError(f"{name} must be a native video length 17n+5")
        if self.promote_clip_frames < self.train_clip_frames:
            raise ValueError("promote_clip_frames must not be shorter than train_clip_frames")
        if self.image_replay_fraction >= 1:
            raise ValueError("image_replay_fraction must leave steps for real video")
        return self


class SplitConfig(StrictModel):
    unit: Literal["source_group"] = "source_group"
    validation_fraction: Annotated[FiniteFloat, Field(gt=0, lt=1)] = 0.15
    seed: Seed = 42


class PersonDetectorConfig(StrictModel):
    weights: NonemptyString = "models/yolo11s.pt"
    class_name: Literal["person"] = "person"
    imgsz: PositiveInt = 960
    confidence: Probability = 0.30
    nms_iou: Probability = 0.50


class FaceDetectorConfig(StrictModel):
    weights: NonemptyString = "models/yolov11m-face.pt"
    class_name: Literal["face"] = "face"
    imgsz: PositiveInt = 960
    confidence: Probability = 0.25
    nms_iou: Probability = 0.50


class DetectionConfig(StrictModel):
    family: Literal["yolo11"] = "yolo11"
    person: PersonDetectorConfig = Field(default_factory=PersonDetectorConfig)
    face: FaceDetectorConfig = Field(default_factory=FaceDetectorConfig)
    fallback_other_family: AlwaysFalse = False
    tracking: Literal["iou_geometry_only"] = "iou_geometry_only"
    missing_face: Literal["keep_fullbody_skip_face"] = "keep_fullbody_skip_face"
    ambiguous_person: Literal["quarantine"] = "quarantine"
    allow_identity_embeddings: AlwaysFalse = False

    @model_validator(mode="after")
    def independent_weights(self) -> DetectionConfig:
        if Path(self.person.weights).resolve() == Path(self.face.weights).resolve():
            raise ValueError("person and face require independent dedicated weights")
        return self


class SamplingConfig(StrictModel):
    profile: Literal["debug", "balanced"] = "debug"
    canvas_multiple: Annotated[Literal[32], BeforeValidator(_integer)] = 32
    debug_buckets_hw: list[HW] = Field(
        default_factory=lambda: [[256, 256], [384, 256], [256, 384]], min_length=1
    )
    balanced_buckets_hw: list[HW] = Field(
        default_factory=lambda: [[512, 512], [768, 512], [512, 768]], min_length=1
    )
    fullbody_context_fraction: NonnegativeFloat = 0.25
    face_head_shoulders_expand_xy: PositivePair = Field(default_factory=lambda: [1.5, 2.0])
    min_face_hq_pixels: PositiveInt = 64
    video_crop: Literal["fixed_clip_union"] = "fixed_clip_union"
    fullbody_include_head: AlwaysTrue = True
    fullbody_wide_probability: Probability = 0.70
    face_close_probability: Probability = 0.70
    stretch: AlwaysFalse = False
    scene_from: Literal["degraded_full_frame_only"] = "degraded_full_frame_only"

    @model_validator(mode="after")
    def bucket_alignment(self) -> SamplingConfig:
        for name in ("debug_buckets_hw", "balanced_buckets_hw"):
            for hw in getattr(self, name):
                if any(side % self.canvas_multiple for side in hw):
                    raise ValueError(f"{name} sides must be multiples of canvas_multiple")
        if any(factor < 1 for factor in self.face_head_shoulders_expand_xy):
            raise ValueError("face/head/shoulders expansion factors must be >= 1")
        return self


class ResolutionConfig(StrictModel):
    mode: Literal["ratio", "short_edge"] = "ratio"
    ratio_range: Annotated[list[Ratio], Field(min_length=2, max_length=2)] = Field(
        default_factory=lambda: [0.25, 0.75]
    )
    short_edge_range: IntPair = Field(default_factory=lambda: [128, 512])
    distribution: Literal["uniform"] = "uniform"
    min_lr_side: PositiveInt = 2
    downsample_filter: Literal["area"] = "area"
    upsample_filter: Literal["bicubic"] = "bicubic"
    allow_superresolution_of_target: AlwaysFalse = False

    @model_validator(mode="after")
    def ordered_ranges(self) -> ResolutionConfig:
        for name in ("ratio_range", "short_edge_range"):
            low, high = getattr(self, name)
            if low > high:
                raise ValueError(f"{name} requires minimum <= maximum")
        return self


class BlurConfig(StrictModel):
    probability: Probability = 0.80
    kind: Literal["gaussian_isotropic"] = "gaussian_isotropic"
    sigma_reference_px_range: FloatPair = Field(default_factory=lambda: [0.0, 1.5])
    reference_short_edge: PositiveInt = 512
    truncate: PositiveFloat = 3.0
    padding: Literal["reflect"] = "reflect"

    @model_validator(mode="after")
    def ordered_range(self) -> BlurConfig:
        if self.sigma_reference_px_range[0] > self.sigma_reference_px_range[1]:
            raise ValueError("sigma_reference_px_range requires minimum <= maximum")
        return self


class NoiseConfig(StrictModel):
    probability: Probability = 0.0
    sigma_255_range: FloatPair = Field(default_factory=lambda: [0.0, 3.0])
    realization: Literal["independent_per_frame"] = "independent_per_frame"

    @model_validator(mode="after")
    def ordered_range(self) -> NoiseConfig:
        if self.sigma_255_range[0] > self.sigma_255_range[1]:
            raise ValueError("sigma_255_range requires minimum <= maximum")
        return self


class CompressionConfig(StrictModel):
    kind: Literal["none", "jpeg", "h264"] = "none"
    probability: Probability = 0.0
    jpeg_quality_range: Annotated[
        list[Annotated[int, Field(strict=True, ge=1, le=100)]],
        Field(min_length=2, max_length=2),
    ] = Field(default_factory=lambda: [75, 95])
    h264_crf_range: Annotated[
        list[Annotated[int, Field(strict=True, ge=0, le=51)]],
        Field(min_length=2, max_length=2),
    ] = Field(default_factory=lambda: [16, 24])

    @model_validator(mode="after")
    def ordered_ranges(self) -> CompressionConfig:
        for name in ("jpeg_quality_range", "h264_crf_range"):
            low, high = getattr(self, name)
            if low > high:
                raise ValueError(f"{name} requires minimum <= maximum")
        if self.kind == "none" and self.probability != 0:
            raise ValueError("compression probability must be 0 when kind=none")
        return self


class DegradationConfig(StrictModel):
    enabled: StrictBool = True
    version: Literal["degrade_v2"] = "degrade_v2"
    variants_per_image: PositiveInt = 4
    variants_per_clip: PositiveInt = 2
    include_clean_pair: StrictBool = True
    seed: Seed = 42
    materialize: StrictBool = True
    resample_each_epoch: AlwaysFalse = False
    recipe_order: list[str] = Field(
        default_factory=lambda: ["blur", "downsample", "noise", "compression", "upsample"]
    )
    video_parameter_policy: Literal["constant_per_clip"] = "constant_per_clip"
    resolution: ResolutionConfig = Field(default_factory=ResolutionConfig)
    blur: BlurConfig = Field(default_factory=BlurConfig)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    max_resample_attempts: PositiveInt = 32

    @model_validator(mode="after")
    def fixed_order(self) -> DegradationConfig:
        if self.recipe_order != ["blur", "downsample", "noise", "compression", "upsample"]:
            raise ValueError("v2 requires blur -> downsample -> noise -> compression -> upsample")
        return self


class DataConfig(StrictModel):
    mode: Literal["hq_self_degrade", "aligned_pairs"] = "hq_self_degrade"
    task: Literal["restoration", "harmonization"] = "restoration"
    auto_prepare: StrictBool = True
    max_independent_frames: PositiveInt = 50000
    working_long_edge_max: PositiveInt = 2048
    color: Literal["sdr_srgb"] = "sdr_srgb"
    hdr_policy: Literal["reject_until_explicit_transform"] = "reject_until_explicit_transform"
    split: SplitConfig = Field(default_factory=SplitConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    degradation: DegradationConfig = Field(default_factory=DegradationConfig)


class CacheConfig(StrictModel):
    content_addressed: AlwaysTrue = True
    root_marker: Literal[".h3ce-cache-root"] = ".h3ce-cache-root"
    atomic_writes: AlwaysTrue = True
    verify_checksums: AlwaysTrue = True
    pin_active_manifests: AlwaysTrue = True
    prune_default: Literal["dry_run"] = "dry_run"
    latent_storage_dtype: Literal["float32"] = "float32"
    quota_gib: PositiveFloat | None = None
    cache_encoder_latents: StrictBool = True
    decoder_cache_policy: Literal["frozen_effective_hash_only"] = "frozen_effective_hash_only"


class SceneContextConfig(StrictModel):
    longest_edge: Annotated[Literal[256], BeforeValidator(_integer)] = 256
    grid_hw: HW = Field(default_factory=lambda: [8, 8])
    width: Annotated[Literal[256], BeforeValidator(_integer)] = 256
    heads: Annotated[Literal[8], BeforeValidator(_integer)] = 8
    context_per_latent: Literal["nearest_uniform_source_frame"] = "nearest_uniform_source_frame"
    cross_time_attention: AlwaysFalse = False
    include_crop_coordinates: AlwaysTrue = True

    @model_validator(mode="after")
    def fixed_grid(self) -> SceneContextConfig:
        if self.grid_hw != [8, 8]:
            raise ValueError("h3ce_spatial_refiner_v2 requires an 8x8 scene grid")
        return self


class OutputConfig(StrictModel):
    mode: Literal["decoded_delta"] = "decoded_delta"
    strength: FiniteFloat = 1.0
    coarse_box_feather: AlwaysTrue = True


class ModelConfig(StrictModel):
    architecture_id: Literal["h3ce_spatial_refiner_v2"] = "h3ce_spatial_refiner_v2"
    widths: list[PositiveInt] = Field(default_factory=lambda: [128, 192, 256])
    encoder_blocks: list[PositiveInt] = Field(default_factory=lambda: [2, 2])
    bottleneck_blocks: Annotated[Literal[4], BeforeValidator(_integer)] = 4
    decoder_blocks: list[PositiveInt] = Field(default_factory=lambda: [2, 2])
    ffn_multiplier: Annotated[Literal[2], BeforeValidator(_integer)] = 2
    zero_init_output: AlwaysTrue = True
    custom_temporal_modules: AlwaysFalse = False
    scene_context: SceneContextConfig = Field(default_factory=SceneContextConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def architecture_contract(self) -> ModelConfig:
        expected = {"widths": [128, 192, 256], "encoder_blocks": [2, 2], "decoder_blocks": [2, 2]}
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"h3ce_spatial_refiner_v2 requires {name}={value}")
        return self


class MixCoefficients(StrictModel):
    # Arbitrary finite weights are deliberate: never clamp or normalize mixtures.
    fullbody: FiniteFloat
    face: FiniteFloat


class RoutingCoefficients(StrictModel):
    close: MixCoefficients = Field(default_factory=lambda: MixCoefficients(fullbody=0, face=1))
    medium: MixCoefficients = Field(default_factory=lambda: MixCoefficients(fullbody=0.7, face=0.6))
    far: MixCoefficients = Field(default_factory=lambda: MixCoefficients(fullbody=1, face=0))


class RoutingConfig(StrictModel):
    mode: Literal["manual", "auto"] = "manual"
    # Same syntax as CLI --lora: alias:finite_weight, preserving arbitrary ranks/weights.
    adapters: list[NonemptyString] = Field(default_factory=list)
    coefficients: RoutingCoefficients = Field(default_factory=RoutingCoefficients)
    update_at_chunk_boundary: AlwaysTrue = True


class LoraConfig(StrictModel):
    train_mode: Literal["fullbody", "face"] = "fullbody"
    rank: PositiveInt = 8
    alpha: PositiveFloat = 8
    dropout: Probability = 0.0
    target_suffixes: list[str] = Field(default_factory=lambda: ["ffn.in_proj", "ffn.out_proj"])
    spatial_mutual_exclusion: AlwaysFalse = False
    normalize_mix_weights: AlwaysFalse = False
    prompt_semantics: Literal["adapter_alias_only"] = "adapter_alias_only"
    require_base_match: AlwaysTrue = True
    require_decoder_validation: AlwaysTrue = True
    routing: RoutingConfig = Field(default_factory=RoutingConfig)

    @model_validator(mode="after")
    def exact_targets(self) -> LoraConfig:
        if len(self.target_suffixes) != 2 or set(self.target_suffixes) != {"ffn.in_proj", "ffn.out_proj"}:
            raise ValueError("character LoRA targets must be ffn.in_proj and ffn.out_proj exactly once")
        return self


class StageConfig(StrictModel):
    lr: PositiveFloat
    max_steps: PositiveInt


class CharacterStageConfig(StageConfig):
    review_at_step: PositiveInt = 300

    @model_validator(mode="after")
    def review_within_training(self) -> CharacterStageConfig:
        if self.review_at_step > self.max_steps:
            raise ValueError("review_at_step must be <= max_steps")
        return self


class StagesConfig(StrictModel):
    bootstrap_latent: StageConfig = Field(default_factory=lambda: StageConfig(lr=0.0002, max_steps=2000))
    bootstrap_pixel: StageConfig = Field(default_factory=lambda: StageConfig(lr=0.00005, max_steps=500))
    character_lora: CharacterStageConfig = Field(
        default_factory=lambda: CharacterStageConfig(lr=0.0001, max_steps=1000, review_at_step=300)
    )


class ApplicationLosses(StrictModel):
    rgb: PositiveFloat = 1.0
    latent: NonnegativeFloat = 0.1
    perceptual: NonnegativeFloat = 0.05
    lighting_target: NonnegativeFloat = 0.2
    charbonnier_epsilon: PositiveFloat = 0.001
    region_normalization: Literal["weight_sum"] = "weight_sum"


class TrainingConfig(StrictModel):
    microbatch: PositiveInt = 1
    gradient_accumulation: PositiveInt = 4
    optimizer: Literal["adamw"] = "adamw"
    weight_decay: NonnegativeFloat = 0.0
    refiner_autocast: Literal["bfloat16_if_supported"] = "bfloat16_if_supported"
    gradient_clip_norm: PositiveFloat = 1.0
    preview_every_steps: PositiveInt = 100
    checkpoint_every_steps: PositiveInt = 200
    decoder_input_gradient: AlwaysTrue = True
    stop_on_budget_exhaustion: AlwaysTrue = True
    stages: StagesConfig = Field(default_factory=StagesConfig)
    losses: ApplicationLosses = Field(default_factory=ApplicationLosses)


class InferenceConfig(StrictModel):
    chunk_frames: PositiveInt = 22
    overlap_frames: NonnegativeInt = 5
    fps_target: None = None
    preserve_frame_count: AlwaysTrue = True
    preserve_audio_timeline: AlwaysTrue = True
    reset_on_cut: AlwaysTrue = True
    performance_report: StrictBool = True

    @model_validator(mode="after")
    def overlap_fits(self) -> InferenceConfig:
        if self.chunk_frames < 5 or (self.chunk_frames - 5) % 17:
            raise ValueError("chunk_frames must be a native video length 17n+5")
        if self.overlap_frames >= self.chunk_frames:
            raise ValueError("overlap_frames must be smaller than chunk_frames")
        return self


class NativeCodecReleaseConfig(StrictModel):
    max_relative_rgb_regression: NonnegativeFloat = 0.05
    max_absolute_lpips_regression: NonnegativeFloat = 0.02
    min_relative_motion_error_improvement: Probability = 0.02
    require_human_approval: AlwaysTrue = True
    missing_required_metric: Literal["block_release"] = "block_release"


class EvaluationConfig(StrictModel):
    require_independent_sources: AlwaysTrue = True
    native_codec_release: NativeCodecReleaseConfig = Field(default_factory=NativeCodecReleaseConfig)


class ProjectConfig(StrictModel):
    schema_version: Annotated[Literal[2], BeforeValidator(_integer)]
    project: ProjectSettings = Field(default_factory=ProjectSettings)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    vae: VAEConfig = Field(default_factory=VAEConfig)
    native_temporal: NativeTemporalConfig = Field(default_factory=NativeTemporalConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="before")
    @classmethod
    def reject_v1(cls, value: Any) -> Any:
        _reject_legacy(value)
        if isinstance(value, dict) and type(value.get("schema_version")) is int and value["schema_version"] == 1:
            raise H3CEError("E_CONFIG_V1", "schema_version=1 requires explicit migration to v2.")
        return value

    @model_validator(mode="after")
    def cross_section_contract(self) -> ProjectConfig:
        if self.data.mode == "aligned_pairs" and not self.paths.pairs_manifest:
            raise ValueError("aligned_pairs requires paths.pairs_manifest with aligned X/Y geometry")
        root = Path(self.project.root).expanduser().resolve()
        resolved = {name: (root / getattr(self.paths, name)).resolve() for name in ("raw", "tmp", "runs", "exports")}
        items = list(resolved.items())
        for index, (left_name, left) in enumerate(items):
            for right_name, right in items[index + 1 :]:
                if left == right or left in right.parents or right in left.parents:
                    raise ValueError(f"paths.{left_name} and paths.{right_name} must be separate non-nested directories")
        return self


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys rather than silently replacing user values."""


def _unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    loader.flatten_mapping(node)
    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(None, None, "mapping key must be scalar", key_node.start_mark) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(None, None, f"duplicate mapping key: {key!r}", key_node.start_mark)
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def load_config(path: str | Path) -> ProjectConfig:
    """Load strict YAML; diagnostics use E_CONFIG or E_CONFIG_V1."""
    try:
        with Path(path).open("r", encoding="utf-8-sig") as stream:
            value = yaml.load(stream, Loader=_UniqueKeyLoader)
        return ProjectConfig.model_validate(value)
    except H3CEError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError, ValueError) as exc:
        raise H3CEError("E_CONFIG", f"Cannot load configuration {path}: {exc}") from exc


def project_root(config: ProjectConfig, config_path: str | Path | None = None) -> Path:
    """Resolve root against CWD; ``config_path`` is informational, never a base."""
    return Path(config.project.root).expanduser().resolve()


def write_resolved(config: ProjectConfig, path: str | Path) -> Path:
    """Atomically write every effective configuration field as UTF-8 YAML."""
    destination = Path(path)
    temporary: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".partial", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        return destination
    except (OSError, ValueError) as exc:
        raise H3CEError("E_CONFIG", f"Cannot write resolved configuration {path}: {exc}") from exc
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
