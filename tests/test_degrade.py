"""Real CPU pixel checks for degradation; these are not H3/GPU acceptance tests."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml

from h3ce.data.degrade import (
    _area_rgb, canonical_json, degrade_rgb, frame_noise_seed, plan_variants,
    processor_contract, recipe_core, recipe_hash, validate_degradation,
)
from h3ce.errors import H3CEError


CONFIG = yaml.safe_load((Path(__file__).parents[1] / "configs/project.v2.yaml").read_text(encoding="utf-8"))


class DegradationTests(unittest.TestCase):
    def setUp(self):
        self.d = copy.deepcopy(CONFIG["data"]["degradation"])
        self.pixels = np.random.default_rng(73).random((48, 64, 3), dtype=np.float32)

    def plans(self, kind="image", hw=(48, 64)):
        return plan_variants(self.d, "same-target", kind, hw)

    def assert_code(self, code, action):
        with self.assertRaises(H3CEError) as caught:
            action()
        self.assertEqual(caught.exception.code, code)

    def test_t03_four_degraded_plus_one_separate_clean_pair(self):
        plans = self.plans()
        self.assertEqual(len(plans), 5)
        self.assertEqual([p["variant_index"] for p in plans if not p["clean_pair"]], list(range(4)))
        self.assertEqual(sum(p["clean_pair"] for p in plans), 1)
        identity = degrade_rgb(self.pixels, plans[-1])
        np.testing.assert_array_equal(identity, self.pixels)
        self.assertFalse(np.shares_memory(identity, self.pixels))

    def test_t04_count_extension_preserves_old_seeds_ids_and_pixels(self):
        first = self.plans()
        first_pixels = [degrade_rgb(self.pixels, p) for p in first[:4]]
        self.d["variants_per_image"] = 6
        extended = self.plans()
        self.assertEqual(first[:4], extended[:4])
        self.assertEqual(first[-1], extended[-1])
        for expected, plan in zip(first_pixels, extended[:4]):
            np.testing.assert_array_equal(expected, degrade_rgb(self.pixels, plan))
        self.d["variants_per_image"] = 2
        self.assertEqual(self.plans(), extended[:2] + extended[-1:])

    def test_t04_workers_and_repeat_execution_preserve_pixels(self):
        plans = self.plans()[:4]
        expected = {p["variant_id"]: degrade_rgb(self.pixels, p) for p in plans}
        with ThreadPoolExecutor(max_workers=3) as workers:
            results = list(workers.map(lambda p: (p["variant_id"], degrade_rgb(self.pixels, p)), reversed(plans)))
        for key, pixels in results:
            np.testing.assert_array_equal(expected[key], pixels)

    def test_canonical_seed_and_variant_id_match_contract(self):
        plan = self.plans()[0]
        core_hash = hashlib.sha256(canonical_json(recipe_core(self.d))).hexdigest()
        expected_seed = int.from_bytes(hashlib.sha256(canonical_json(
            [self.d["seed"], "same-target", core_hash, 0])).digest()[:8], "big")
        self.assertEqual(plan["seed"], expected_seed)
        self.assertEqual(plan["variant_id"], hashlib.sha256(canonical_json(
            ["same-target", core_hash, self.d["seed"], 0])).hexdigest())

    def test_global_seed_overrides_degradation_seed(self):
        a = plan_variants(self.d, "same-target", "image", (48, 64), global_seed=96)
        self.d["seed"] = 97
        b = plan_variants(self.d, "same-target", "image", (48, 64), global_seed=96)
        self.assertEqual(a, b)
        self.assertNotEqual(a, self.plans())

    def test_inactive_and_quantity_fields_do_not_affect_recipe(self):
        original = recipe_hash(self.d)
        self.d["variants_per_image"] = 8
        self.d["variants_per_clip"] = 7
        self.d["include_clean_pair"] = False
        self.d["seed"] = 82
        self.d["materialize"] = False
        self.d["resolution"]["short_edge_range"] = [100, 999]
        self.d["noise"]["sigma_255_range"] = [4, 9]
        self.d["compression"]["jpeg_quality_range"] = [4, 9]
        self.d["compression"]["h264_crf_range"] = [1, 2]
        self.assertEqual(original, recipe_hash(self.d))
        self.d["resolution"]["ratio_range"] = [0.4, 0.8]
        self.assertNotEqual(original, recipe_hash(self.d))

    def test_t05_side_ratio_and_half_up_rounding(self):
        self.d["resolution"]["ratio_range"] = [0.25, 0.25]
        self.assertEqual(self.plans(hw=(768, 1024))[0]["lr_hw"], [192, 256])
        self.assertEqual(self.plans(hw=(10, 14))[0]["lr_hw"], [3, 4])

    def test_t05_blur_sigma_reference_scaling_and_kernel(self):
        self.d["blur"]["probability"] = 1.0
        self.d["blur"]["sigma_reference_px_range"] = [1.0, 1.0]
        blur = self.plans(hw=(1024, 1536))[0]["resolved_operations"]["blur"]
        self.assertEqual(blur["sigma_actual_px"], 2.0)
        self.assertEqual(blur["radius"], 6)
        self.assertEqual(blur["kernel"], 13)

    def test_t06_short_edge_intersection_and_no_superresolution(self):
        self.d["resolution"]["mode"] = "short_edge"
        self.d["resolution"]["short_edge_range"] = [48, 1200]
        self.assertTrue(all(p["lr_hw"] == [48, 64] for p in self.plans()))
        self.d["resolution"]["short_edge_range"] = [49, 1200]
        self.assert_code("E_DEGRADE_RANGE", self.plans)

    def test_t06_bad_ranges_counts_and_probabilities_are_rejected(self):
        invalids = [
            (("variants_per_image",), 0), (("variants_per_image",), True),
            (("variants_per_clip",), 1.5), (("max_resample_attempts",), 0),
            (("resolution", "ratio_range"), [0, 1]),
            (("resolution", "ratio_range"), [0.8, 0.7]),
            (("resolution", "ratio_range"), [0.2, 1.1]),
            (("resolution", "short_edge_range"), [1.5, 12]),
            (("blur", "sigma_reference_px_range"), [-1, 2]),
            (("blur", "sigma_reference_px_range"), [0, float("nan")]),
            (("blur", "probability"), 1.1), (("noise", "probability"), -0.1),
            (("compression", "probability"), float("inf")),
            (("compression", "jpeg_quality_range"), [70.5, 80]),
            (("compression", "h264_crf_range"), [0, 52]),
        ]
        for path, value in invalids:
            with self.subTest(path=path, value=value):
                invalid = copy.deepcopy(self.d)
                parent = invalid
                for key in path[:-1]:
                    parent = parent[key]
                parent[path[-1]] = value
                self.assert_code("E_CONFIG", lambda: validate_degradation(invalid))

    def test_resolution_below_minimum_is_not_silently_clamped(self):
        self.d["resolution"]["ratio_range"] = [0.01, 0.01]
        self.assert_code("E_DEGRADE_RANGE", self.plans)

    def test_invalid_reflect_kernel_is_explicit(self):
        self.d["blur"]["probability"] = 1
        self.d["blur"]["sigma_reference_px_range"] = [1000, 1000]
        self.assert_code("E_BLUR_KERNEL", self.plans)

    def test_t07_clip_parameters_constant_noise_independent_and_repeatable(self):
        self.d["resolution"]["ratio_range"] = [1, 1]
        self.d["blur"]["probability"] = 0
        self.d["noise"]["probability"] = 1
        self.d["noise"]["sigma_255_range"] = [8, 8]
        clip = np.full((5, 48, 64, 3), 0.5, dtype=np.float32)
        before = clip.copy()
        plans = self.plans(kind="video")
        self.assertEqual(len(plans), 3)
        output = degrade_rgb(clip, plans[0])
        self.assertEqual(output.shape, clip.shape)
        np.testing.assert_array_equal(output, degrade_rgb(clip, plans[0]))
        np.testing.assert_array_equal(clip, before)
        self.assertFalse(np.array_equal(output[0], output[1]))
        self.assertNotEqual(frame_noise_seed(plans[0]["seed"], 0), frame_noise_seed(plans[0]["seed"], 1))
        for frame in output:
            self.assertAlmostEqual(float(frame.std()), 8 / 255, delta=0.001)

    def test_video_identical_frames_remain_identical_when_noise_disabled(self):
        clip = np.repeat(self.pixels[None], 5, axis=0)
        output = degrade_rgb(clip, self.plans(kind="video")[0])
        for frame in output[1:]:
            np.testing.assert_array_equal(frame, output[0])

    def test_pixels_really_degrade_without_changing_ground_truth_or_plan(self):
        plan = self.plans()[0]
        plan_before, pixels_before = copy.deepcopy(plan), self.pixels.copy()
        output = degrade_rgb(self.pixels, plan)
        self.assertFalse(np.array_equal(self.pixels, output))
        self.assertLess(float(output.std()), float(self.pixels.std()))
        np.testing.assert_array_equal(self.pixels, pixels_before)
        self.assertEqual(plan, plan_before)
        self.assertEqual(output.dtype, np.float32)
        self.assertGreaterEqual(float(output.min()), 0)
        self.assertLessEqual(float(output.max()), 1)

    def test_area_downsample_matches_exact_pixel_area(self):
        image = np.repeat(np.arange(24, dtype=np.float32).reshape(4, 6, 1), 3, axis=2)
        expected = image.reshape(2, 2, 3, 2, 3).mean(axis=(1, 3))
        np.testing.assert_allclose(_area_rgb(image, (2, 3)), expected, atol=1e-6)
        noninteger = np.repeat(np.array([0, 1, 0], np.float32)[None, :, None], 3, axis=2)
        np.testing.assert_allclose(_area_rgb(noninteger, (1, 2)), 1 / 3, atol=1e-6)

    def test_gaussian_blur_is_real_and_precedes_downsampling(self):
        self.d["resolution"]["ratio_range"] = [1, 1]
        self.d["blur"]["probability"] = 1
        self.d["blur"]["reference_short_edge"] = 48
        self.d["blur"]["sigma_reference_px_range"] = [1, 1]
        impulse = np.zeros_like(self.pixels)
        impulse[24, 32] = 1
        output = degrade_rgb(impulse, self.plans()[0])
        self.assertGreater(float(output[24, 32, 0]), 0)
        self.assertLess(float(output[24, 32, 0]), 1)
        self.assertGreater(float(output[24, 31, 0]), 0)
        self.assertAlmostEqual(float(output[..., 0].sum()), 1, places=6)
        self.assertEqual(self.plans()[0]["resolved_operations"]["order"],
                         ["blur", "downsample", "noise", "compression", "upsample"])

    def test_jpeg_is_applied_deterministically(self):
        self.d["resolution"]["ratio_range"] = [1, 1]
        self.d["blur"]["probability"] = 0
        self.d["compression"].update(kind="jpeg", probability=1, jpeg_quality_range=[25, 25])
        plan = self.plans()[0]
        jpeg = degrade_rgb(self.pixels, plan)
        self.assertEqual(plan["resolved_operations"]["compression"]["quality"], 25)
        self.assertGreater(float(np.mean(np.abs(jpeg - self.pixels))), 0.02)
        np.testing.assert_array_equal(jpeg, degrade_rgb(self.pixels, plan))
        self.assertTrue(processor_contract()["jpeg_codec"])

    def test_h264_media_and_unimplemented_errors_are_explicit(self):
        self.d["compression"]["kind"] = "h264"
        self.assert_code("E_DEGRADE_MEDIA", self.plans)
        self.assert_code("E_NOT_IMPLEMENTED", lambda: self.plans(kind="video"))

    def test_disabled_degradation_respects_separate_clean_pair_flag(self):
        self.d["enabled"] = False
        self.assertEqual(len(self.plans()), 1)
        self.assertTrue(self.plans()[0]["clean_pair"])
        self.d["include_clean_pair"] = False
        self.assertEqual(self.plans(), [])

    def test_unknown_or_unsupported_fields_are_not_silent(self):
        self.d["noise"]["extra"] = True
        self.assert_code("E_CONFIG", self.plans)
        self.d["noise"].pop("extra")
        self.d["resolution"]["downsample_filter"] = "nearest"
        self.assert_code("E_NOT_IMPLEMENTED", self.plans)

    def test_input_range_and_geometry_are_checked(self):
        plan = self.plans()[0]
        self.assert_code("E_DEGRADE_INPUT", lambda: degrade_rgb(self.pixels.astype(np.float64), plan))
        self.assert_code("E_DEGRADE_INPUT", lambda: degrade_rgb(self.pixels + 1, plan))
        self.assert_code("E_DEGRADE_INPUT", lambda: degrade_rgb(self.pixels[:20], plan))
        self.assert_code("E_DEGRADE_MEDIA", lambda: degrade_rgb(np.repeat(self.pixels[None], 2, axis=0), plan))
        wrong = copy.deepcopy(plan)
        wrong["resolved_operations"]["processor"]["pillow"] = "different-version"
        self.assert_code("E_DEGRADE_PROCESSOR", lambda: degrade_rgb(self.pixels, wrong))

    def test_full_project_mapping_works(self):
        self.assertEqual(recipe_hash(CONFIG), recipe_hash(self.d))

    def test_full_project_and_degradation_pydantic_models_work(self):
        from h3ce.config import ProjectConfig
        config = ProjectConfig.model_validate(CONFIG)
        self.assertEqual(recipe_hash(config), recipe_hash(self.d))
        self.assertEqual(recipe_hash(config.data.degradation), recipe_hash(self.d))
        self.assertEqual(plan_variants(config, "same-target", "image", (48, 64)), self.plans())


if __name__ == "__main__":
    unittest.main(verbosity=2)
