# Publication validation

Validated locally on 2026-09-12 in the isolated publication checkout, using the recorded Python 3.12 environment:

- **366 CPU tests passed**, with **19 subtests passed**; **1 real-model test was explicitly deselected**.
- The first collection attempt identified two missing ProRes test helper imports. The helpers and their project dependencies were included before the successful complete run.
- Built `flashh3vr-0.1.0-py3-none-any.whl` with dependency resolution disabled, installed it into a separate temporary target, and verified that the current file-inference module and chunk helper imported from that installation.
- The wheel import check did not construct or execute a model. It does not establish inference portability; supported setup is the editable source checkout with separately supplied assets.
- Existing numerical source parity checks for the MIT-derived H3 implementation passed. Git preserves source bytes to retain pinned upstream hashes on other platforms.
- No new real-model forward, GPU benchmark, training update or weight download was performed for publication.

The repository includes a GitHub Actions CPU workflow. Its remote result is separate from this recorded local validation.

The selected upstream NAFNet readme is preserved as a provenance input. Its relative links to upstream figures and files outside the selected source subset require the original upstream repository; no upstream demonstration media is bundled here.
