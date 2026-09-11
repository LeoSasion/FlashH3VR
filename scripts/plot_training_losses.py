"""Plot recorded losses with effective weights; optionally observe an existing run.

This observer never starts, resumes, or changes training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

COMPONENTS = ("rgb", "latent", "perceptual", "lighting_target")


def read_records(path: Path, *, live=False):
    raw = path.read_bytes()
    # Only an unfinished final line may be ignored while another process writes.
    complete = raw if not live or raw.endswith(b"\n") else raw[:raw.rfind(b"\n") + 1]
    rows = [json.loads(line) for line in complete.splitlines() if line.strip()]
    if not rows:
        raise ValueError("No recorded optimizer steps")
    for row in rows:
        if (not isinstance(row, dict) or type(row.get("step")) is not int or row["step"] < 1
                or row.get("phase") not in {"overfit", "pixel", "latent", "mixed_probe"}):
            raise ValueError("Invalid optimizer-step record")
        names = ("total", "latent") if row["phase"] == "latent" else ("total", *COMPONENTS)
        for name in names:
            value = row.get("losses", {}).get(name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value):
                raise ValueError(f"Missing or nonfinite loss: {name}")
    if any(b["step"] != a["step"] + 1 or b["phase"] != a["phase"] for a, b in zip(rows, rows[1:])):
        raise ValueError("Optimizer steps contain gaps, duplicates, or mixed phases")
    return rows, complete


def contributions(rows, config):
    phase = rows[0]["phase"]
    weights = ({name: float(name == "latent") for name in COMPONENTS} if phase == "latent"
               else {name: float(config["training"]["losses"][name]) for name in COMPONENTS})
    if any(not np.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("Invalid effective loss weight")
    values = {name: np.array([r["losses"][name] for r in rows]) * weight if weight else np.zeros(len(rows))
              for name, weight in weights.items()}
    total = np.array([r["losses"]["total"] for r in rows])
    if not np.allclose(sum(values.values()), total, rtol=2e-6, atol=1e-9):
        raise ValueError("Logged total does not match the effective weighted objective")
    return weights, values


def plot(runs: list[Path], output: Path, window: int = 8, *, live=False):
    if window < 1 or not runs or output.suffix.lower() != ".png":
        raise ValueError("Use a positive window, at least one run, and a PNG output")
    loaded = []
    for run in runs:
        rows, raw = read_records(run / "metrics.jsonl", live=live)
        config_raw = (run / "resolved.yaml").read_bytes()
        weights, values = contributions(rows, yaml.safe_load(config_raw))
        loaded.append((run, rows, raw, config_raw, weights, values))
    figure, axes = plt.subplots(len(runs), 3, figsize=(16, 3.8 * len(runs)), squeeze=False)
    evidence = []
    for panels, (run, rows, raw, config_raw, weights, values) in zip(axes, loaded):
        steps = np.array([row["step"] for row in rows])
        average_window = min(window, len(rows))

        def line(axis, series, label):
            smooth = np.convolve(series, np.ones(average_window) / average_window, mode="valid")
            handle, = axis.plot(steps[average_window - 1:], smooth, lw=1.7, label=label)
            axis.plot(steps, series, color=handle.get_color(), alpha=.23, lw=.6)

        total = np.array([row["losses"]["total"] for row in rows])
        line(panels[0], total, f"Total; trailing {average_window}-step mean")
        panels[0].set_title(f"{rows[0]['phase']} | logged training loss")
        for name, weight in weights.items():
            if weight:
                line(panels[1], values[name], f"{name} x {weight:g}")
        panels[1].set_title("Effective contributions (zero weights excluded)")
        line(panels[2], np.array([row["losses"]["latent"] for row in rows]), "Raw latent loss")
        panels[2].set_title(f"Latent diagnostic | objective weight {weights['latent']:g}")
        for axis in panels:
            axis.set(xlabel="Optimizer step", ylabel="Loss")
            axis.grid(alpha=.18)
            axis.legend(fontsize=8)
        evidence.append({"run": str(run.resolve()), "metrics_snapshot_sha256": hashlib.sha256(raw).hexdigest(),
            "resolved_config_sha256": hashlib.sha256(config_raw).hexdigest(), "phase": rows[0]["phase"],
            "recorded_optimizer_steps": len(rows), "first_step": int(steps[0]), "last_step": int(steps[-1]),
            "effective_weights": weights, "total": {"all_logged_steps": float(total.mean()),
                "first_window": float(total[:average_window].mean()), "last_window": float(total[-average_window:].mean())}})
    figure.suptitle("H3CE | actual training records; shuffled views, no independent validation", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, .95))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.png")
    try:
        figure.savefig(temporary, dpi=150)
        temporary.replace(output)
    finally:
        plt.close(figure)
    result = {"runs": evidence, "window": window, "new_optimizer_steps": 0,
        "notes": ["Loss at step k is measured before optimizer update k, then logged after that update.",
                  "Adjacent points have different views; moving averages do not remove all composition effects.",
                  "The total uses effective coefficients; latent-only stages ignore application coefficients.",
                  "Raw latent diagnostics on a separate axis are not an extra contribution to total.",
                  "No missing steps or validation points are synthesized."]}
    metadata = output.with_suffix(".json")
    temporary_json = metadata.with_suffix(".tmp.json")
    temporary_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary_json.replace(metadata)
    print(json.dumps({"event": "loss_chart_updated", "path": str(output.resolve()),
                      "steps": [item["last_step"] for item in evidence]}), flush=True)
    return result


def watch(runs, output, window, interval):
    if not 1 <= interval <= 60:
        raise ValueError("Watch interval must be between 1 and 60 seconds")
    if any(not (run / "resolved.yaml").is_file() for run in runs):
        raise ValueError("Select existing run directories after training has initialized")
    previous = None
    while True:
        snapshots = [(run / "metrics.jsonl").read_bytes() if (run / "metrics.jsonl").exists() else b"" for run in runs]
        done = all(any((run / name).exists() for name in ("training_report.json", "training_interruption.json")) for run in runs)
        if snapshots != previous and all(b"\n" in raw for raw in snapshots):
            plot(runs, output, window, live=not done)
            previous = snapshots
        if done:
            return
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", dest="runs", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--watch", action="store_true", help="Observe existing runs only; never launches training")
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    if args.watch:
        watch(args.runs, args.output, args.window, args.interval)
    else:
        plot(args.runs, args.output, args.window)
