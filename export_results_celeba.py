"""
Export TensorBoard training results to ordinary files.

This script makes the training output visible without opening TensorBoard:
  - one PNG curve plot per run
  - latest generated/traversal images from TensorBoard image summaries
  - one CSV with scalar metrics per run
  - one Markdown summary pointing to the exported files

Usage:
    python export_results_celeba.py
    python export_results_celeba.py --log_dir logs/celeba_stage4_single_code_infogan --out_dir results/celeba_stage4_single_code_infogan_exported
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


SCALAR_TAGS = [
    "train/d_loss",
    "train/g_loss",
    "train/mi_disc",
    "train/mi_cont",
    "train/LI_disc",
]


def safe_name(text: str) -> str:
    text = text.replace("\\", "/")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "item"


def load_run(run_dir: Path) -> EventAccumulator | None:
    if not any(run_dir.glob("events.out.tfevents*")):
        return None
    accumulator = EventAccumulator(
        str(run_dir),
        size_guidance={"scalars": 0, "images": 0},
    )
    accumulator.Reload()
    return accumulator


def write_scalars_csv(run_dir: Path, accumulator: EventAccumulator, out_dir: Path) -> dict[str, dict[str, float]]:
    scalar_tags = [tag for tag in SCALAR_TAGS if tag in accumulator.Tags().get("scalars", [])]
    if not scalar_tags:
        return {}

    csv_path = out_dir / "scalars.csv"
    rows_by_step: dict[int, dict[str, float]] = {}
    summary: dict[str, dict[str, float]] = {}

    for tag in scalar_tags:
        values = accumulator.Scalars(tag)
        if not values:
            continue
        short = tag.split("/", 1)[-1]
        for event in values:
            rows_by_step.setdefault(event.step, {"step": event.step})[short] = event.value
        tail = values[-5:] if len(values) >= 5 else values
        summary[short] = {
            "count": len(values),
            "first_step": values[0].step,
            "last_step": values[-1].step,
            "first": values[0].value,
            "last": values[-1].value,
            "tail5_avg": sum(v.value for v in tail) / len(tail),
            "min": min(v.value for v in values),
            "max": max(v.value for v in values),
        }

    fieldnames = ["step"] + [tag.split("/", 1)[-1] for tag in scalar_tags]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in sorted(rows_by_step):
            writer.writerow(rows_by_step[step])

    return summary


def plot_scalars(accumulator: EventAccumulator, out_dir: Path, run_name: str) -> Path | None:
    scalar_tags = [tag for tag in SCALAR_TAGS if tag in accumulator.Tags().get("scalars", [])]
    if not scalar_tags:
        return None

    n_cols = 2
    n_rows = (len(scalar_tags) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 3.2 * n_rows), squeeze=False)
    fig.suptitle(run_name, fontsize=13)

    for ax in axes.ravel():
        ax.set_visible(False)

    for ax, tag in zip(axes.ravel(), scalar_tags):
        values = accumulator.Scalars(tag)
        steps = [v.step for v in values]
        vals = [v.value for v in values]
        short = tag.split("/", 1)[-1]

        ax.set_visible(True)
        ax.plot(steps, vals, linewidth=1.7)
        if short == "LI_disc":
            ax.axhline(2.302585, linestyle="--", linewidth=1.0, color="#888888", label="log(10)")
            ax.legend(fontsize=8)
        ax.set_title(short)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    plot_path = out_dir / "loss_curves.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def export_latest_images(accumulator: EventAccumulator, out_dir: Path) -> list[Path]:
    image_paths: list[Path] = []
    for tag in accumulator.Tags().get("images", []):
        images = accumulator.Images(tag)
        if not images:
            continue
        latest = images[-1]
        image_path = out_dir / f"{safe_name(tag)}_step{latest.step}.png"
        image_path.write_bytes(latest.encoded_image_string)
        image_paths.append(image_path)
    return image_paths


def format_float(value: float) -> str:
    return f"{value:.6g}"


def export(log_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(path for path in log_dir.iterdir() if path.is_dir())
    report_lines = [
        "# Exported Training Results",
        "",
        f"Source log directory: `{log_dir}`",
        "",
    ]

    exported_count = 0
    for run_dir in run_dirs:
        accumulator = load_run(run_dir)
        if accumulator is None:
            continue

        run_out = out_dir / safe_name(run_dir.name)
        run_out.mkdir(parents=True, exist_ok=True)
        scalar_summary = write_scalars_csv(run_dir, accumulator, run_out)
        curve_path = plot_scalars(accumulator, run_out, run_dir.name)
        image_paths = export_latest_images(accumulator, run_out)
        exported_count += 1

        report_lines.append(f"## {run_dir.name}")
        report_lines.append("")
        if curve_path is not None:
            report_lines.append(f"- Loss curves: `{curve_path}`")
        report_lines.append(f"- Scalars CSV: `{run_out / 'scalars.csv'}`")

        if scalar_summary:
            report_lines.append("")
            report_lines.append("| metric | steps | first | last | tail5 avg | min | max |")
            report_lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for metric, stats in scalar_summary.items():
                report_lines.append(
                    f"| {metric} | {int(stats['first_step'])}-{int(stats['last_step'])} "
                    f"| {format_float(stats['first'])} | {format_float(stats['last'])} "
                    f"| {format_float(stats['tail5_avg'])} | {format_float(stats['min'])} "
                    f"| {format_float(stats['max'])} |"
                )

        if image_paths:
            report_lines.append("")
            report_lines.append("Latest TensorBoard images:")
            for image_path in image_paths:
                report_lines.append(f"- `{image_path}`")

        report_lines.append("")

    if exported_count == 0:
        report_lines.append("No TensorBoard runs found.")

    report_path = out_dir / "summary.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CelebA TensorBoard logs to PNG/CSV/Markdown files.")
    parser.add_argument("--log_dir", default="logs/celeba_stage4_single_code_infogan", help="CelebA TensorBoard log directory.")
    parser.add_argument("--out_dir", default="results/celeba_stage4_single_code_infogan_exported", help="Directory for exported CelebA files.")
    args = parser.parse_args()

    report_path = export(Path(args.log_dir), Path(args.out_dir))
    print(f"Export complete: {report_path}")


if __name__ == "__main__":
    main()
