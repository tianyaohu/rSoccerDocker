#!/usr/bin/env python3
"""
Render videos for all experiment checkpoints that don't already have a video.

Usage:
  python scripts/render_videos.py
  python scripts/render_videos.py --deterministic
  python scripts/render_videos.py --force
"""
import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

with console.status("[bold cyan]Loading libraries..."):
    from render_core import (
        ALGO_MAP, collect_run_dirs, find_checkpoints,
        get_missing_videos, infer_env_algo, render_checkpoint, run_label,
    )


def main():
    ap = argparse.ArgumentParser(description="Render videos for experiment checkpoints.")
    ap.add_argument("--root", default=".", help="repo root (default: .)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-render even if video exists")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    experiments_root = root / "experiments"
    if not experiments_root.exists():
        console.print(f"[red]✗[/] experiments/ directory not found: {experiments_root}")
        raise SystemExit(1)

    run_dirs = collect_run_dirs(experiments_root)
    if not run_dirs:
        console.print(f"[red]✗[/] No run folders under: {experiments_root}")
        raise SystemExit(1)

    # ── Scan & plan ───────────────────────────────────────────────────────────
    console.rule("[bold green]Scanning runs")

    plan = []
    skipped = []

    for run_dir in run_dirs:
        all_zips = find_checkpoints(run_dir)
        if not all_zips:
            skipped.append((run_dir, "no checkpoints found"))
            continue

        env_id, algo = infer_env_algo(run_dir)
        if not env_id or not algo or algo not in ALGO_MAP:
            skipped.append((run_dir, f"can't infer env/algo (got {env_id}/{algo})"))
            continue

        missing = get_missing_videos(run_dir, all_zips, force=args.force)
        if not missing:
            skipped.append((run_dir, f"all {len(all_zips)} video(s) up to date"))
            continue

        plan.append((run_dir, env_id, algo, missing))

    for run_dir, env_id, algo, zips in plan:
        console.print(f"  [green]▶[/] {run_label(run_dir)}  ({env_id} / {algo})  — {len(zips)} new video(s)")
    for run_dir, reason in skipped:
        console.print(f"  [dim]⏭ {run_label(run_dir)}  — {reason}[/]")

    if not plan:
        console.print("\n[yellow]Nothing to render.[/] Use --force to re-render all.")
        return

    total_ckpts = sum(len(z) for _, _, _, z in plan)
    console.print(f"\n[bold]Rendering {total_ckpts} video(s) across {len(plan)} run(s)...[/]\n")

    # ── Render ────────────────────────────────────────────────────────────────
    results = []

    def on_progress(step):
        console.print(f"        step [cyan]{step}[/]")

    for run_idx, (run_dir, env_id, algo, zips) in enumerate(plan, 1):
        console.rule(f"[bold cyan]Run {run_idx}/{len(plan)}: {run_label(run_dir)}")
        video_base = run_dir / "videos"

        for ckpt_idx, zip_path in enumerate(zips, 1):
            out_path = video_base / f"{zip_path.stem}.mp4"
            console.print(f"    [{ckpt_idx}/{len(zips)}] {zip_path.stem}")

            try:
                stats = render_checkpoint(
                    zip_path, env_id, algo, out_path,
                    fps=args.fps, deterministic=args.deterministic,
                    on_progress=on_progress,
                )
                console.print(
                    f"        [green]✓[/] {stats['steps']} steps  "
                    f"{stats['duration_sec']:.1f}s  "
                    f"{stats['size_mb']:.1f} MB"
                )
                results.append({"run": run_label(run_dir), "checkpoint": zip_path.stem, "status": "✓", **stats})
            except Exception as e:
                console.print(f"        [red]✗ ERROR:[/] {e}")
                results.append({"run": run_label(run_dir), "checkpoint": zip_path.stem, "status": "✗",
                                "steps": 0, "duration_sec": 0, "size_mb": 0})

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold green]Summary")

    table = Table(box=box.ROUNDED, show_lines=False)
    table.add_column("", justify="center", width=3)
    table.add_column("Run", style="cyan", max_width=40)
    table.add_column("Checkpoint", style="white")
    table.add_column("Steps", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Size", justify="right")

    for r in results:
        style = "green" if r["status"] == "✓" else "red"
        table.add_row(
            f"[{style}]{r['status']}[/]",
            r["run"], r["checkpoint"],
            str(r["steps"]),
            f"{r['duration_sec']:.1f}s",
            f"{r['size_mb']:.1f} MB",
        )

    console.print(table)

    ok = sum(1 for r in results if r["status"] == "✓")
    fail = sum(1 for r in results if r["status"] == "✗")
    total_mb = sum(r["size_mb"] for r in results)
    total_time = sum(r["duration_sec"] for r in results)

    console.print(f"\n  [bold green]{ok} video(s) rendered[/]", end="")
    if fail:
        console.print(f"  [bold red]{fail} failed[/]", end="")
    console.print(f"  [dim]({total_mb:.1f} MB total, {total_time:.1f}s elapsed)[/]\n")


if __name__ == "__main__":
    main()