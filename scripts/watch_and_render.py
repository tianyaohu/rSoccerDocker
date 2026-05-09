#!/usr/bin/env python3
"""
Smart checkpoint watcher — selective rendering + W&B upload.

Instead of rendering every checkpoint, picks up to --max-videos evenly
spaced checkpoints per run (plus the final model).  Uploads each video
to the training run's W&B dashboard so you can see policy evolution inline.

Reads from the run directory:
  - command.txt        → --timesteps and --save-freq (for spacing math)
  - wandb_run_id.txt   → which W&B run to attach the video to

Usage:
  python scripts/watch_and_render.py
  python scripts/watch_and_render.py --max-videos 10 --interval 30
  python scripts/watch_and_render.py --deterministic

Press Ctrl+C to stop.
"""
import re
import time
import signal
import argparse
from pathlib import Path
from datetime import datetime

import wandb
from rich.console import Console

console = Console()

from render_core import (
    ALGO_MAP, collect_run_dirs, find_checkpoints,
    infer_env_algo, render_checkpoint, run_label,
)

rendered_set: set[str] = set()
running = True


def handle_sigint(sig, frame):
    global running
    running = False
    console.print("\n[yellow]Shutting down watcher...[/]")


signal.signal(signal.SIGINT, handle_sigint)


# ── Helpers ───────────────────────────────────────────────────

def parse_command_txt(run_dir: Path) -> dict:
    """Extract --timesteps and --save-freq from command.txt."""
    cmd_file = run_dir / "command.txt"
    if not cmd_file.exists():
        return {}
    txt = cmd_file.read_text(errors="ignore")
    result = {}
    for pattern, key in [
        (r"--timesteps\s+(\d+)", "timesteps"),
        (r"--save-freq\s+(\d+)", "save_freq"),
    ]:
        m = re.search(pattern, txt)
        if m:
            result[key] = int(m.group(1))
    return result


def checkpoint_step(zip_path: Path) -> int | None:
    """Extract step number from filename like sac_SSLDribbling-v0_01000000.zip."""
    m = re.search(r"_(\d+)\.zip$", zip_path.name)
    return int(m.group(1)) if m else None


def is_final_checkpoint(zip_path: Path) -> bool:
    return "_final.zip" in zip_path.name


def compute_render_steps(total_steps: int, save_freq: int, max_videos: int) -> set[int]:
    """
    Pick up to max_videos evenly spaced checkpoint steps.

    If total checkpoints <= max_videos, renders all of them.
    Otherwise picks the checkpoint closest to each evenly spaced target.
    Always includes the last checkpoint.
    """
    all_steps = list(range(save_freq, total_steps + 1, save_freq))
    if len(all_steps) <= max_videos:
        return set(all_steps)

    selected = set()
    for i in range(1, max_videos + 1):
        target = total_steps * i / max_videos
        closest = min(all_steps, key=lambda s: abs(s - target))
        selected.add(closest)
    return selected


def should_render(zip_path: Path, render_steps: set[int] | None) -> bool:
    """Decide whether this checkpoint is worth rendering."""
    # Always render the final model
    if is_final_checkpoint(zip_path):
        return True
    # No timing info → render everything (fallback)
    if render_steps is None:
        return True
    step = checkpoint_step(zip_path)
    if step is None:
        return True
    return step in render_steps


# ── W&B upload ────────────────────────────────────────────────

def read_wandb_run_id(run_dir: Path) -> str | None:
    f = run_dir / "wandb_run_id.txt"
    return f.read_text().strip() if f.exists() else None


def upload_to_wandb(run_dir: Path, env_id: str, mp4_path: Path, step: int | None):
    """Attach a video to the training run's W&B dashboard."""
    run_id = read_wandb_run_id(run_dir)
    if not run_id:
        console.print("    [dim]no wandb_run_id.txt — skipping upload[/]")
        return

    project = env_id.lower().replace("-v0", "")

    try:
        wandb.init(
            id=run_id,
            project=project,
            entity="sfurs",
            resume="allow",
        )
        log_data = {"video": wandb.Video(str(mp4_path), fps=30, format="mp4")}
        if step is not None:
            wandb.log(log_data, step=step)
        else:
            wandb.log(log_data)
        wandb.finish()
        console.print(f"    [blue]↑ W&B[/] step={step}")
    except Exception as e:
        console.print(f"    [red]W&B upload failed:[/] {e}")
        try:
            wandb.finish()
        except Exception:
            pass


# ── Render-step cache ─────────────────────────────────────────

_render_steps_cache: dict[str, set[int] | None] = {}


def get_render_steps(run_dir: Path, max_videos: int) -> set[int] | None:
    """Compute (and cache) which steps to render for a run."""
    key = str(run_dir)
    if key in _render_steps_cache:
        return _render_steps_cache[key]

    params = parse_command_txt(run_dir)
    total = params.get("timesteps")
    freq = params.get("save_freq")

    if total and freq:
        steps = compute_render_steps(total, freq, max_videos)
        _render_steps_cache[key] = steps
        console.print(
            f"  [dim]{run_label(run_dir)}: "
            f"{len(steps)} videos planned from {total:,} steps "
            f"(every {total // max(len(steps), 1):,})[/]"
        )
    else:
        _render_steps_cache[key] = None
        console.print(f"  [dim]{run_label(run_dir)}: no timing info, rendering all[/]")

    return _render_steps_cache[key]


# ── Main scan loop ────────────────────────────────────────────

def scan_and_render(experiments_root: Path, fps: int, deterministic: bool,
                    max_videos: int) -> int:
    """One pass: scan all runs, render + upload selected checkpoints."""
    count = 0

    for run_dir in collect_run_dirs(experiments_root):
        zips = find_checkpoints(run_dir)
        if not zips:
            continue

        env_id, algo = infer_env_algo(run_dir)
        if not env_id or not algo or algo not in ALGO_MAP:
            continue

        render_steps = get_render_steps(run_dir, max_videos)
        video_dir = run_dir / "videos"
        existing = {p.stem for p in video_dir.glob("*.mp4")} if video_dir.exists() else set()

        for zip_path in zips:
            path_key = str(zip_path)
            if path_key in rendered_set:
                continue
            if zip_path.stem in existing:
                rendered_set.add(path_key)
                continue
            if not should_render(zip_path, render_steps):
                rendered_set.add(path_key)
                continue

            out_path = video_dir / f"{zip_path.stem}.mp4"
            step = checkpoint_step(zip_path)
            console.print(f"  [green]▶ RENDER[/] {run_label(run_dir)} / [bold]{zip_path.stem}[/]")

            try:
                stats = render_checkpoint(
                    zip_path, env_id, algo, out_path,
                    fps=fps, deterministic=deterministic,
                    on_progress=lambda s: console.print(f"    step [cyan]{s}[/]"),
                )
                console.print(
                    f"    [green]✓[/] {stats['steps']} steps  "
                    f"{stats['duration_sec']:.1f}s  "
                    f"{stats['size_mb']:.1f} MB"
                )
                upload_to_wandb(run_dir, env_id, out_path, step)
                count += 1
            except Exception as e:
                console.print(f"    [red]✗ ERROR:[/] {e}")

            rendered_set.add(path_key)

    return count


def main():
    ap = argparse.ArgumentParser(
        description="Smart checkpoint watcher — selective rendering + W&B upload.")
    ap.add_argument("--root", default=".", help="repo root")
    ap.add_argument("--interval", type=int, default=30, help="poll interval (seconds)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--max-videos", type=int, default=10,
                    help="max videos per run (evenly spaced + final)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    experiments_root = root / "experiments"

    console.rule("[bold cyan]Smart Checkpoint Watcher")
    console.print(f"  Watching    : {experiments_root}/")
    console.print(f"  Max videos  : {args.max_videos} per run (+ final)")
    console.print(f"  Interval    : {args.interval}s")
    console.print("  W&B upload  : enabled (reads wandb_run_id.txt)")
    console.print("  Ctrl+C to stop\n")

    total = 0
    while running:
        if experiments_root.exists():
            total += scan_and_render(experiments_root, args.fps, args.deterministic,
                                     args.max_videos)
        if not running:
            break
        now = datetime.now().strftime("%H:%M:%S")
        console.print(f"  [dim]{now}  watching… ({total} rendered)[/]", end="\r")
        for _ in range(args.interval * 2):
            if not running:
                break
            time.sleep(0.5)

    console.print(f"\n[bold]Watcher stopped. {total} video(s) rendered this session.[/]")


if __name__ == "__main__":
    main()