"""
Shared rendering logic for render_videos.py and watch_and_render.py.

Change rendering settings here and both scripts pick them up.
"""
import re
import time
from pathlib import Path

import gymnasium as gym
import rsoccer_gym  # noqa: F401  registers envs
import imageio
from stable_baselines3 import PPO, SAC, DDPG, TD3

ALGO_MAP = {"ppo": PPO, "sac": SAC, "ddpg": DDPG, "td3": TD3}


# ── Inference helpers ─────────────────────────────────────────────────────────

def infer_from_command_txt(run_dir: Path):
    """Read env/algo from command.txt if it exists."""
    cmd_file = run_dir / "command.txt"
    if not cmd_file.exists():
        return None, None
    txt = cmd_file.read_text(errors="ignore")
    env_m = re.search(r"--env(?:-id)?\s+([^\s]+)", txt)
    algo_m = re.search(r"--algo\s+([^\s]+)", txt)
    return (
        env_m.group(1) if env_m else None,
        algo_m.group(1).lower() if algo_m else None,
    )


def infer_from_folder_name(run_dir: Path):
    """Extract env_id from folder name like 2026-02-22_143012_SSLDribbling-v0_seed0."""
    parts = run_dir.name.split("_")
    if len(parts) >= 4:
        return parts[2], None
    return None, None


def infer_env_algo(run_dir: Path):
    """Infer env_id and algo from command.txt, folder name, and parent dir."""
    env_id, algo = infer_from_command_txt(run_dir)
    if algo is None:
        algo = run_dir.parent.name.lower()
    if env_id is None:
        env2, _ = infer_from_folder_name(run_dir)
        env_id = env2
    return env_id, algo


# ── Checkpoint discovery ──────────────────────────────────────────────────────

def find_checkpoints(run_dir: Path) -> list[Path]:
    """Find all .zip files in checkpoints/ (flat or nested)."""
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return []
    return sorted(set(ckpt_dir.glob("*.zip")) | set(ckpt_dir.glob("*/*.zip")))


def get_missing_videos(run_dir: Path, zips: list[Path], force: bool = False) -> list[Path]:
    """Return checkpoints that don't yet have a matching video."""
    if force:
        return zips
    video_dir = run_dir / "videos"
    existing = {p.stem for p in video_dir.glob("*.mp4")} if video_dir.exists() else set()
    return [z for z in zips if z.stem not in existing]


def collect_run_dirs(experiments_root: Path) -> list[Path]:
    """Collect all run directories under experiments/<ALGO>/<run>/."""
    run_dirs = []
    for algo_dir in sorted(experiments_root.iterdir()):
        if not algo_dir.is_dir():
            continue
        for run_dir in sorted(algo_dir.iterdir()):
            if run_dir.is_dir():
                run_dirs.append(run_dir)
    return run_dirs


# ── Core renderer ─────────────────────────────────────────────────────────────

def render_checkpoint(zip_path: Path, env_id: str, algo: str, out_path: Path,
                      fps: int = 30, deterministic: bool = True,
                      on_progress=None) -> dict:
    """
    Render one checkpoint to mp4. Returns stats dict.

    Args:
        on_progress: optional callback(step: int) called every 1000 steps.
    """
    env = gym.make(env_id, render_mode="rgb_array")
    model = ALGO_MAP[algo].load(str(zip_path), env=env, device="cpu")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)

    obs, _ = env.reset()
    writer.append_data(env.render())

    step = 0
    t0 = time.time()

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        writer.append_data(env.render())

        if step % 1000 == 0 and on_progress:
            on_progress(step)

        if terminated or truncated:
            break

    writer.close()
    env.close()
    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / (1024 * 1024)

    return {
        "steps": step,
        "duration_sec": elapsed,
        "size_mb": size_mb,
    }


def run_label(run_dir: Path) -> str:
    """Short label for display: ALGO/run_name."""
    return f"{run_dir.parent.name}/{run_dir.name}"