"""
Training utilities for rSoccer RL experiments.

- Algo registry (ALGOS, OFF_POLICY)
- Environment and model builders
- Rotating checkpoint callback
- Global progress callback (resume-aware progress bar)
- Run directory creation
"""
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import rsoccer_gym  # noqa: F401  registers the envs
from stable_baselines3 import DDPG, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor


# ---------------------------------------------------------------------------
# Run state (for progress bar and budget)
# ---------------------------------------------------------------------------

@dataclass
class RunState:
    """Immutable run parameters for global progress and checkpoint info."""
    target_total: int   # original --timesteps (total run budget)
    start_step: int     # 0 for fresh run, checkpoint step when resuming
    save_freq: int      # checkpoint every N steps


# ---------------------------------------------------------------------------
# Algo registry
# ---------------------------------------------------------------------------

ALGOS = {"sac": SAC, "ddpg": DDPG, "ppo": PPO}
OFF_POLICY = {"sac", "ddpg"}


# ---------------------------------------------------------------------------
# Environment + model builders
# ---------------------------------------------------------------------------

def make_env(env_id: str, algo: str, n_envs: int, seed: int):
    """Create a single Monitor-wrapped env (off-policy) or a VecEnv (PPO)."""
    if algo in OFF_POLICY:
        return Monitor(gym.make(env_id))
    return make_vec_env(env_id, n_envs=n_envs, seed=seed)


def build_model(algo: str, env, log_dir: Path, seed: int, resume: str | None, device: str):
    """Create a fresh model or load from checkpoint."""
    AlgoClass = ALGOS[algo]
    if resume:
        print(f"Resuming from: {resume}")
        return AlgoClass.load(resume, env=env, tensorboard_log=str(log_dir), device=device)
    return AlgoClass(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        tensorboard_log=str(log_dir),
        device=device,
    )


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------

def create_run_dir(root: Path, algo: str, env_id: str, seed: int) -> Path:
    """Create a timestamped run directory: experiments/<ALGO>/<timestamp>_<env>_seed<N>/"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_name = f"{timestamp}_{env_id}_seed{seed}"
    run_dir = root / "experiments" / algo.upper() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Rotating checkpoint callback
# ---------------------------------------------------------------------------

class RotatingCheckpointCallback(BaseCallback):
    """
    Save a checkpoint every `save_freq` timesteps.
    Keep at most `max_keep` checkpoints; when the limit is hit the oldest
    checkpoint is deleted before the new one is written.
    """

    def __init__(
        self,
        save_freq: int,
        save_dir: Path,
        name_prefix: str,
        max_keep: int = 5,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = Path(save_dir)
        self.name_prefix = name_prefix
        self.max_keep = max_keep
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _sorted_checkpoints(self) -> list[Path]:
        ckpts = list(self.save_dir.glob(f"{self.name_prefix}_*.zip"))

        def step_key(p: Path) -> int:
            m = re.search(r"_(\d+)\.zip$", p.name)
            return int(m.group(1)) if m else 0

        return sorted(ckpts, key=step_key)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True

        existing = self._sorted_checkpoints()
        while len(existing) >= self.max_keep:
            oldest = existing.pop(0)
            oldest.unlink()
            if self.verbose:
                print(f"  [ckpt] Removed old checkpoint: {oldest.name}")

        ckpt_path = self.save_dir / f"{self.name_prefix}_{self.num_timesteps:08d}"
        self.model.save(str(ckpt_path))
        kept = len(self._sorted_checkpoints())
        if self.verbose:
            print(f"  [ckpt] Saved → {ckpt_path}.zip  ({kept}/{self.max_keep})")
        return True


# ---------------------------------------------------------------------------
# Global progress callback (resume-aware progress bar)
# ---------------------------------------------------------------------------

class GlobalProgressCallback(BaseCallback):
    """
    Progress bar that reflects global run progress (including resumed steps).
    Segments: previously done (grey/dim), this session (green), remaining (dash).
    Shows distance to next checkpoint and checkpoints passed.
    """

    BAR_WIDTH = 36
    LOG_FREQ = 100  # update progress every N steps

    def __init__(self, run_state: RunState, verbose: int = 1):
        super().__init__(verbose)
        self.run_state = run_state
        self._last_log = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """Draw progress bar after rollout (and any dump_logs), so it stays at bottom."""
        if not self.verbose or self.num_timesteps - self._last_log < self.LOG_FREQ:
            return
        self._last_log = self.num_timesteps

        s = self.run_state
        current = self.num_timesteps
        current_clamped = min(current, s.target_total)
        frac_total = current_clamped / s.target_total if s.target_total else 0.0
        frac_before = s.start_step / s.target_total if s.target_total else 0.0
        frac_this = max(0.0, frac_total - frac_before)

        w = self.BAR_WIDTH
        n_before = int(frac_before * w)
        n_this = int(frac_this * w)
        n_todo = max(0, w - n_before - n_this)

        # Segments: = previously done, # this session, - remaining
        bar = "=" * n_before + "#" * n_this + "-" * n_todo
        pct = frac_total * 100

        next_ckpt = ((current // s.save_freq) + 1) * s.save_freq
        steps_to_next = max(0, min(next_ckpt - current, s.target_total - current))
        ckpts_passed = current // s.save_freq

        msg = (
            f"\r[{bar}] {pct:5.1f}% | step {current:,}/{s.target_total:,} | "
            f"next ckpt in {steps_to_next:,} (#{ckpts_passed})   "
        )
        sys.stdout.write(msg)
        sys.stdout.flush()

    def _on_training_end(self) -> None:
        if self.verbose:
            print()
