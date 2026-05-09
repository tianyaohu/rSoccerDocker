#!/usr/bin/env python3
"""
Train RL agents on rSoccer environments (SAC / DDPG / PPO).

All outputs are written to a timestamped run folder under experiments/<ALGO>/.

Usage
-----
python src/train.py --env SSLDribbling-v0 --algo sac
python src/train.py --env SSLDribbling-v0 --algo ddpg --timesteps 1000000
python src/train.py --env VSS-v0          --algo ppo  --n-envs 4
python src/train.py --env SSLDribbling-v0 --algo sac  --resume experiments/SAC/.../checkpoints/sac_SSLDribbling-v0_00100000.zip
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import rsoccer_gym  # noqa: F401  registers the envs
from stable_baselines3 import DDPG, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

import wandb
from wandb.integration.sb3 import WandbCallback

ALGOS = {"sac": SAC, "ddpg": DDPG, "ppo": PPO}
OFF_POLICY = {"sac", "ddpg"}


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
# Helpers
# ---------------------------------------------------------------------------

def make_env(env_id: str, algo: str, n_envs: int, seed: int):
    if algo in OFF_POLICY:
        return Monitor(gym.make(env_id))
    return make_vec_env(env_id, n_envs=n_envs, seed=seed)


def build_model(algo: str, env, log_dir: Path, seed: int, resume: str | None):
    AlgoClass = ALGOS[algo]
    if resume:
        print(f"Resuming from: {resume}")
        return AlgoClass.load(resume, env=env, tensorboard_log=str(log_dir))
    return AlgoClass("MlpPolicy", env, verbose=1, seed=seed, tensorboard_log=str(log_dir))


def create_run_dir(root: Path, algo: str, env_id: str, seed: int) -> Path:
    """Create a timestamped run directory: experiments/<ALGO>/<timestamp>_<env>_seed<N>/"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_name = f"{timestamp}_{env_id}_seed{seed}"
    run_dir = root / "experiments" / algo.upper() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train RL agents on rSoccer environments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env",              default="SSLDribbling-v0",
                        help="Gymnasium environment ID")
    parser.add_argument("--algo",             choices=list(ALGOS), default="sac",
                        help="RL algorithm")
    parser.add_argument("--timesteps",        type=int, default=500_000,
                        help="Total training timesteps")
    parser.add_argument("--save-freq",        type=int, default=100_000,
                        help="Checkpoint every N timesteps")
    parser.add_argument("--max-checkpoints",  type=int, default=5,
                        help="Max checkpoints to keep (oldest deleted first)")
    parser.add_argument("--n-envs",           type=int, default=4,
                        help="Parallel envs (PPO only)")
    parser.add_argument("--seed",             type=int, default=0)
    parser.add_argument("--resume",           default=None, metavar="CHECKPOINT_ZIP",
                        help="Resume training from a checkpoint .zip")
    parser.add_argument("--root",             default=".",
                        help="Project root (default: current directory)")
    args = parser.parse_args()

    root = Path(args.root).resolve()

    # ---- create run directory -----------------------------------------
    run_dir   = create_run_dir(root, args.algo, args.env, args.seed)
    ckpt_dir  = run_dir / "checkpoints"
    log_dir   = run_dir / "logs"
    final_path = ckpt_dir / f"{args.algo}_{args.env.replace('-', '_').lower()}_final"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---- initialize wandb ----------------------------------------------
    run = wandb.init(
        project=args.env.lower().replace("-v0", ""),  # "ssldribbling", "vss", etc.
        entity="sfurs",
        name=run_dir.name,              # e.g. "2025-02-24_120000_SSLDribbling-v0_seed0"
        config=vars(args),              # logs all your CLI args automatically
        tags=[args.algo, f"seed{args.seed}"],          # easy filtering
        group=args.algo.upper(),        # groups SAC runs together visually
        sync_tensorboard=True,          # pulls in your existing TensorBoard logs
        monitor_gym=True,
        save_code=True,
    )

    # ---- save the command for reproducibility -------------------------
    command_txt = run_dir / "command.txt"
    command_txt.write_text(" ".join(sys.argv) + "\n")

    print("=" * 52)
    print(f"  Algorithm      : {args.algo.upper()}")
    print(f"  Env            : {args.env}")
    print(f"  Timesteps      : {args.timesteps:,}")
    print(f"  Seed           : {args.seed}")
    print(f"  Run dir        : {run_dir}/")
    print(f"  Checkpoint dir : {ckpt_dir}/")
    print(f"  Max checkpoints: {args.max_checkpoints}")
    print(f"  Final model    : {final_path}.zip")
    print(f"  TensorBoard    : {log_dir}/")
    if args.algo not in OFF_POLICY:
        print(f"  Parallel envs  : {args.n_envs}")
    print("=" * 52)

    env   = make_env(args.env, args.algo, args.n_envs, args.seed)
    model = build_model(args.algo, env, log_dir, args.seed, args.resume)

    n_envs = getattr(env, "num_envs", 1)
    save_freq_calls = max(args.save_freq // n_envs, 1)

    callbacks = [
        RotatingCheckpointCallback(
            save_freq   = save_freq_calls,
            save_dir    = ckpt_dir,
            name_prefix = f"{args.algo}_{args.env}",
            max_keep    = args.max_checkpoints,
        ),
        WandbCallback(
            gradient_save_freq=0,   # set >0 if you want gradient histograms
            verbose=0,
        )
    ]

    print("\nTraining started  (Ctrl-C saves and exits)\n")
    try:
        model.learn(
            total_timesteps    = args.timesteps,
            callback           = callbacks,
            progress_bar       = True,
            reset_num_timesteps= args.resume is None,
        )
    except KeyboardInterrupt:
        print("\nInterrupted — saving final model before exit…")

    model.save(str(final_path))
    print(f"\nDone. Final model → {final_path}.zip")
    print(f"Run directory    → {run_dir}/")
    print("Render videos    → python scripts/render_videos.py")

    # ---- finish wandb -------------------------------------------------
    run.finish()

    env.close()


if __name__ == "__main__":
    main()
