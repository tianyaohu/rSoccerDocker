#!/usr/bin/env python3
"""
Train RL agents on rSoccer environments (SAC / DDPG / PPO).

Usage
-----
python src/train.py --env SSLDribbling-v0 --algo sac
python src/train.py --env VSS-v0 --algo ppo --n-envs 4

# Resume newest unfinished run:
python src/train.py --resume latest

# Resume a specific run by folder name:
python src/train.py --resume 2026-02-25_024550_SSLDribbling-v0_seed0
"""

import argparse
import re
import sys
from pathlib import Path

import wandb
from wandb.integration.sb3 import WandbCallback

from resume import ResumeError, apply_config, find_latest_run, resolve_explicit
from utils import (
    ALGOS, OFF_POLICY,
    RotatingCheckpointCallback,
    build_model, create_run_dir, make_env,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments. Accepts argv for testing."""
    parser = argparse.ArgumentParser(
        description="Train RL agents on rSoccer environments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env",              default=None,
                        help="Gymnasium environment ID (default: SSLDribbling-v0)")
    parser.add_argument("--algo",             choices=list(ALGOS), default=None,
                        help="RL algorithm (default: sac)")
    parser.add_argument("--timesteps",        type=int, default=500_000,
                        help="Total training timesteps")
    parser.add_argument("--save-freq",        type=int, default=100_000,
                        help="Checkpoint every N timesteps")
    parser.add_argument("--max-checkpoints",  type=int, default=5,
                        help="Max checkpoints to keep (oldest deleted first)")
    parser.add_argument("--n-envs",           type=int, default=4,
                        help="Parallel envs (PPO only)")
    parser.add_argument("--seed",             type=int, default=0)
    parser.add_argument("--resume",           default=None, metavar="RUN_DIR_OR_LATEST",
                        help="'latest' to auto-find newest unfinished run, "
                             "or a run folder name/path")
    parser.add_argument("--root",             default=".",
                        help="Project root (default: current directory)")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Resume resolution
# ---------------------------------------------------------------------------

def resolve_resume(args) -> tuple[str | None, Path | None]:
    """
    Resolve --resume into (resume_mode, run_dir).

    Mutates args: applies config from command.txt, sets args.resume
    to the checkpoint path.

    Returns (resume_mode, run_dir):
        None, None        → fresh run
        "latest", Path    → auto-found newest unfinished run
        "explicit", Path  → user-specified run directory
    """
    root = Path(args.root).resolve()
    experiments = root / "experiments"

    if args.resume == "latest":
        result = find_latest_run(experiments, env_filter=args.env, algo_filter=args.algo)
        apply_config(args, result.config)
        args.resume = str(result.checkpoint)
        _print_resume_info(result.run_dir, result.checkpoint)
        return "latest", result.run_dir

    elif args.resume:
        result = resolve_explicit(args.resume, experiments)
        apply_config(args, result.config)
        args.resume = str(result.checkpoint)
        if result.is_finished:
            print("WARNING: This run already has _final.zip. Resuming will overwrite it.\n")
        _print_resume_info(result.run_dir, result.checkpoint)
        return "explicit", result.run_dir

    else:
        if args.env is None:
            args.env = "SSLDribbling-v0"
        if args.algo is None:
            args.algo = "sac"
        return None, None


def _print_resume_info(run_dir: Path, checkpoint: Path):
    step = re.search(r"_(\d+)\.zip$", checkpoint.name)
    step_str = f"step {int(step.group(1)):,}" if step else checkpoint.name
    print(f"  Found: {run_dir.parent.name}/{run_dir.name}")
    print(f"  Checkpoint: {checkpoint.name} ({step_str})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None):
    args = parse_args(argv)
    root = Path(args.root).resolve()

    # ---- resolve resume ---------------------------------------------------
    try:
        resume_mode, run_dir = resolve_resume(args)
    except ResumeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if run_dir is None:
        run_dir = create_run_dir(root, args.algo, args.env, args.seed)

    ckpt_dir   = run_dir / "checkpoints"
    log_dir    = run_dir / "logs"
    final_path = ckpt_dir / f"{args.algo}_{args.env.replace('-', '_').lower()}_final"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---- wandb ------------------------------------------------------------
    wandb_run_id_file = run_dir / "wandb_run_id.txt"
    wandb_kwargs = dict(
        project=args.env.lower().replace("-v0", ""),
        entity="sfurs",
        name=run_dir.name,
        config=vars(args),
        tags=[args.algo, f"seed{args.seed}"],
        group=args.algo.upper(),
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )
    if resume_mode and wandb_run_id_file.exists():
        prev_id = wandb_run_id_file.read_text().strip()
        wandb_kwargs.update(id=prev_id, resume="allow")
        print(f"  Resuming W&B run: {prev_id}")

    wb_run = wandb.init(**wandb_kwargs)
    wandb_run_id_file.write_text(wb_run.id)

    # ---- save command for reproducibility ---------------------------------
    if not resume_mode:
        (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")

    # ---- banner -----------------------------------------------------------
    print("=" * 52)
    print(f"  Algorithm      : {args.algo.upper()}")
    print(f"  Env            : {args.env}")
    print(f"  Timesteps      : {args.timesteps:,}")
    print(f"  Seed           : {args.seed}")
    print(f"  Run dir        : {run_dir}/")
    if args.resume:
        print(f"  Resuming from  : {args.resume}")
    if args.algo not in OFF_POLICY:
        print(f"  Parallel envs  : {args.n_envs}")
    print("=" * 52)

    # ---- train ------------------------------------------------------------
    env   = make_env(args.env, args.algo, args.n_envs, args.seed)
    model = build_model(args.algo, env, log_dir, args.seed, args.resume)

    n_envs = getattr(env, "num_envs", 1)
    callbacks = [
        RotatingCheckpointCallback(
            save_freq   = max(args.save_freq // n_envs, 1),
            save_dir    = ckpt_dir,
            name_prefix = f"{args.algo}_{args.env}",
            max_keep    = args.max_checkpoints,
        ),
        WandbCallback(gradient_save_freq=0, verbose=0),
    ]

    print("\nTraining started  (Ctrl-C saves and exits)\n")
    try:
        model.learn(
            total_timesteps     = args.timesteps,
            callback            = callbacks,
            progress_bar        = True,
            reset_num_timesteps = resume_mode is None,
        )
    except KeyboardInterrupt:
        print("\nInterrupted — saving final model before exit…")

    model.save(str(final_path))
    print(f"\nDone. Final model → {final_path}.zip")

    wb_run.finish()
    env.close()


if __name__ == "__main__":
    main()