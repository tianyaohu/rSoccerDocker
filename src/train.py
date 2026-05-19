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
import cProfile
import os
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path

import torch
import wandb
from wandb.integration.sb3 import WandbCallback

from resume import ResumeError, ResumeResult, apply_config, find_latest_run, resolve_explicit
from utils import (
    ALGOS, OFF_POLICY,
    GlobalProgressCallback,
    RotatingCheckpointCallback,
    RunState,
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
    parser.add_argument("--device",           choices=("auto", "cpu", "cuda"), default="auto",
                        help="Training device: auto, cpu, or cuda")
    parser.add_argument("--resume",           default=None, metavar="RUN_DIR_OR_LATEST",
                        help="'latest' to auto-find newest unfinished run, "
                             "or a run folder name/path")
    parser.add_argument("--root",             default=".",
                        help="Project root (default: current directory)")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Resume resolution
# ---------------------------------------------------------------------------

def resolve_resume(args) -> tuple[str | None, Path | None, ResumeResult | None]:
    """
    Resolve --resume into (resume_mode, run_dir, resume_result).

    Mutates args: applies config from command.txt, sets args.resume
    to the checkpoint path.

    Returns (resume_mode, run_dir, resume_result):
        None, None, None     → fresh run
        "latest", Path, Res  → auto-found newest unfinished run (Res has start_step)
        "explicit", Path, Res → user-specified run directory
    """
    root = Path(args.root).resolve()
    experiments = root / "experiments"

    if args.resume == "latest":
        result = find_latest_run(experiments, env_filter=args.env, algo_filter=args.algo)
        apply_config(args, result.config)
        args.resume = str(result.checkpoint)
        _print_resume_info(result.run_dir, result.checkpoint)
        return "latest", result.run_dir, result

    elif args.resume:
        result = resolve_explicit(args.resume, experiments)
        apply_config(args, result.config)
        args.resume = str(result.checkpoint)
        if result.is_finished:
            print("WARNING: This run already has _final.zip. Resuming will overwrite it.\n")
        _print_resume_info(result.run_dir, result.checkpoint)
        return "explicit", result.run_dir, result

    else:
        if args.env is None:
            args.env = "SSLDribbling-v0"
        if args.algo is None:
            args.algo = "sac"
        return None, None, None


def _print_resume_info(run_dir: Path, checkpoint: Path):
    step = re.search(r"_(\d+)\.zip$", checkpoint.name)
    step_str = f"step {int(step.group(1)):,}" if step else checkpoint.name
    print(f"  Found: {run_dir.parent.name}/{run_dir.name}")
    print(f"  Checkpoint: {checkpoint.name} ({step_str})")


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def resolve_device(requested: str) -> str:
    """Resolve the requested training device and fail clearly for missing CUDA."""
    cuda_available = torch.cuda.is_available()
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested == "cuda" and not cuda_available:
        print(
            "ERROR: --device cuda was requested, but CUDA is not available to PyTorch. "
            "Check NVIDIA drivers, Docker GPU access, and the CUDA-enabled PyTorch install."
        )
        sys.exit(1)
    return requested


def describe_torch_device(device: str) -> list[str]:
    """Return short banner lines describing PyTorch and the selected device."""
    lines = [
        f"  Device         : {device}",
        f"  PyTorch        : {torch.__version__}",
        f"  CUDA available : {torch.cuda.is_available()}",
        f"  CUDA runtime   : {torch.version.cuda or 'n/a'}",
    ]
    if torch.cuda.is_available():
        try:
            lines.append(f"  GPU            : {torch.cuda.get_device_name(0)}")
        except Exception as exc:
            lines.append(f"  GPU            : unavailable ({exc})")
    return lines


def command_text(argv: list[str] | None, device: str) -> str:
    """Return a reproducible command with the effective training device."""
    tokens = list(sys.argv if argv is None else ["python", "src/train.py", *argv])
    cleaned = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token == "--device":
            skip_next = True
            continue
        if token.startswith("--device="):
            continue
        cleaned.append(token)
    cleaned.extend(["--device", device])
    return " ".join(shlex.quote(token) for token in cleaned) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None, *, hard_exit_on_interrupt: bool = False):
    args = parse_args(argv)
    root = Path(args.root).resolve()

    # ---- resolve resume ---------------------------------------------------
    try:
        resume_mode, run_dir, resume_result = resolve_resume(args)
    except ResumeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    resolved_device = resolve_device(args.device)
    args.device = resolved_device

    if run_dir is None:
        run_dir = create_run_dir(root, args.algo, args.env, args.seed)

    start_step = resume_result.start_step if resume_result is not None else 0
    remaining = max(0, args.timesteps - start_step)

    ckpt_dir   = run_dir / "checkpoints"
    log_dir    = run_dir / "logs"
    final_path = ckpt_dir / f"{args.algo}_{args.env.replace('-', '_').lower()}_final"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---- wandb ------------------------------------------------------------
    wandb_run_id_file = run_dir / "wandb_run_id.txt"
    wandb_kwargs = dict(
        project=args.env.lower().replace("-v0", ""),
        entity="mpb8-",
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
        wandb_kwargs.update(id=prev_id, resume="must")
        print(f"  Resuming W&B run: {prev_id}")

    # Initialize W&B run
    wb_run = wandb.init(**wandb_kwargs)
    wandb_run_id_file.write_text(wb_run.id)

    # ---- save command for reproducibility ---------------------------------
    if not resume_mode:
        (run_dir / "command.txt").write_text(command_text(argv, resolved_device))

    # ---- banner -----------------------------------------------------------
    print("=" * 52)
    print(f"  Algorithm      : {args.algo.upper()}")
    print(f"  Env            : {args.env}")
    print(f"  Timesteps      : {args.timesteps:,}")
    if resume_result is not None:
        print(f"  Remaining      : {remaining:,}")
    print(f"  Seed           : {args.seed}")
    for line in describe_torch_device(resolved_device):
        print(line)
    print(f"  Run dir        : {run_dir}/")
    if args.resume:
        print(f"  Resuming from  : {args.resume}")
    if args.algo not in OFF_POLICY:
        print(f"  Parallel envs  : {args.n_envs}")
    print("=" * 52)

    # ---- train ------------------------------------------------------------
    env   = make_env(args.env, args.algo, args.n_envs, args.seed)
    model = build_model(args.algo, env, log_dir, args.seed, args.resume, resolved_device)

    # Already at or past target: save _final and exit without calling learn()
    if remaining <= 0:
        print("\nRun already at or past target. Saving final and exiting.\n")
        model.save(str(final_path))
        print(f"Final model → {final_path}.zip")
        wb_run.finish()
        env.close()
        return

    # Verify loaded step matches checkpoint filename (warn if mismatch)
    if resume_result is not None:
        loaded = getattr(model, "num_timesteps", None)
        try:
            loaded_int = int(loaded) if loaded is not None else None
        except (TypeError, ValueError):
            loaded_int = None
        if loaded_int is not None and loaded_int != resume_result.start_step:
            print(
                f"  WARNING: Checkpoint filename step ({resume_result.start_step:,}) "
                f"!= model.num_timesteps ({loaded_int:,}). Proceeding anyway.\n"
            )

    n_envs = getattr(env, "num_envs", 1)
    run_state = RunState(
        target_total=args.timesteps,
        start_step=start_step,
        save_freq=max(args.save_freq // n_envs, 1),
    )
    callbacks = [
        RotatingCheckpointCallback(
            save_freq   = run_state.save_freq,
            save_dir    = ckpt_dir,
            name_prefix = f"{args.algo}_{args.env}",
            max_keep    = args.max_checkpoints,
        ),
        GlobalProgressCallback(run_state=run_state, verbose=1),
        WandbCallback(gradient_save_freq=0, verbose=0),
    ]
    # Create profiler
    profiler = cProfile.Profile()
    
    print("\nProfiling started...\n")
    profiler.enable()


    print("\nTraining started  (Ctrl-C to stop)\n")

    try:
        model.learn(
            total_timesteps     = remaining,
            callback            = callbacks,
            progress_bar        = False,
            reset_num_timesteps = resume_mode is None,
        )
    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoint for resume.")
        interrupt_path = ckpt_dir / f"{args.algo}_{args.env}_{model.num_timesteps:08d}"
        model.save(str(interrupt_path))
        print(f"  Saved → {interrupt_path}.zip")
        wb_run.finish()
        env.close()
        if hard_exit_on_interrupt:
            os._exit(0)
        return
    ## Save profiling
    profiler.disable()

    profile_dir = root / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    profile_path = profile_dir / f"training_profile_{timestamp}.prof"
    profiler.dump_stats(profile_path)
    print(f"\nProfiling completed. Saved to {profile_path}\n")




    model.save(str(final_path))
    print(f"\nDone. Final model → {final_path}.zip")

    wb_run.finish()
    env.close()


if __name__ == "__main__":
    main(hard_exit_on_interrupt=True)
