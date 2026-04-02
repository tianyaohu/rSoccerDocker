#!/usr/bin/env python3
"""
W&B end-to-end smoke test — verifies metrics, video, and artifact upload.

Requires: WANDB_API_KEY in environment (from .env via Docker).
Uses project "wandb-smoke-test" to avoid polluting real experiments.
Cleans up the test run on exit.

Run:
    docker compose run --rm wandb-smoke
    # or directly inside the container:
    python scripts/wandb_smoke_test.py
"""

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

import wandb
from wandb.integration.sb3 import WandbCallback

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ENV_ID = "Pendulum-v1"
SEED = 42
TRAIN_STEPS = 500
ROLLOUT_STEPS = 200
PROJECT = "wandb-smoke-test"
ENTITY = "sfurs"

torch.set_num_threads(2)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(seed: int = SEED) -> Monitor:
    env = Monitor(gym.make(ENV_ID))
    env.unwrapped.np_random = np.random.Generator(np.random.PCG64(seed))
    return env


def print_result(name: str, passed: bool):
    symbol = "\033[32m PASS \033[0m" if passed else "\033[31m FAIL \033[0m"
    print(f"  [{symbol}] {name}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # -- Preflight ---------------------------------------------------------
    if not os.environ.get("WANDB_API_KEY"):
        print("ERROR: WANDB_API_KEY not set. Add it to .env or export it.", flush=True)
        sys.exit(1)

    results = {"Metrics": False, "Video": False, "Artifact": False}
    run_id = None
    tmp_dir = tempfile.mkdtemp(prefix="wandb_smoke_")
    tmp_path = Path(tmp_dir)

    print(f"\n{'='*50}", flush=True)
    print("  W&B End-to-End Smoke Test", flush=True)
    print(f"{'='*50}\n", flush=True)

    try:
        # -- Init W&B run --------------------------------------------------
        run_name = f"smoke-{datetime.now():%Y%m%d-%H%M%S}"
        run = wandb.init(
            project=PROJECT,
            entity=ENTITY,
            name=run_name,
            config={"env": ENV_ID, "algo": "sac", "steps": TRAIN_STEPS},
            tags=["smoke-test"],
        )
        run_id = run.id
        print(f"  W&B run: {ENTITY}/{PROJECT}/{run_id}\n", flush=True)

        # -- Check 1: Metrics ----------------------------------------------
        print("  Check 1: Metrics ...", flush=True)
        env = make_env()
        model = SAC("MlpPolicy", env, verbose=0, seed=SEED, batch_size=64)
        model.learn(
            TRAIN_STEPS,
            callback=WandbCallback(gradient_save_freq=0, verbose=0),
        )
        wandb.log({"smoke/custom_metric": 42.0})
        results["Metrics"] = True
        print_result("Metrics", True)

        # -- Check 2: Video ------------------------------------------------
        print("  Check 2: Video ...", flush=True)
        vid_env = gym.make(ENV_ID, render_mode="rgb_array")
        mp4_path = tmp_path / "smoke_test.mp4"
        writer = imageio.get_writer(str(mp4_path), fps=30, codec="libx264", quality=8)

        obs, _ = vid_env.reset(seed=SEED)
        writer.append_data(vid_env.render())
        for _ in range(ROLLOUT_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = vid_env.step(action)
            writer.append_data(vid_env.render())
            if terminated or truncated:
                break
        writer.close()
        vid_env.close()

        wandb.log({"video": wandb.Video(str(mp4_path), fps=30, format="mp4")})
        results["Video"] = True
        print_result("Video", True)

        # -- Check 3: Artifact ---------------------------------------------
        print("  Check 3: Artifact ...", flush=True)
        model_path = tmp_path / "model"
        model.save(str(model_path))

        artifact = wandb.Artifact("smoke-test-model", type="model")
        artifact.add_file(str(model_path) + ".zip")
        wandb.log_artifact(artifact)
        results["Artifact"] = True
        print_result("Artifact", True)

        # -- Finish (flushes all uploads) ----------------------------------
        env.close()
        wandb.finish()

    except Exception as exc:
        print(f"\n  ERROR: {exc}", flush=True)
        try:
            wandb.finish(exit_code=1)
        except Exception:
            pass

    # -- Cleanup -----------------------------------------------------------
    if run_id:
        print("\n  Cleaning up test run ...", flush=True)
        try:
            api = wandb.Api()
            api.run(f"{ENTITY}/{PROJECT}/{run_id}").delete()
            print("  Deleted run from W&B.", flush=True)
        except Exception as e:
            print(f"  Warning: cleanup failed ({e}). "
                  f"Orphaned run in {PROJECT} project.", flush=True)

    # -- Summary -----------------------------------------------------------
    print(f"\n{'='*50}", flush=True)
    print("  Results:", flush=True)
    all_passed = True
    for name, passed in results.items():
        print_result(name, passed)
        if not passed:
            all_passed = False
    print(f"{'='*50}\n", flush=True)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
