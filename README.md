# rSoccer Docker Compose — RL Experiments

Docker Compose setup for running **rSoccer** RL experiments (SAC / DDPG / PPO) with **Stable-Baselines3** and **Weights & Biases** logging.

---

## First-time setup

### 1. Build the Docker image

```bash
#FROM repo root
cd rsoccer_docker
docker build -t rsoccer:latest .
```

### 2. Set up W&B

Do this once:

```bash
cp .env.example .env
```

Open `.env` and paste your API key from [wandb.ai/authorize](https://wandb.ai/authorize).  
Sign up with your sfu.ca email account!
All runs log to the shared **sfurs** team workspace automatically — no `wandb login` needed.

> `**.env` is gitignored.** Please NEVER commit your wandb API key. 

---

## All commands assume repo root (`/sw-rl-agent/rsoccer`).

### Training

```bash
cd \rsoccer_docker
```

**Quick Test: Testing W&B Runs, Video Upload, (Take minutes? ).**
verify script functionality setup, logging, and video output._

```bash
docker compose run --rm train \
  --env SSLDribbling-v0 --algo sac --seed 0 \
  --timesteps 5000 --save-freq 1000 --max-checkpoints 5
```

Normal Experiments (Take hours)
```bash
docker compose run --rm train \
  --env SSLDribbling-v0 --algo sac --seed 0 \
  --timesteps 500000 --save-freq 100000 --max-checkpoints 5
```

#### train.py flags


| Flag                | Default           | Description                           |
| ------------------- | ----------------- | ------------------------------------- |
| `--env`             | `SSLDribbling-v0` | Gymnasium environment ID              |
| `--algo`            | `sac`             | Algorithm: `sac`, `ddpg`, or `ppo`    |
| `--timesteps`       | `500000`          | Total training timesteps              |
| `--save-freq`       | `100000`          | **Rotating checkpoint every N steps** |
| `--max-checkpoints` | `5`               | Max rotating checkpoints kept         |
| `--n-envs`          | `4`               | Parallel envs (PPO only)              |
| `--seed`            | `0`               | Random seed                           |
| `--resume`          | —                 | Resume from a checkpoint `.zip`       |


### Resuming training runs

Resume picks up the highest checkpoint, restores all hyperparameters from `command.txt`, and reattaches to the original W&B run.

```bash
# Resume the newest unfinished run
docker compose run --rm train --resume latest

# Narrow by env/algo
docker compose run --rm train --resume latest --env SSLDribbling-v0 --algo sac

# Resume a specific run (folder name, relative path, or absolute path all work)
docker compose run --rm train --resume 2026-02-22_143012_SSLDribbling-v0_seed0
docker compose run --rm train --resume SAC/2026-02-22_143012_SSLDribbling-v0_seed0
```

**How it works:**

- `command.txt` is parsed to recover the original flags — these **override CLI defaults**, so settings stay consistent.
- The highest-step `.zip` in `checkpoints/` is loaded (excluding `_final.zip`). Timestep counters continue (`reset_num_timesteps=False`).
- W&B run ID is read from `wandb_run_id.txt`, keeping your curves in one continuous run.
- Runs with a `_final.zip` are considered **finished** and skipped by `--resume latest`. Resuming a finished run explicitly shows a warning.
- No new run directory is created — logs, checkpoints, and videos stay in the original folder.

### Watcher

Check `experiments/` and renders videos as .zip checkpoints are saved. 
The video is also uploaded to wandb by picking up wandb_run_id cromb left by `train.py`
**Resource-capped so training keeps priority.**

```bash
# Terminal 2:
cd rsoccer_docker
docker compose run --rm watcher
```

### Batch render (Manual Rendering)

Renders any missing videos for all .zip checkpoint, then exits.

```bash
docker compose run --rm render
```

### TensorBoard [NEEDS TO BE TESTED]

```bash
docker compose run tensorboard
# → http://localhost:6006
```

---

## Run all experiments (Warning: it could take Days)

```bash
ENVS=("VSS-v0" "SSLStaticDefenders-v0" "SSLDribbling-v0" "SSLContestedPossession-v0" "SSLPassEndurance-v0")
ALGOS=("sac" "ddpg" "ppo")

for ENV in "${ENVS[@]}"; do
  [[ "$ENV" == "SSLDribbling-v0" ]] && STEPS=10000000 || STEPS=5000000

  for ALGO in "${ALGOS[@]}"; do
    EXTRA=""
    [[ "$ALGO" == "ppo" ]] && EXTRA="--n-envs 4"

    docker compose run --rm train \
      --env "$ENV" --algo "$ALGO" --seed 0 \
      --timesteps "$STEPS" --save-freq 100000 --max-checkpoints 5 $EXTRA
  done
done
```

---

## Project layout

```
rsoccer/
├── docker-compose.yml
├── Dockerfile
├── .env.example          ← copy to .env, add your W&B key
├── .env                  ← gitignored, your personal key
├── src/
│   ├── train.py          ← training (auto-creates run dirs, logs to W&B)
│   └── run.py            ← evaluation / rollout
├── scripts/
│   ├── render_core.py    ← shared rendering logic
│   ├── render_videos.py  ← one-shot batch renderer
│   └── watch_and_render.py  ← continuous watcher
└── experiments/
    ├── PPO/
    ├── DDPG/
    └── SAC/
```

Each training run auto-creates a timestamped folder:

```
experiments/SAC/2026-02-22_143012_SSLDribbling-v0_seed0/
├── checkpoints/    ← rotating, keeps last N
├── logs/           ← TensorBoard logs (also synced to W&B)
├── videos/         ← rendered by watcher or batch script
└── command.txt     ← exact command used
```

---

## rSoccer environments


| Env                           | Description                                  |
| ----------------------------- | -------------------------------------------- |
| **VSS-v0**                    | 3v3 Very Small Size Soccer                   |
| **SSLStaticDefenders-v0**     | 1 blue vs 6 stopped yellow defenders         |
| **SSLDribbling-v0**           | Dribble through gates past 4 stopped yellows |
| **SSLContestedPossession-v0** | 1v1 ball steal and score                     |
| **SSLPassEndurance-v0**       | 2 blue robots, complete a pass               |


---

## Docker resource tips

```bash
docker info | grep -E "CPUs|Memory"   # check allocation
docker stats                           # monitor live
```

Adjust via **Docker Desktop → Settings → Resources**.
