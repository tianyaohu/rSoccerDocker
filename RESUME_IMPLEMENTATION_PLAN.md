# Crash-Proof Resume: Implementation Plan

**Date:** 2026-03-10  
**Context:** This document captures all design decisions, open questions, and implementation details from a planning session between Frank and Claude (Opus). It is designed to be handed to two separate Claude Code sessions — one for tests, one for implementation.

**The one sentence goal:** Stop and resume a SAC or DDPG training run with zero loss in reward curve — as if training never stopped.

---

## Table of Contents

1. [The Bug](#1-the-bug)
2. [Design Decisions (Locked)](#2-design-decisions-locked)
3. [Approach C: Hybrid Atomic Writes](#3-approach-c-hybrid-atomic-writes)
4. [Phasing Plan](#4-phasing-plan)
5. [CC1: Test Specification](#5-cc1-test-specification)
6. [CC2: Implementation Specification](#6-cc2-implementation-specification)
7. [Open Questions (Unresolved)](#7-open-questions-unresolved)
8. [File-by-File Change Map](#8-file-by-file-change-map)
9. [Future Work (Out of Scope)](#9-future-work-out-of-scope)

---

## 1. The Bug

### What happens now

`RotatingCheckpointCallback._on_step()` calls `self.model.save()`, which saves model weights, optimizer state, and algorithm parameters — but **NOT** the replay buffer. SB3 explicitly excludes the replay buffer from `model.save()` by design.

When `build_model()` calls `AlgoClass.load()` on resume, the model loads with an **empty** replay buffer. The policy network has been trained on hundreds of thousands of transitions, but the buffer is empty. The model immediately starts collecting fresh (low-quality) data and performing gradient updates with it.

### Observable symptoms

- Sudden reward cliff at the resume point
- The cliff is **silent** — training appears to continue, steps increment, but performance collapses
- The model must re-explore from scratch, wasting all prior data collection

### Why it affects SAC and DDPG but not PPO

SAC and DDPG are **off-policy** algorithms. They maintain a replay buffer that stores past transitions and sample from it for gradient updates. Losing this buffer means losing the data distribution the policy was optimized for.

PPO is **on-policy**. It collects fresh rollouts each update and discards them. There is no persistent buffer to lose. PPO resume should work correctly with just `model.save()` / `AlgoClass.load()` + `reset_num_timesteps=False`.

### Additional problem: three separate save paths

Currently `train.py` has THREE places where models are saved, and only one goes through the callback:

**Place 1 — Callback (during training):** `RotatingCheckpointCallback._on_step()` → `self.model.save()`  
**Place 2 — Ctrl+C handler:** `except KeyboardInterrupt` block → `model.save()` directly  
**Place 3 — Final save:** after `model.learn()` completes → `model.save()` directly  

Places 2 and 3 bypass the callback entirely. After we fix Place 1 to save the buffer, Places 2 and 3 still won't save it. All three must use the same save logic.

---

## 2. Design Decisions (Locked)

These were discussed and agreed upon. Not open for renegotiation unless new information surfaces.

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **Approach C (hybrid atomic writes)** for checkpoint format | Checkpoint dir is atomic unit for small state; buffer is single overwritten file. Balance of consistency and disk efficiency. |
| D2 | **SAC + DDPG** are in scope. PPO deferred. TD3 deferred but not removed. | Off-policy buffer loss is the critical bug. TD3 uses identical buffer mechanism — trivial to add later. |
| D3 | **Test-first.** Test file is implemented before the fix. | Minimizes rework. Tests define the interface contract. |
| D4 | **One save path.** Extract save logic into a shared function. Callback, interrupt handler, and final save all call it. | Two save paths = two formats = bugs. |
| D5 | **Pendulum-v1** for automated tests. SSLDribbling-v0 for manual smoke test. | Pendulum is fast (~2 min for 50k steps), has continuous action space (works with SAC/DDPG), no rSoccer dependency. |
| D6 | **VecNormalize** is out of scope for this fix. | Current `make_env()` doesn't use VecNormalize for off-policy algos. |
| D7 | **RNG state preservation** for single-env (SAC/DDPG). Deferred for VecEnv (PPO). | Single-env RNG is feasible (torch, numpy, env). VecEnv requires cross-process IPC that SB3 doesn't expose. |
| D8 | **Inconsistency detection.** If buffer and checkpoint are from different steps, emit a warning on load. Not silent, not an error. | "Fresh buffer + stale model" is benign for off-policy but must be detectable for debugging. |
| D9 | **`checkpoint_final/`** directory instead of bare `_final.zip`. | Consistency with Approach C. Same format everywhere. |
| D10 | **Tests validate observable behavior**, not internal file layout. | Tests survive refactors. "Reward continues" is more meaningful than "file exists at path X." |
| D11 | **`resume.py` needs updating** to find checkpoint dirs instead of bare .zip files. | Current `highest_checkpoint()` globs `*.zip` in the flat dir. With Approach C, model.zip is inside `checkpoint_step_*/`. |

---

## 3. Approach C: Hybrid Atomic Writes

### On-disk layout after a training run

```
experiments/SAC/2026-03-10_143000_SSLDribbling-v0_seed0/
├── command.txt                          ← training command for reproducibility
├── wandb_run_id.txt                     ← W&B run ID for resume
├── logs/                                ← TensorBoard logs
│   └── SAC_1/events.out.tfevents...
├── checkpoints/
│   ├── checkpoint_step_100000/          ← atomic dir (renamed from _temp_...)
│   │   ├── model.zip                    ← SB3: weights + optimizer + algo params
│   │   ├── rng_state.pkl                ← torch, numpy, env RNG state (~KB)
│   │   └── metadata.json               ← step, timestamp, algo, buffer_step
│   ├── checkpoint_step_200000/
│   │   ├── model.zip
│   │   ├── rng_state.pkl
│   │   └── metadata.json
│   ├── checkpoint_step_300000/
│   │   ├── model.zip
│   │   ├── rng_state.pkl
│   │   └── metadata.json
│   ├── checkpoint_final/                ← terminal checkpoint (same format)
│   │   ├── model.zip
│   │   ├── rng_state.pkl
│   │   └── metadata.json
│   └── replay_buffer.pkl               ← SINGLE file, overwritten each save
│                                          lives alongside dirs, not inside
└── videos/
```

### Save sequence (what happens at step 300000)

```
1. Save buffer → checkpoints/_temp_replay_buffer.pkl
2. Atomic rename → checkpoints/replay_buffer.pkl
   ─── if crash here: buffer is ahead of checkpoint (detectable, benign) ───
3. Create checkpoints/_temp_checkpoint_step_300000/
4. Write model.zip into temp dir
5. Write rng_state.pkl into temp dir (torch + numpy + env RNG)
6. Write metadata.json into temp dir
      {
        "checkpoint_step": 300000,
        "buffer_step": 300000,       ← for inconsistency detection
        "algorithm": "SAC",
        "timestamp": "2026-03-10T14:30:00Z",
        "env_id": "SSLDribbling-v0",
        "seed": 42
      }
7. Atomic rename → checkpoints/checkpoint_step_300000/
   ─── both writes complete: fully consistent state ───
8. Apply retention: delete checkpoint_step_100000/ (if max_keep=3)
   NOTE: replay_buffer.pkl is NEVER deleted by retention
```

### Resume sequence

```
1. Discover: find checkpoint_step_*/ dirs, pick highest step
2. Read metadata.json → validate algorithm matches, extract buffer_step
3. Load model: AlgoClass.load(checkpoint_dir / "model.zip", env=env)
4. Load buffer: model.load_replay_buffer(checkpoints/ replay_buffer.pkl)
5. Compare: if metadata.buffer_step != metadata.checkpoint_step → warn
6. Load RNG: restore torch, numpy, env RNG from rng_state.pkl
7. Train: model.learn(remaining, reset_num_timesteps=False)
```

### Inconsistency detection

On load, compare `buffer_step` from metadata against `checkpoint_step`. Three cases:

| Case | Meaning | Action |
|------|---------|--------|
| `buffer_step == checkpoint_step` | Normal. Both saved successfully. | Proceed silently. |
| `buffer_step > checkpoint_step` | Buffer saved but crash before checkpoint dir rename. | Warn: "Buffer is from step X, model from step Y. Safe for off-policy." |
| No buffer file exists | Old run (before this fix), or buffer save failed. | Warn: "No replay buffer found. Resuming with empty buffer." |

### Retention policy interaction

- Rolling retention counts and deletes `checkpoint_step_*/` **directories**.
- `replay_buffer.pkl` is **never** touched by retention. It's a single file that always represents the latest state.
- Discovery pattern: glob `checkpoint_step_*/` directories (not `*.zip`).
- `checkpoint_final/` is **not** counted toward rolling window. It's a terminal artifact.

---

## 4. Phasing Plan

### Phase 0: Foundation (this document + test contract)
**Output:** This plan document. Reviewed and agreed upon.
**Boundary:** No code changes. Pure planning.

### Phase 1: Test Suite (CC1)
**Output:** `tests/test_resume.py` — complete test file.
**Boundary:** Tests only. No changes to `utils.py`, `train.py`, or `resume.py`.
**Key principle:** Tests define the *interface contract*. They test observable behavior (reward continuity, buffer size, RNG consistency), not file layout. When Phase 2 makes the tests pass, the feature is done.

**What tests cover:**
- Buffer roundtrip: save and load preserves transition count (SAC, DDPG)
- Reward continuity: interrupted + resumed run does not collapse vs continuous run
- RNG preservation: interrupted + resumed run produces identical trajectory to continuous run (SAC, DDPG single-env)
- PPO: no buffer file created, resume works without buffer
- Interrupt handler: Ctrl+C produces a valid, resumable checkpoint
- Backward compatibility: old runs without buffer file resume with warning, not crash
- Retention: buffer file survives checkpoint rotation
- Inconsistency detection: mismatched buffer/checkpoint steps produce warning

**What tests do NOT cover (yet):**
- VecNormalize save/restore
- Callback state persistence
- WandB integration
- Remote storage upload
- TD3 (trivial to add later — same buffer mechanism)

### Phase 2: Implementation (CC2)
**Output:** Changes to `utils.py`, `train.py`, `resume.py`.
**Boundary:** Make the Phase 1 tests pass. Nothing more.
**Depends on:** Phase 1 tests being complete.

**What changes:**
- `utils.py`: Replace `RotatingCheckpointCallback` with Approach C save logic
- `utils.py`: Update `build_model()` with full restore (buffer + RNG)
- `train.py`: Interrupt handler and final save use shared save function
- `train.py`: Remove direct `model.save()` calls from interrupt and final save paths
- `resume.py`: Update `highest_checkpoint()` to find checkpoint dirs, not bare .zip files

### Phase 3: Manual Smoke Test
**Output:** Visual confirmation on SSLDribbling-v0.
**Boundary:** Human runs training, kills it, resumes, checks TensorBoard.
**Depends on:** Phase 2 passing all automated tests.

### Phase 4: Skeleton & Extensibility (future)
**Output:** Stubs and interfaces for future components.
**What this includes (not exhaustive):**
- Signal handler registration (SIGTERM/SIGINT) — opt-in, default off
- Callback state persistence protocol (get_state/set_state)
- VecNormalize save/restore hooks
- Time-based checkpoint trigger (safety net alongside step-based)
- Permanent checkpoint tier in retention policy
- Remote storage upload hook (post-save)
- TD3 algorithm support

---

## 5. CC1: Test Specification

### File: `tests/test_resume.py`

### Test environment

- **Primary:** `Pendulum-v1` (Gymnasium built-in, continuous action, fast)
- **Algorithms under test:** SAC, DDPG
- **Parameterization:** Tests should be parameterized by algorithm where possible, so adding TD3 later is a one-line change

### Dependencies

```
pytest
stable-baselines3
gymnasium
numpy
torch
```

No rSoccer dependency. No WandB dependency.

### Test list

Each test below is described by: what it proves, how it works, and what "pass" looks like.

---

#### T1: `test_buffer_file_created_on_checkpoint`
**Parameterized by:** SAC, DDPG

**What it proves:** After a checkpoint save, a replay buffer file exists.

**How:**
1. Create model with Pendulum-v1
2. Train for 5000 steps with the checkpoint callback (save_freq=2000)
3. Assert: `replay_buffer.pkl` exists in checkpoints dir
4. Assert: file size > 0

---

#### T2: `test_buffer_transition_count_preserved`
**Parameterized by:** SAC, DDPG

**What it proves:** Buffer roundtrip preserves exact transition count.

**How:**
1. Create model, train for 5000 steps
2. Record `model.replay_buffer.size()`
3. Save checkpoint (full Approach C save)
4. Load checkpoint into fresh model
5. Load replay buffer
6. Assert: `new_model.replay_buffer.size() == original_size`
7. Assert: `original_size > 0` (sanity)

---

#### T3: `test_reward_continuity_on_resume`
**Parameterized by:** SAC, DDPG

**What it proves:** Resumed training does not suffer catastrophic reward collapse.

**How:**
1. Train model for 10000 steps. Evaluate: `reward_before` (mean over 10 episodes, deterministic).
2. Save full checkpoint.
3. Load into fresh model + buffer + RNG.
4. Evaluate immediately (no further training): `reward_after_load`.
5. Assert: `reward_after_load` is within 20% of `reward_before` (tight — policy behavior should be identical).
6. Train for 5000 more steps (`reset_num_timesteps=False`).
7. Evaluate: `reward_after_resume`.
8. Assert: `reward_after_resume >= reward_before * 0.7` (looser — some variance, but no collapse).

**Note:** If this test flakes due to stochasticity, widen tolerance or increase episode count. A true buffer-missing bug causes reward to drop to near-random, far outside any tolerance.

---

#### T4: `test_rng_produces_identical_trajectory`
**Parameterized by:** SAC, DDPG

**What it proves:** With RNG preservation, an interrupted+resumed run produces the exact same sequence of actions and rewards as a continuous run.

**How:**
1. Train model A for 5000 steps with fixed seed. Record RNG state. Save full checkpoint.
2. Continue model A for 2000 more steps. Record the sequence: `[(obs, action, reward), ...]` for all 2000 steps.
3. Load checkpoint into model B. Restore buffer + RNG.
4. Train model B for 2000 steps. Record the same sequence.
5. Assert: sequences are identical (np.allclose for floats).

**This is the strongest resume test.** If action sequences match step-for-step, the resume is functionally perfect.

**Risk:** This test may be fragile if SB3 has internal state we're not capturing. If it fails despite our best RNG restoration, we document what we couldn't capture and fall back to T3 (reward continuity) as the primary signal.

---

#### T5: `test_ppo_no_buffer_file`

**What it proves:** PPO (on-policy) does not create a replay buffer file.

**How:**
1. Create PPO model, train for 5000 steps with checkpoint callback.
2. Assert: no `replay_buffer.pkl` in checkpoints dir.

---

#### T6: `test_interrupt_produces_valid_checkpoint`
**Parameterized by:** SAC, DDPG

**What it proves:** Pressing Ctrl+C during training produces a checkpoint in the same Approach C format that can be resumed from.

**How:**
1. Train model. After first checkpoint fires (step 2000), send SIGINT or raise KeyboardInterrupt.
2. Assert: an interrupt checkpoint directory exists with model.zip + rng_state.pkl + metadata.json.
3. Assert: replay_buffer.pkl was saved.
4. Load the interrupt checkpoint. Verify model produces predictions (not corrupt).

**Implementation note:** Simulating KeyboardInterrupt in a test requires either threading (run `model.learn()` in a thread, send interrupt from main) or monkey-patching the callback to raise after N steps. The second approach is simpler and more reliable.

---

#### T7: `test_resume_with_missing_buffer_warns_not_crashes`
**Parameterized by:** SAC, DDPG

**What it proves:** Backward compatibility — old runs without buffer file still resume.

**How:**
1. Create model, train for 5000 steps, save checkpoint.
2. **Delete** `replay_buffer.pkl` manually.
3. Load checkpoint.
4. Assert: warning emitted (capture with `warnings.catch_warnings`).
5. Assert: `model.replay_buffer.size() == 0`.
6. Assert: no exception raised.
7. Train for 1000 more steps — no crash.

---

#### T8: `test_buffer_survives_checkpoint_rotation`
**Parameterized by:** SAC, DDPG

**What it proves:** Retention policy deletes old checkpoint dirs but never touches the buffer file.

**How:**
1. Train with save_freq=1000, max_keep=2, for 5000 steps.
2. Assert: at most 2 `checkpoint_step_*/` directories exist.
3. Assert: `replay_buffer.pkl` exists and has a recent modification time.

---

#### T9: `test_inconsistency_detection_warns`

**What it proves:** If buffer_step and checkpoint_step disagree, a warning is emitted on load.

**How:**
1. Save a full checkpoint at step 5000.
2. Manually overwrite metadata.json to set `buffer_step` to a different value (e.g., 6000).
3. Load checkpoint.
4. Assert: warning emitted mentioning the step mismatch.
5. Assert: no exception raised — loading proceeds.

---

#### T10: `test_model_predictions_identical_after_load`
**Parameterized by:** SAC, DDPG

**What it proves:** Loaded model produces bit-identical predictions to saved model.

**How:**
1. Train for 5000 steps. Save checkpoint.
2. Generate 10 observations from env.observation_space.sample().
3. Record: `[model.predict(obs, deterministic=True) for obs in observations]`.
4. Load checkpoint into fresh model.
5. Record predictions from loaded model.
6. Assert: `np.allclose(before, after)` for all observations.

---

### Test infrastructure

**Fixtures:**
- `tmp_checkpoint_dir` — pytest `tmp_path` based, cleaned up automatically
- `make_model(algo, env_id="Pendulum-v1", seed=42)` — factory that creates a model with deterministic seed
- `save_checkpoint(model, checkpoint_dir, step)` — calls the shared save function (the interface CC2 must implement)
- `load_checkpoint(checkpoint_dir, algo_class, env)` — calls the shared load function

**These fixtures define the interface contract.** CC2 must implement functions matching these signatures. The fixtures import from the implementation module. Initially they can raise `NotImplementedError` until CC2 fills them in.

**Parameterization pattern:**
```python
@pytest.mark.parametrize("algo_name,algo_class", [("sac", SAC), ("ddpg", DDPG)])
def test_buffer_transition_count_preserved(algo_name, algo_class, tmp_path):
    ...
```

Adding TD3 later = adding `("td3", TD3)` to the parameter list.

### Expected runtimes

| Test | Steps | Estimated time |
|------|-------|---------------|
| T1 | 5k | ~10s |
| T2 | 5k | ~15s |
| T3 | 15k | ~45s |
| T4 | 7k | ~30s |
| T5 | 5k (PPO) | ~10s |
| T6 | 5k | ~15s |
| T7 | 6k | ~15s |
| T8 | 5k | ~15s |
| T9 | 5k | ~10s |
| T10 | 5k | ~10s |
| **Total** | | **~3 min** |

---

## 6. CC2: Implementation Specification

### Prerequisite

All Phase 1 tests (T1-T10) exist and currently **fail** (because the implementation doesn't exist yet). CC2's job is to make them pass.

### Shared save function

Extract from the callback into a standalone function. This is the "one save path" that the callback, interrupt handler, and final save all call.

```python
def save_checkpoint(
    model,
    checkpoints_dir: Path,
    step: int,
    checkpoint_type: str = "rolling",  # "rolling" | "interrupt" | "final"
    verbose: int = 1,
) -> Path:
    """
    Save a complete Approach C checkpoint.
    
    Returns the path to the checkpoint directory.
    
    Steps:
    1. Save replay buffer (if off-policy) with atomic rename
    2. Create temp checkpoint dir
    3. Write model.zip, rng_state.pkl, metadata.json into temp dir
    4. Atomic rename temp dir to final name
    """
    ...
```

### Shared load function

```python
def load_checkpoint(
    checkpoint_dir: Path,
    algo_class,
    env,
    checkpoints_root: Path | None = None,
    verbose: int = 1,
) -> tuple:  # (model, metadata)
    """
    Load a complete Approach C checkpoint.
    
    Steps:
    1. Read and validate metadata.json
    2. Load model via AlgoClass.load()
    3. Load replay buffer (if exists, warn if missing)
    4. Compare buffer_step vs checkpoint_step, warn if mismatched
    5. Restore RNG state (if rng_state.pkl exists)
    
    Returns (model, metadata).
    """
    ...
```

### RNG save/restore

```python
def save_rng_state(env, path: Path):
    """Save torch, numpy, and single-env gymnasium RNG state."""
    import torch, numpy as np
    state = {
        "torch_rng": torch.random.get_rng_state(),
        "numpy_rng": np.random.get_state(),
    }
    # Gymnasium env RNG (if accessible)
    if hasattr(env, "np_random"):
        state["env_rng"] = env.np_random.bit_generator.state
    elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "np_random"):
        state["env_rng"] = env.unwrapped.np_random.bit_generator.state
    # Note: CUDA RNG not saved (we run on CPU for rSoccer)
    with open(path, "wb") as f:
        pickle.dump(state, f)

def load_rng_state(env, path: Path):
    """Restore torch, numpy, and single-env gymnasium RNG state."""
    import torch, numpy as np
    with open(path, "rb") as f:
        state = pickle.load(f)
    torch.random.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])
    if "env_rng" in state:
        target = getattr(env, "np_random", None)
        if target is None and hasattr(env, "unwrapped"):
            target = getattr(env.unwrapped, "np_random", None)
        if target is not None:
            target.bit_generator.state = state["env_rng"]
```

**Honest caveat:** SB3 may have internal RNG state beyond what we capture here (e.g., action noise in DDPG uses its own RNG). Test T4 will tell us if our RNG restoration is complete. If T4 fails despite these saves, we'll need to investigate what we're missing.

### Changes to existing files

#### `utils.py`

**RotatingCheckpointCallback** — modify `_on_step()` to call `save_checkpoint()` instead of `self.model.save()`. Modify `_sorted_checkpoints()` to glob `checkpoint_step_*/` dirs instead of `*.zip` files. Retention deletes directories.

**`build_model()`** — after `AlgoClass.load()`, call `load_checkpoint()` logic to restore buffer + RNG. Or: refactor so `build_model()` delegates to `load_checkpoint()` when resuming.

#### `train.py`

**KeyboardInterrupt handler** — replace `model.save(str(interrupt_path))` with `save_checkpoint(model, ckpt_dir, step=model.num_timesteps, checkpoint_type="interrupt")`.

**Final save** — replace `model.save(str(final_path))` with `save_checkpoint(model, ckpt_dir, step=model.num_timesteps, checkpoint_type="final")`.

#### `resume.py`

**`highest_checkpoint()`** — change from globbing `*.zip` to globbing `checkpoint_step_*/`. Extract step from directory name instead of filename. Skip `checkpoint_final/`.

**`_step_from_filename()`** — rename or add `_step_from_dirname()` that parses `checkpoint_step_(\d+)` from directory names.

---

## 7. Open Questions (Unresolved)

These need answers before or during implementation.

| # | Question | Owner | Impact |
|---|----------|-------|--------|
| O1 | Does SB3's DDPG action noise have its own RNG that we need to save? | Claude (SB3 source check) | If yes, T4 may fail without it |
| O2 | Does `AlgoClass.load()` reset `model.num_timesteps` correctly from the saved `data` file? | Claude (SB3 source check) | Affects `reset_num_timesteps=False` correctness |
| O3 | Network topology for volunteer cluster: host networking (a) or pod networking (b) with Tailscale? | Frank (defer) | Out of scope for this fix, but affects Phase 4 remote storage |
| O4 | Should `checkpoint_final/` be counted by `is_run_finished()`? | Both | Currently checks `*_final.zip`. Needs update for dir format. |
| O5 | Monitor wrapper: does `Monitor` wrap env in a way that hides `np_random`? | Claude (check during implementation) | Affects RNG save/restore path |

---

## 8. File-by-File Change Map

| File | Changes | Touched by |
|------|---------|------------|
| `tests/test_resume.py` | **NEW FILE** — all tests from §5 | CC1 only |
| `utils.py` | Modify `RotatingCheckpointCallback`, add `save_checkpoint()`, add `load_checkpoint()`, add RNG helpers, modify `build_model()` | CC2 only |
| `train.py` | Modify interrupt handler, modify final save, both use `save_checkpoint()` | CC2 only |
| `resume.py` | Modify `highest_checkpoint()`, add `_step_from_dirname()`, update `is_run_finished()` | CC2 only |
| `render_core.py` | **NO CHANGES** — rendering loads models for inference only | — |
| `render_videos.py` | **NO CHANGES** | — |
| `watch_and_render.py` | **NO CHANGES** | — |
| `compose.yml` | **NO CHANGES** (test service already exists) | — |

---

## 9. Future Work (Out of Scope)

These are explicitly deferred. They are mentioned here so that design decisions above don't accidentally close the door on them.

| Feature | Why deferred | What to watch for |
|---------|-------------|-------------------|
| **PPO RNG preservation** | VecEnv multi-process RNG requires IPC | Don't couple RNG save to SubprocVecEnv assumptions |
| **TD3 support** | Same buffer mechanism as SAC/DDPG | Just add to ALGOS dict + test params |
| **VecNormalize save/restore** | Not used in current make_env() for off-policy | save_checkpoint() should have a hook point for env wrapper state |
| **Callback state persistence** | No custom stateful callbacks in MVP | save_checkpoint() could accept optional `extra_state: dict` |
| **Time-based trigger** | Step-based is sufficient for testing | _should_checkpoint() can gain a time condition later |
| **Permanent checkpoint tier** | Rolling-only is fine for MVP | retention logic should be parameterizable |
| **Signal handling (SIGTERM)** | Interrupt handler covers Ctrl+C; SIGTERM is cluster concern | save_checkpoint() being a standalone function makes signal handler trivial to add |
| **WandB snapshot workflow** | Test locally first | save_checkpoint() output (a directory) is zip-and-uploadable |
| **Remote storage (MinIO/S3)** | Requires cluster networking decisions | save_checkpoint() returns the checkpoint path — a post-save hook can upload it |
| **TensorBoard resume markers** | Visual nicety, not functional | _init_callback or load_checkpoint can add a scalar log |
| **Dry-run / inspect mode** | Can inspect files manually for now | metadata.json is human-readable by design |

---

## Handoff Instructions

### For CC1 (Test Session)

**Your job:** Create `tests/test_resume.py` with tests T1-T10 from §5.

**What you have:** This document. The current source files (utils.py, train.py, resume.py) for understanding the existing codebase.

**What you produce:** A complete test file that imports from a `checkpoint_utils` module (or similar). The import will initially fail because the module doesn't exist — that's expected. The test file defines the interface contract that CC2 must satisfy.

**Key files to read first:** `utils.py` (current callback and build_model), `train.py` (current interrupt handler and final save), this plan.

### For CC2 (Implementation Session)

**Your job:** Make all CC1 tests pass. Touch only: `utils.py`, `train.py`, `resume.py`.

**What you have:** This document. The test file from CC1. The current source files.

**What you produce:** Modified source files where all tests pass. The shared `save_checkpoint()` and `load_checkpoint()` functions. RNG save/restore. Updated resume logic.

**Key constraint:** Do not modify the tests. If a test seems wrong, flag it — don't change it.
