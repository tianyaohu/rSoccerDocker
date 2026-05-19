"""
Resume logic for rSoccer training runs.

All functions are pure filesystem operations — no training dependencies,
no sys.exit(), no side effects beyond reading the filesystem.
Raises ResumeError on any failure so callers can handle it.
"""
import errno
import re
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ResumeError(Exception):
    """Any failure in resume resolution."""
    pass


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ResumeResult:
    """Everything needed to resume a training run."""
    run_dir: Path
    checkpoint: Path
    config: dict            # parsed from command.txt
    is_finished: bool       # True if _final.zip exists (warning, not error)
    start_step: int         # step count from checkpoint filename (for budget + progress)


# ---------------------------------------------------------------------------
# command.txt parsing
# ---------------------------------------------------------------------------

# All keys we expect in command.txt and how to parse them
COMMAND_KEYS = [
    (r"--env\s+(\S+)",              "env",              str),
    (r"--algo\s+(\S+)",             "algo",             str),
    (r"--timesteps\s+(\d+)",        "timesteps",        int),
    (r"--save-freq\s+(\d+)",        "save_freq",        int),
    (r"--seed\s+(\d+)",             "seed",             int),
    (r"--max-checkpoints\s+(\d+)",  "max_checkpoints",  int),
    (r"--n-envs\s+(\d+)",           "n_envs",           int),
    (r"--device\s+(\S+)",           "device",           str),
]

REQUIRED_KEYS = {"env", "algo"}


def parse_command_txt(run_dir: Path) -> dict:
    """
    Extract training args from a run's command.txt.

    Returns dict with string keys and already-typed values
    (str for env/algo, int for everything else).
    Returns empty dict if command.txt doesn't exist.
    """
    cmd_file = run_dir / "command.txt"
    if not cmd_file.exists():
        return {}

    try:
        txt = cmd_file.read_text(errors="ignore")
    except OSError as e:
        # On some host filesystems (e.g. macOS + Docker bind mounts),
        # reading a file can occasionally raise EDEADLK ("Resource deadlock avoided")
        # instead of a standard read error. Treat this as an unreadable
        # command.txt so callers can safely skip this run.
        if getattr(e, "errno", None) == errno.EDEADLK:
            return {}
        raise
    result = {}
    for pattern, key, parse_fn in COMMAND_KEYS:
        m = re.search(pattern, txt)
        if m:
            result[key] = parse_fn(m.group(1))
    return result


def validate_config(config: dict, source: Path) -> None:
    """Raise ResumeError if required keys are missing."""
    missing = REQUIRED_KEYS - set(config.keys())
    if missing:
        raise ResumeError(
            f"command.txt is missing required fields: {', '.join(f'--{k}' for k in sorted(missing))}\n"
            f"  File: {source / 'command.txt'}"
        )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def is_run_finished(run_dir: Path) -> bool:
    """A run is finished if it has a _final.zip checkpoint."""
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return False
    return any(ckpt_dir.glob("*_final.zip"))


def _step_from_filename(path: Path) -> int:
    """Extract step number from checkpoint filename. Returns 0 if not found."""
    m = re.search(r"_(\d+)\.zip$", path.name)
    return int(m.group(1)) if m else 0


def highest_checkpoint(run_dir: Path) -> Path | None:
    """Return the highest-step .zip in checkpoints/ (excluding _final)."""
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    zips = [z for z in ckpt_dir.glob("*.zip") if "_final" not in z.name]
    if not zips:
        return None
    return max(zips, key=_step_from_filename)


# ---------------------------------------------------------------------------
# Run directory resolution
# ---------------------------------------------------------------------------

def resolve_run_dir(value: str, experiments: Path) -> Path:
    """
    Resolve a run directory from a folder name or path.

    Accepts:
      - Full path:   experiments/SAC/2026-02-25_024550_SSLDribbling-v0_seed0
      - Relative:    SAC/2026-02-25_024550_SSLDribbling-v0_seed0
      - Just name:   2026-02-25_024550_SSLDribbling-v0_seed0

    Raises ResumeError if not found or ambiguous.
    """
    # Try as a direct path (absolute or relative to cwd)
    candidate = Path(value)
    if candidate.is_dir():
        return candidate.resolve()

    # Try relative to project root (parent of experiments/)
    if not candidate.is_absolute():
        for base in [Path.cwd(), experiments.parent]:
            full = base / value
            if full.is_dir():
                return full.resolve()

    # Search by folder name across all algo dirs
    name = Path(value).name
    if not experiments.exists():
        raise ResumeError(f"No experiments/ directory found at {experiments}")

    matches = []
    for algo_dir in sorted(experiments.iterdir()):
        if not algo_dir.is_dir():
            continue
        candidate = algo_dir / name
        if candidate.is_dir():
            matches.append(candidate)

    if len(matches) == 1:
        return matches[0].resolve()
    elif len(matches) > 1:
        listing = "\n".join(f"  {m.parent.name}/{m.name}" for m in matches)
        raise ResumeError(
            f"Multiple runs match '{name}':\n{listing}\n"
            f"  Use the full path to disambiguate."
        )
    else:
        raise ResumeError(
            f"No run found matching '{value}'\n"
            f"  Searched: {experiments}/"
        )


# ---------------------------------------------------------------------------
# find_latest_run
# ---------------------------------------------------------------------------

def find_latest_run(
    experiments: Path,
    env_filter: str | None = None,
    algo_filter: str | None = None,
) -> ResumeResult:
    """
    Find the newest unfinished run under experiments/.

    Args:
        experiments:  path to experiments/ directory
        env_filter:   only match this env (or None for any)
        algo_filter:  only match this algo (or None for any)

    Returns:
        ResumeResult with run_dir, checkpoint, config, is_finished.

    Raises:
        ResumeError if no matching run is found.
    """
    if not experiments.exists():
        raise ResumeError(f"No experiments/ directory found at {experiments}")

    candidates = []

    for algo_dir in sorted(experiments.iterdir()):
        if not algo_dir.is_dir():
            continue
        for run_dir in sorted(algo_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if is_run_finished(run_dir):
                continue

            config = parse_command_txt(run_dir)
            if not config.get("env") or not config.get("algo"):
                continue

            if env_filter and config["env"] != env_filter:
                continue
            if algo_filter and config["algo"] != algo_filter:
                continue

            ckpt = highest_checkpoint(run_dir)
            if ckpt is None:
                continue

            candidates.append((run_dir, ckpt, config))

    if not candidates:
        parts = []
        if env_filter:
            parts.append(f"env={env_filter}")
        if algo_filter:
            parts.append(f"algo={algo_filter}")
        filter_msg = f" ({', '.join(parts)})" if parts else ""
        raise ResumeError(
            f"No unfinished runs found{filter_msg}\n"
            f"  Searched: {experiments}/"
        )

    # Sort by folder name (timestamp prefix → chronological), pick newest
    candidates.sort(key=lambda c: c[0].name)
    run_dir, ckpt, config = candidates[-1]

    return ResumeResult(
        run_dir=run_dir,
        checkpoint=ckpt,
        config=config,
        is_finished=False,
        start_step=_step_from_filename(ckpt),
    )


# ---------------------------------------------------------------------------
# resolve_explicit
# ---------------------------------------------------------------------------

def resolve_explicit(value: str, experiments: Path) -> ResumeResult:
    """
    Resolve an explicit run dir (by name or path), find its highest
    checkpoint, parse its config.

    Raises ResumeError on any failure.
    """
    run_dir = resolve_run_dir(value, experiments)
    config = parse_command_txt(run_dir)

    if not config:
        raise ResumeError(
            f"No command.txt found in {run_dir}\n"
            f"  Cannot infer training settings."
        )

    validate_config(config, run_dir)

    ckpt = highest_checkpoint(run_dir)
    if ckpt is None:
        raise ResumeError(f"No checkpoints found in {run_dir / 'checkpoints'}")

    return ResumeResult(
        run_dir=run_dir,
        checkpoint=ckpt,
        config=config,
        is_finished=is_run_finished(run_dir),
        start_step=_step_from_filename(ckpt),
    )


# ---------------------------------------------------------------------------
# apply_config — merge resolved config into argparse namespace
# ---------------------------------------------------------------------------

def apply_config(args, config: dict) -> None:
    """
    Apply all values from a parsed command.txt config onto an argparse
    namespace. Overwrites existing values — the original run's settings
    take precedence over CLI defaults.
    """
    for key, value in config.items():
        attr = key.replace("-", "_")
        setattr(args, attr, value)
