"""
Tests for src/resume.py — all pure filesystem operations.

Run: pytest tests/test_resume.py -v
"""
import pytest
from pathlib import Path

# We add src/ to path so we can import resume directly
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from resume import (
    ResumeError,
    apply_config,
    find_latest_run,
    highest_checkpoint,
    is_run_finished,
    parse_command_txt,
    resolve_explicit,
    resolve_run_dir,
    validate_config,
)


# ---------------------------------------------------------------------------
# Fixtures — build fake experiment trees
# ---------------------------------------------------------------------------

SAMPLE_COMMAND = (
    "python src/train.py --env SSLDribbling-v0 --algo sac --seed 0 "
    "--timesteps 10000000 --save-freq 100000 --max-checkpoints 5 --n-envs 4\n"
)


def make_run(experiments: Path, algo: str, name: str,
             command_txt: str = SAMPLE_COMMAND,
             checkpoints: list[str] | None = None,
             final: bool = False) -> Path:
    """Create a fake run directory with optional checkpoints."""
    run_dir = experiments / algo.upper() / name
    run_dir.mkdir(parents=True, exist_ok=True)

    if command_txt:
        (run_dir / "command.txt").write_text(command_txt)

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    for ckpt in (checkpoints or []):
        (ckpt_dir / ckpt).write_bytes(b"fake zip data")

    if final:
        (ckpt_dir / "sac_ssldribbling_v0_final.zip").write_bytes(b"fake final")

    return run_dir


@pytest.fixture
def experiments(tmp_path):
    """Return an empty experiments/ directory."""
    exp = tmp_path / "experiments"
    exp.mkdir()
    return exp


# ===========================================================================
# parse_command_txt
# ===========================================================================

class TestParseCommandTxt:

    def test_parses_all_fields(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "command.txt").write_text(SAMPLE_COMMAND)

        config = parse_command_txt(run_dir)
        assert config == {
            "env": "SSLDribbling-v0",
            "algo": "sac",
            "seed": 0,
            "timesteps": 10000000,
            "save_freq": 100000,
            "max_checkpoints": 5,
            "n_envs": 4,
        }

    def test_returns_empty_dict_when_no_file(self, tmp_path):
        assert parse_command_txt(tmp_path) == {}

    def test_partial_command(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "command.txt").write_text("python src/train.py --env VSS-v0 --algo ppo\n")

        config = parse_command_txt(run_dir)
        assert config["env"] == "VSS-v0"
        assert config["algo"] == "ppo"
        assert "timesteps" not in config

    def test_types_are_correct(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "command.txt").write_text(SAMPLE_COMMAND)

        config = parse_command_txt(run_dir)
        assert isinstance(config["env"], str)
        assert isinstance(config["algo"], str)
        assert isinstance(config["timesteps"], int)
        assert isinstance(config["save_freq"], int)
        assert isinstance(config["seed"], int)
        assert isinstance(config["max_checkpoints"], int)
        assert isinstance(config["n_envs"], int)


# ===========================================================================
# validate_config
# ===========================================================================

class TestValidateConfig:

    def test_passes_with_required_keys(self, tmp_path):
        validate_config({"env": "VSS-v0", "algo": "sac"}, tmp_path)

    def test_raises_when_env_missing(self, tmp_path):
        with pytest.raises(ResumeError, match="--env"):
            validate_config({"algo": "sac"}, tmp_path)

    def test_raises_when_algo_missing(self, tmp_path):
        with pytest.raises(ResumeError, match="--algo"):
            validate_config({"env": "VSS-v0"}, tmp_path)

    def test_raises_when_both_missing(self, tmp_path):
        with pytest.raises(ResumeError):
            validate_config({}, tmp_path)


# ===========================================================================
# is_run_finished
# ===========================================================================

class TestIsRunFinished:

    def test_not_finished_no_checkpoints_dir(self, tmp_path):
        assert is_run_finished(tmp_path) is False

    def test_not_finished_no_final(self, experiments):
        run = make_run(experiments, "sac", "run1",
                       checkpoints=["sac_SSLDribbling-v0_00100000.zip"])
        assert is_run_finished(run) is False

    def test_finished_with_final(self, experiments):
        run = make_run(experiments, "sac", "run1",
                       checkpoints=["sac_SSLDribbling-v0_00100000.zip"],
                       final=True)
        assert is_run_finished(run) is True

    def test_finished_only_final(self, experiments):
        run = make_run(experiments, "sac", "run1", final=True)
        assert is_run_finished(run) is True


# ===========================================================================
# highest_checkpoint
# ===========================================================================

class TestHighestCheckpoint:

    def test_returns_none_no_dir(self, tmp_path):
        assert highest_checkpoint(tmp_path) is None

    def test_returns_none_empty_dir(self, experiments):
        run = make_run(experiments, "sac", "run1")
        assert highest_checkpoint(run) is None

    def test_returns_none_only_final(self, experiments):
        run = make_run(experiments, "sac", "run1", final=True)
        assert highest_checkpoint(run) is None

    def test_picks_highest_step(self, experiments):
        run = make_run(experiments, "sac", "run1", checkpoints=[
            "sac_SSLDribbling-v0_00100000.zip",
            "sac_SSLDribbling-v0_00300000.zip",
            "sac_SSLDribbling-v0_00200000.zip",
        ])
        best = highest_checkpoint(run)
        assert best.name == "sac_SSLDribbling-v0_00300000.zip"

    def test_ignores_final(self, experiments):
        run = make_run(experiments, "sac", "run1",
                       checkpoints=[
                           "sac_SSLDribbling-v0_00100000.zip",
                           "sac_SSLDribbling-v0_00200000.zip",
                       ],
                       final=True)
        best = highest_checkpoint(run)
        assert best.name == "sac_SSLDribbling-v0_00200000.zip"

    def test_single_checkpoint(self, experiments):
        run = make_run(experiments, "sac", "run1",
                       checkpoints=["sac_SSLDribbling-v0_00100000.zip"])
        best = highest_checkpoint(run)
        assert best.name == "sac_SSLDribbling-v0_00100000.zip"


# ===========================================================================
# resolve_run_dir
# ===========================================================================

class TestResolveRunDir:

    def test_by_folder_name(self, experiments):
        name = "2026-02-25_024550_SSLDribbling-v0_seed0"
        make_run(experiments, "sac", name)
        result = resolve_run_dir(name, experiments)
        assert result.name == name

    def test_by_full_path(self, experiments):
        name = "2026-02-25_024550_SSLDribbling-v0_seed0"
        run = make_run(experiments, "sac", name)
        result = resolve_run_dir(str(run), experiments)
        assert result == run.resolve()

    def test_by_relative_path(self, experiments):
        name = "2026-02-25_024550_SSLDribbling-v0_seed0"
        make_run(experiments, "sac", name)
        result = resolve_run_dir(f"experiments/SAC/{name}", experiments)
        assert result.name == name

    def test_not_found_raises(self, experiments):
        make_run(experiments, "sac", "some_run")
        with pytest.raises(ResumeError, match="No run found"):
            resolve_run_dir("nonexistent_run", experiments)

    def test_no_experiments_dir_raises(self, tmp_path):
        fake_exp = tmp_path / "experiments"
        with pytest.raises(ResumeError, match="No experiments/ directory"):
            resolve_run_dir("anything", fake_exp)

    def test_ambiguous_raises(self, experiments):
        """Same folder name under two different algo dirs."""
        name = "2026-02-25_024550_SSLDribbling-v0_seed0"
        make_run(experiments, "sac", name)
        make_run(experiments, "ddpg", name)
        with pytest.raises(ResumeError, match="Multiple runs"):
            resolve_run_dir(name, experiments)

    def test_disambiguate_with_full_path(self, experiments):
        """Full path resolves even when name is ambiguous."""
        name = "2026-02-25_024550_SSLDribbling-v0_seed0"
        make_run(experiments, "sac", name)
        run_ddpg = make_run(experiments, "ddpg", name)
        result = resolve_run_dir(str(run_ddpg), experiments)
        assert result == run_ddpg.resolve()

    def test_finds_dir_without_command_txt(self, experiments):
        """resolve_run_dir finds dirs even without command.txt;
        downstream callers (resolve_explicit) check for command.txt."""
        name = "2026-02-25_024550_SSLDribbling-v0_seed0"
        run_dir = experiments / "SAC" / name
        run_dir.mkdir(parents=True)
        result = resolve_run_dir(name, experiments)
        assert result == run_dir.resolve()


# ===========================================================================
# find_latest_run
# ===========================================================================

class TestFindLatestRun:

    def test_finds_newest_unfinished(self, experiments):
        make_run(experiments, "sac", "2026-02-24_100000_SSLDribbling-v0_seed0",
                 checkpoints=["sac_SSLDribbling-v0_00100000.zip"])
        make_run(experiments, "sac", "2026-02-25_100000_SSLDribbling-v0_seed0",
                 checkpoints=["sac_SSLDribbling-v0_00200000.zip"])

        result = find_latest_run(experiments)
        assert "2026-02-25" in result.run_dir.name
        assert result.checkpoint.name == "sac_SSLDribbling-v0_00200000.zip"
        assert result.is_finished is False

    def test_skips_finished_runs(self, experiments):
        make_run(experiments, "sac", "2026-02-25_100000_SSLDribbling-v0_seed0",
                 checkpoints=["sac_SSLDribbling-v0_00200000.zip"],
                 final=True)
        make_run(experiments, "sac", "2026-02-24_100000_SSLDribbling-v0_seed0",
                 checkpoints=["sac_SSLDribbling-v0_00100000.zip"])

        result = find_latest_run(experiments)
        assert "2026-02-24" in result.run_dir.name

    def test_filter_by_env(self, experiments):
        make_run(experiments, "sac", "2026-02-25_100000_VSS-v0_seed0",
                 command_txt="python src/train.py --env VSS-v0 --algo sac --timesteps 5000000 --save-freq 100000\n",
                 checkpoints=["sac_VSS-v0_00200000.zip"])
        make_run(experiments, "sac", "2026-02-25_200000_SSLDribbling-v0_seed0",
                 checkpoints=["sac_SSLDribbling-v0_00300000.zip"])

        result = find_latest_run(experiments, env_filter="VSS-v0")
        assert "VSS-v0" in result.run_dir.name

    def test_filter_by_algo(self, experiments):
        make_run(experiments, "sac", "2026-02-25_100000_SSLDribbling-v0_seed0",
                 checkpoints=["sac_SSLDribbling-v0_00200000.zip"])
        cmd_ddpg = SAMPLE_COMMAND.replace("--algo sac", "--algo ddpg")
        make_run(experiments, "ddpg", "2026-02-25_200000_SSLDribbling-v0_seed0",
                 command_txt=cmd_ddpg,
                 checkpoints=["ddpg_SSLDribbling-v0_00300000.zip"])

        result = find_latest_run(experiments, algo_filter="ddpg")
        assert result.config["algo"] == "ddpg"

    def test_filter_by_env_and_algo(self, experiments):
        make_run(experiments, "sac", "2026-02-25_100000_SSLDribbling-v0_seed0",
                 checkpoints=["sac_SSLDribbling-v0_00200000.zip"])
        cmd_ppo_vss = "python src/train.py --env VSS-v0 --algo ppo --timesteps 5000000 --save-freq 100000\n"
        make_run(experiments, "ppo", "2026-02-25_200000_VSS-v0_seed0",
                 command_txt=cmd_ppo_vss,
                 checkpoints=["ppo_VSS-v0_00300000.zip"])

        result = find_latest_run(experiments, env_filter="VSS-v0", algo_filter="ppo")
        assert result.config["env"] == "VSS-v0"
        assert result.config["algo"] == "ppo"

    def test_no_unfinished_raises(self, experiments):
        make_run(experiments, "sac", "2026-02-25_100000_SSLDribbling-v0_seed0",
                 checkpoints=["sac_SSLDribbling-v0_00200000.zip"],
                 final=True)
        with pytest.raises(ResumeError, match="No unfinished runs"):
            find_latest_run(experiments)

    def test_no_experiments_dir_raises(self, tmp_path):
        with pytest.raises(ResumeError, match="No experiments/ directory"):
            find_latest_run(tmp_path / "experiments")

    def test_skips_runs_without_checkpoints(self, experiments):
        make_run(experiments, "sac", "2026-02-25_100000_SSLDribbling-v0_seed0")
        make_run(experiments, "sac", "2026-02-24_100000_SSLDribbling-v0_seed0",
                 checkpoints=["sac_SSLDribbling-v0_00100000.zip"])

        result = find_latest_run(experiments)
        assert "2026-02-24" in result.run_dir.name

    def test_skips_runs_without_command_txt(self, experiments):
        make_run(experiments, "sac", "2026-02-25_100000_SSLDribbling-v0_seed0",
                 command_txt=None,
                 checkpoints=["sac_SSLDribbling-v0_00200000.zip"])
        make_run(experiments, "sac", "2026-02-24_100000_SSLDribbling-v0_seed0",
                 checkpoints=["sac_SSLDribbling-v0_00100000.zip"])

        result = find_latest_run(experiments)
        assert "2026-02-24" in result.run_dir.name

    def test_no_filter_match_raises(self, experiments):
        make_run(experiments, "sac", "2026-02-25_100000_SSLDribbling-v0_seed0",
                 checkpoints=["sac_SSLDribbling-v0_00200000.zip"])
        with pytest.raises(ResumeError, match="env=VSS-v0"):
            find_latest_run(experiments, env_filter="VSS-v0")

    def test_picks_highest_checkpoint_within_run(self, experiments):
        make_run(experiments, "sac", "2026-02-25_100000_SSLDribbling-v0_seed0",
                 checkpoints=[
                     "sac_SSLDribbling-v0_00100000.zip",
                     "sac_SSLDribbling-v0_00300000.zip",
                     "sac_SSLDribbling-v0_00200000.zip",
                 ])
        result = find_latest_run(experiments)
        assert result.checkpoint.name == "sac_SSLDribbling-v0_00300000.zip"

    def test_returns_start_step_from_checkpoint_filename(self, experiments):
        """ResumeResult.start_step is derived from the checkpoint filename."""
        make_run(experiments, "sac", "2026-02-25_100000_SSLDribbling-v0_seed0",
                 checkpoints=["sac_SSLDribbling-v0_00200000.zip"])
        result = find_latest_run(experiments)
        assert result.start_step == 200_000


# ===========================================================================
# resolve_explicit
# ===========================================================================

class TestResolveExplicit:

    def test_resolves_by_name(self, experiments):
        name = "2026-02-25_024550_SSLDribbling-v0_seed0"
        make_run(experiments, "sac", name,
                 checkpoints=["sac_SSLDribbling-v0_00200000.zip"])

        result = resolve_explicit(name, experiments)
        assert result.run_dir.name == name
        assert result.checkpoint.name == "sac_SSLDribbling-v0_00200000.zip"
        assert result.config["env"] == "SSLDribbling-v0"
        assert result.is_finished is False

    def test_detects_finished_run(self, experiments):
        name = "2026-02-25_024550_SSLDribbling-v0_seed0"
        make_run(experiments, "sac", name,
                 checkpoints=["sac_SSLDribbling-v0_00200000.zip"],
                 final=True)

        result = resolve_explicit(name, experiments)
        assert result.is_finished is True

    def test_no_command_txt_raises(self, experiments):
        name = "2026-02-25_024550_SSLDribbling-v0_seed0"
        make_run(experiments, "sac", name,
                 command_txt=None,
                 checkpoints=["sac_SSLDribbling-v0_00200000.zip"])

        with pytest.raises(ResumeError, match="No command.txt"):
            resolve_explicit(name, experiments)

    def test_missing_required_keys_raises(self, experiments):
        name = "2026-02-25_024550_SSLDribbling-v0_seed0"
        make_run(experiments, "sac", name,
                 command_txt="python src/train.py --timesteps 500000\n",
                 checkpoints=["sac_SSLDribbling-v0_00200000.zip"])

        with pytest.raises(ResumeError, match="missing required"):
            resolve_explicit(name, experiments)

    def test_no_checkpoints_raises(self, experiments):
        name = "2026-02-25_024550_SSLDribbling-v0_seed0"
        make_run(experiments, "sac", name)

        with pytest.raises(ResumeError, match="No checkpoints"):
            resolve_explicit(name, experiments)

    def test_not_found_raises(self, experiments):
        with pytest.raises(ResumeError, match="No run found"):
            resolve_explicit("nonexistent", experiments)

    def test_inherits_all_config(self, experiments):
        name = "2026-02-25_024550_SSLDribbling-v0_seed0"
        make_run(experiments, "sac", name,
                 checkpoints=["sac_SSLDribbling-v0_00200000.zip"])

        result = resolve_explicit(name, experiments)
        assert result.config["env"] == "SSLDribbling-v0"
        assert result.config["algo"] == "sac"
        assert result.config["timesteps"] == 10000000
        assert result.config["save_freq"] == 100000
        assert result.config["seed"] == 0
        assert result.config["max_checkpoints"] == 5
        assert result.config["n_envs"] == 4

    def test_returns_start_step_from_checkpoint_filename(self, experiments):
        """ResumeResult.start_step is derived from the checkpoint filename."""
        name = "2026-02-25_024550_SSLDribbling-v0_seed0"
        make_run(experiments, "sac", name,
                 checkpoints=["sac_SSLDribbling-v0_00150000.zip"])
        result = resolve_explicit(name, experiments)
        assert result.start_step == 150_000


# ===========================================================================
# apply_config
# ===========================================================================

class TestApplyConfig:

    def test_applies_all_values(self):
        """All config values should overwrite the namespace."""
        import argparse
        args = argparse.Namespace(
            env=None, algo=None, timesteps=500_000,
            save_freq=100_000, seed=0, max_checkpoints=5, n_envs=4,
        )
        config = {
            "env": "VSS-v0",
            "algo": "ppo",
            "timesteps": 10000000,
            "save_freq": 200000,
            "seed": 42,
            "max_checkpoints": 3,
            "n_envs": 8,
        }
        apply_config(args, config)

        assert args.env == "VSS-v0"
        assert args.algo == "ppo"
        assert args.timesteps == 10000000
        assert args.save_freq == 200000
        assert args.seed == 42
        assert args.max_checkpoints == 3
        assert args.n_envs == 8

    def test_partial_config_only_overwrites_present_keys(self):
        import argparse
        args = argparse.Namespace(
            env=None, algo=None, timesteps=500_000,
            save_freq=100_000, seed=0,
        )
        config = {"env": "VSS-v0", "algo": "ddpg"}
        apply_config(args, config)

        assert args.env == "VSS-v0"
        assert args.algo == "ddpg"
        assert args.timesteps == 500_000  # unchanged
        assert args.seed == 0              # unchanged

    def test_does_not_add_extra_keys(self):
        import argparse
        args = argparse.Namespace(env=None, algo=None)
        config = {"env": "VSS-v0", "algo": "sac", "timesteps": 1000000}
        apply_config(args, config)

        assert args.env == "VSS-v0"
        assert args.timesteps == 1000000  # added